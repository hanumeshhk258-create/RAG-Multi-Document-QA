import os
import gc
import logging

# Optimize CPU thread allocation to prevent throttling on container environments
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import threading

# Configure project-relative model cache directory for persistent deployment slug caching
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_hf = os.environ.get("HF_HOME")
if _env_hf:
    MODEL_CACHE_DIR = os.path.abspath(_env_hf) if os.path.isabs(_env_hf) else os.path.abspath(os.path.join(BASE_DIR, _env_hf))
else:
    MODEL_CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE_DIR
try:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
except Exception:
    pass

_EMBED_INIT_LOCK = threading.Lock()

try:
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
except Exception:
    pass

# Enable HuggingFace offline mode only if explicitly configured
if os.environ.get("OFFLINE_MODE", "").lower() in ("true", "1", "yes"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Set up logger
logger = logging.getLogger(__name__)

# Fallback-aware import for Hugging Face Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    logger.warning("langchain-huggingface not found, falling back to langchain-community.")
    from langchain_community.embeddings import HuggingFaceEmbeddings


from typing import Dict, List

class CachedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    Subclass of HuggingFaceEmbeddings that caches query embeddings in memory.
    Prevents recomputing the exact same query vector multiple times during multi-query,
    decomposition, or repeated searches.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._query_cache: Dict[str, List[float]] = {}
        self._max_cache_size = 2048

    def embed_query(self, text: str) -> List[float]:
        norm_text = text.strip().lower()
        if norm_text in self._query_cache:
            return self._query_cache[norm_text]
        vec = super().embed_query(text)
        if len(self._query_cache) >= self._max_cache_size:
            self._query_cache.clear()
        self._query_cache[norm_text] = vec
        return vec

    def clear_cache(self):
        self._query_cache.clear()


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Loads and returns a Hugging Face Sentence Transformer embedding model with query caching.
    Uses local_files_only=True when cached to eliminate HuggingFace Hub network checks, rate-limiting, and timeouts.
    """
    with _EMBED_INIT_LOCK:
        try:
            import torch
            torch.set_grad_enabled(False)
            torch.set_num_threads(1)
        except Exception:
            pass

        # Prioritize local caches: check MODEL_CACHE_DIR, env HF_HOME, then standard HuggingFace hub cache
        candidate_folders = []
        if os.path.exists(MODEL_CACHE_DIR):
            candidate_folders.append(MODEL_CACHE_DIR)
        env_hf = os.environ.get("HF_HOME")
        if env_hf and env_hf not in candidate_folders and os.path.exists(env_hf):
            candidate_folders.append(env_hf)
        candidate_folders.append(None)  # Standard ~/.cache/huggingface/hub

        for folder in candidate_folders:
            try:
                emb = CachedHuggingFaceEmbeddings(
                    model_name=model_name,
                    cache_folder=folder,
                    model_kwargs={"device": "cpu", "local_files_only": True},
                    encode_kwargs={"normalize_embeddings": True, "batch_size": 16}
                )
                logger.info(f"Loaded embedding model '{model_name}' locally from '{folder or 'default cache'}'.")
                gc.collect()
                return emb
            except Exception as local_err:
                logger.debug(f"Local weights not found in '{folder}': {local_err}")

        # Fallback: Fetch online and cache in MODEL_CACHE_DIR
        try:
            logger.info(f"Local model weights not found in candidate caches. Fetching '{model_name}'...")
            emb = CachedHuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=MODEL_CACHE_DIR,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 16}
            )
            gc.collect()
            return emb
        except Exception as err:
            logger.error(f"Error loading embedding model '{model_name}': {str(err)}")
            raise RuntimeError(f"Failed to load embedding model: {str(err)}")
