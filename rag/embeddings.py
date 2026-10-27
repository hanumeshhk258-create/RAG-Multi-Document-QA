import os
import logging

# Optimize CPU thread allocation to prevent throttling on container environments
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import threading

# Configure project-relative model cache directory for persistent deployment slug caching
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CACHE_DIR = os.path.abspath(os.environ.get("HF_HOME") or os.path.join(BASE_DIR, "model_cache"))
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", MODEL_CACHE_DIR)

_EMBED_INIT_LOCK = threading.Lock()

try:
    import torch
    torch.set_num_threads(1)
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
        # Attempt 1: Load strictly from local cache folder (0 network calls, sub-second latency)
        try:
            return CachedHuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=MODEL_CACHE_DIR,
                model_kwargs={"device": "cpu", "local_files_only": True},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as local_err:
            logger.info(f"Local model weights not found in '{MODEL_CACHE_DIR}' ({local_err}). Fetching model...")

        # Attempt 2: Fetch and cache in MODEL_CACHE_DIR
        try:
            return CachedHuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=MODEL_CACHE_DIR,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as err:
            logger.error(f"Error loading embedding model '{model_name}': {str(err)}")
            raise RuntimeError(f"Failed to load embedding model: {str(err)}")

