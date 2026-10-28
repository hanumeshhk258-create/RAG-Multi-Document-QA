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

# Serverless fallback: if running on Vercel or read-only filesystem, redirect to /tmp/model_cache
if os.environ.get("VERCEL") or not os.access(BASE_DIR, os.W_OK):
    if not os.path.exists(os.path.join(MODEL_CACHE_DIR, "model.onnx")):
        MODEL_CACHE_DIR = "/tmp/model_cache"

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


from typing import Dict, List, Optional
import numpy as np
from langchain_core.embeddings import Embeddings

class FastOnnxEmbeddings(Embeddings):
    """
    Lightweight, ultra-low-memory embedding generator for 'all-MiniLM-L6-v2'.
    Uses ONNX Runtime and Rust tokenizers instead of PyTorch.
    Reduces memory from 546 MB to ~190 MB, ensuring rock-solid stability under 512MB container limits.
    """
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            self.tokenizer = Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            if tokenizer_path:
                try:
                    self.tokenizer.save(tokenizer_path)
                except Exception:
                    pass

        self.tokenizer.enable_padding(length=256)
        self.tokenizer.enable_truncation(max_length=256)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self._query_cache: Dict[str, List[float]] = {}
        self._max_cache_size = 2048

    def _encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        clean_texts = [str(t).strip() or " " for t in texts]
        encoded = self.tokenizer.encode_batch(clean_texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })
        token_embeddings = outs[0]
        input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), token_embeddings.shape).astype(float)
        pooled = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(input_mask_expanded.sum(1), 1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.maximum(norms, 1e-12)
        return normalized.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 16
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend(self._encode(batch))
        return results

    def embed_query(self, text: str) -> List[float]:
        norm_text = text.strip().lower()
        if norm_text in self._query_cache:
            return self._query_cache[norm_text]
        vec = self._encode([text])[0]
        if len(self._query_cache) >= self._max_cache_size:
            self._query_cache.clear()
        self._query_cache[norm_text] = vec
        return vec

    def clear_cache(self):
        self._query_cache.clear()


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


class GeminiEmbeddings(Embeddings):
    """
    Lightweight cloud embedding generator using Google GenAI SDK (gemini-embedding-001).
    Configured with output_dimensionality=384 to match all-MiniLM-L6-v2 without
    requiring local ONNX weights or PyTorch dependencies, ideal for Vercel Serverless.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-embedding-001"):
        from google import genai
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required to initialize GeminiEmbeddings.")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self._query_cache: Dict[str, List[float]] = {}
        self._max_cache_size = 2048

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from google.genai import types
        if not texts:
            return []
        clean_texts = [str(t).strip() or " " for t in texts]
        results = []
        batch_size = 32
        for i in range(0, len(clean_texts), batch_size):
            batch = clean_texts[i:i + batch_size]
            resp = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=384)
            )
            for emb in resp.embeddings:
                results.append(list(emb.values))
        return results

    def embed_query(self, text: str) -> List[float]:
        from google.genai import types
        norm = text.strip().lower()
        if norm in self._query_cache:
            return self._query_cache[norm]
        resp = self.client.models.embed_content(
            model=self.model_name,
            contents=text.strip() or " ",
            config=types.EmbedContentConfig(output_dimensionality=384)
        )
        vec = list(resp.embeddings[0].values)
        if len(self._query_cache) >= self._max_cache_size:
            self._query_cache.clear()
        self._query_cache[norm] = vec
        return vec

    def clear_cache(self):
        self._query_cache.clear()


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Loads and returns a high-performance 384-dimensional embedding model with query caching.
    Supports FastOnnxEmbeddings (default, <200MB RAM) and GeminiEmbeddings (cloud serverless),
    falling back to cached PyTorch SentenceTransformer if ONNX is unavailable.
    """
    with _EMBED_INIT_LOCK:
        # Check for explicit cloud embedding provider
        provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
        if provider in ("gemini", "google", "cloud"):
            logger.info("Initializing 384-dimensional GeminiEmbeddings provider.")
            return GeminiEmbeddings()

        # 1. Prioritize ultra-low-memory ONNX Runtime engine (190MB RAM vs 540MB PyTorch)
        onnx_model_path = os.path.join(MODEL_CACHE_DIR, "model.onnx")
        onnx_tokenizer_path = os.path.join(MODEL_CACHE_DIR, "tokenizer.json")
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            if os.path.exists(onnx_model_path):
                logger.info(f"Loaded ultra-low-memory FastOnnxEmbeddings from '{onnx_model_path}'.")
                return FastOnnxEmbeddings(onnx_model_path, onnx_tokenizer_path)
            else:
                logger.info("ONNX weights not found locally. Fetching 'model.onnx' (~90MB)...")
                import urllib.request
                import socket
                socket.setdefaulttimeout(60)
                url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
                tmp_path = onnx_model_path + ".tmp"
                urllib.request.urlretrieve(url, tmp_path)
                os.replace(tmp_path, onnx_model_path)
                return FastOnnxEmbeddings(onnx_model_path, onnx_tokenizer_path)
        except Exception as onnx_err:
            logger.info(f"FastOnnxEmbeddings unavailable ({onnx_err}).")
            # Cloud fallback: If Gemini API key is configured, fallback to GeminiEmbeddings (384-dim)
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                try:
                    logger.info("Falling back to 384-dimensional GeminiEmbeddings.")
                    return GeminiEmbeddings(api_key=gemini_key)
                except Exception as g_err:
                    logger.warning(f"GeminiEmbeddings fallback failed: {g_err}")
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
