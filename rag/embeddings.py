import logging

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
    Modular design allows easy replacement with alternative embedding models in the future.
    
    Args:
        model_name: Hugging Face model identifier. Default is 'all-MiniLM-L6-v2'.
        
    Returns:
        CachedHuggingFaceEmbeddings instance compatible with LangChain.
    """
    try:
        # Load local Hugging Face SentenceTransformer model
        embedding_model = CachedHuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        return embedding_model
    except Exception as err:
        logger.error(f"Error loading embedding model '{model_name}': {str(err)}")
        raise RuntimeError(f"Failed to load embedding model: {str(err)}")

