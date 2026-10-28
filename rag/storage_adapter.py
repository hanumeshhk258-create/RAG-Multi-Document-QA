import os
import io
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from langchain_core.documents import Document

from rag.vectorstore import (
    load_vectorstore_metadata,
    _save_metadata_dict,
    build_vectorstore,
    save_vectorstore,
    load_vectorstore,
    clear_vectorstore,
    compute_file_hash,
    clear_vectorstore_metadata_cache
)
from rag.supabase_client import (
    is_supabase_configured,
    upload_pdf_bytes,
    download_pdf_bytes,
    delete_pdf_file,
    db_save_document_metadata,
    db_load_all_metadata,
    db_save_chunks,
    db_similarity_search,
    db_lexical_search,
    db_get_all_chunks,
    db_delete_document,
    db_save_indexing_job,
    db_get_indexing_job
)

logger = logging.getLogger(__name__)


class SupabaseDocstore:
    """Simulates LangChain FAISS docstore._dict interface backed by Supabase."""
    def __init__(self, filter_documents: Optional[List[str]] = None):
        self._filter_documents = filter_documents

    @property
    def _dict(self) -> Dict[str, Document]:
        chunks = db_get_all_chunks(filter_documents=self._filter_documents)
        out = {}
        for idx, doc in enumerate(chunks):
            doc_id = doc.metadata.get("chunk_id") or f"chunk_{idx}"
            out[doc_id] = doc
        return out


class SupabaseVectorStore:
    """
    Drop-in vector store adapter implementing LangChain FAISS interface
    using Supabase Postgres and pgvector for 384-dimensional dense vectors.
    """
    def __init__(self, embedding_model, filter_documents: Optional[List[str]] = None):
        self.embedding_model = embedding_model
        self.filter_documents = filter_documents
        self.docstore = SupabaseDocstore(filter_documents=filter_documents)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_documents: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        active_filter = filter_documents or self.filter_documents
        if not self.embedding_model:
            return []
        try:
            query_vector = self.embedding_model.embed_query(query)
            return db_similarity_search(query_vector=query_vector, k=k, filter_documents=active_filter)
        except Exception as e:
            logger.error(f"[SUPABASE VECTORSTORE] Search error: {e}")
            return []

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        scored = self.similarity_search_with_score(query=query, k=k)
        return [doc for doc, _ in scored]

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []
        try:
            texts = [d.page_content for d in documents]
            embeddings = self.embedding_model.embed_documents(texts)
            doc_id = documents[0].metadata.get("document_id") or "doc_unknown"
            doc_name = documents[0].metadata.get("document") or "unknown.pdf"
            db_save_chunks(document_id=doc_id, document_name=doc_name, chunks=documents, embeddings=embeddings)
            return [str(i) for i in range(len(documents))]
        except Exception as e:
            logger.error(f"[SUPABASE VECTORSTORE] add_documents error: {e}")
            return []


# =============================================================================
# STORAGE ADAPTER IMPLEMENTATIONS
# =============================================================================

class BaseStorageAdapter:
    def get_storage_type(self) -> str:
        raise NotImplementedError

    def save_document(self, filename: str, file_bytes: bytes, doc_entry: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def get_document_bytes(self, filename: str) -> Optional[bytes]:
        raise NotImplementedError

    def delete_document(self, filename: str, doc_id: Optional[str] = None) -> bool:
        raise NotImplementedError

    def load_metadata(self, active_indexing: Optional[Set[str]] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def save_metadata(self, meta: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def save_indexing_job(self, job_data: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def get_indexing_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def build_or_update_vectorstore(
        self,
        chunks: List[Document],
        embedding_model,
        doc_names: List[str],
        doc_info_map: Dict[str, Any],
        is_rebuild: bool = False
    ) -> Any:
        raise NotImplementedError

    def get_active_vectorstore(self, embedding_model) -> Any:
        raise NotImplementedError


class LocalStorageAdapter(BaseStorageAdapter):
    """Local filesystem storage adapter preserving Render and local development behavior."""
    def __init__(self, docs_dir: str, vectorstore_dir: str):
        self.docs_dir = os.path.abspath(docs_dir)
        self.vectorstore_dir = os.path.abspath(vectorstore_dir)
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        self.jobs_file = os.path.join(self.vectorstore_dir, "indexing_jobs.json")
        self._in_memory_jobs: Dict[str, Dict[str, Any]] = {}

    def get_storage_type(self) -> str:
        return "local"

    def save_document(self, filename: str, file_bytes: bytes, doc_entry: Dict[str, Any]) -> bool:
        try:
            file_path = os.path.join(self.docs_dir, filename)
            with open(file_path, "wb") as f_out:
                f_out.write(file_bytes)
            return True
        except Exception as e:
            logger.error(f"[LOCAL STORAGE] Failed to write '{filename}': {e}")
            return False

    def get_document_bytes(self, filename: str) -> Optional[bytes]:
        file_path = os.path.join(self.docs_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"[LOCAL STORAGE] Failed to read '{filename}': {e}")
        return None

    def delete_document(self, filename: str, doc_id: Optional[str] = None) -> bool:
        file_path = os.path.join(self.docs_dir, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                logger.warning(f"[LOCAL STORAGE] Failed to remove '{file_path}': {e}")
        return False

    def load_metadata(self, active_indexing: Optional[Set[str]] = None) -> Dict[str, Any]:
        return load_vectorstore_metadata(
            self.vectorstore_dir,
            docs_dir=self.docs_dir,
            active_indexing_filenames=active_indexing
        )

    def save_metadata(self, meta: Dict[str, Any]) -> bool:
        return _save_metadata_dict(self.vectorstore_dir, meta)

    def save_indexing_job(self, job_data: Dict[str, Any]) -> bool:
        job_id = job_data.get("job_id")
        if not job_id:
            return False
        self._in_memory_jobs[job_id] = job_data
        try:
            with open(self.jobs_file, "w", encoding="utf-8") as f:
                json.dump(self._in_memory_jobs, f, indent=2)
            return True
        except Exception:
            return False

    def get_indexing_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        if job_id in self._in_memory_jobs:
            return self._in_memory_jobs[job_id]
        if os.path.exists(self.jobs_file):
            try:
                with open(self.jobs_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    if isinstance(saved, dict) and job_id in saved:
                        self._in_memory_jobs.update(saved)
                        return saved[job_id]
            except Exception:
                pass
        return None

    def build_or_update_vectorstore(
        self,
        chunks: List[Document],
        embedding_model,
        doc_names: List[str],
        doc_info_map: Dict[str, Any],
        is_rebuild: bool = False
    ) -> Any:
        total_pages = sum(d.get("pages", 0) for d in doc_info_map.values())
        if is_rebuild or not os.path.exists(os.path.join(self.vectorstore_dir, "index.faiss")):
            vs = build_vectorstore(
                chunks=chunks,
                embedding_model=embedding_model,
                doc_names=doc_names,
                documents_info=list(doc_info_map.values()),
                total_pages=total_pages
            )
        else:
            vs = load_vectorstore(embedding_model, folder_path=self.vectorstore_dir)
            if vs:
                vs.add_documents(chunks)
            else:
                vs = build_vectorstore(
                    chunks=chunks,
                    embedding_model=embedding_model,
                    doc_names=doc_names,
                    documents_info=list(doc_info_map.values()),
                    total_pages=total_pages
                )

        if vs:
            save_vectorstore(
                vectorstore=vs,
                folder_path=self.vectorstore_dir,
                doc_names=doc_names,
                total_chunks=len(chunks),
                documents_info=list(doc_info_map.values()),
                total_pages=total_pages
            )
        return vs

    def get_active_vectorstore(self, embedding_model) -> Any:
        return load_vectorstore(embedding_model, folder_path=self.vectorstore_dir)


class SupabaseStorageAdapter(BaseStorageAdapter):
    """
    Cloud persistent storage adapter using Supabase Storage for PDFs
    and Supabase PostgreSQL with pgvector for 384-dim chunks, metadata, and jobs.
    """
    def __init__(self, temp_docs_dir: str):
        self.temp_docs_dir = os.path.abspath(temp_docs_dir)
        os.makedirs(self.temp_docs_dir, exist_ok=True)
        self._cached_meta: Optional[Dict[str, Any]] = None

    def get_storage_type(self) -> str:
        return "supabase"

    def save_document(self, filename: str, file_bytes: bytes, doc_entry: Dict[str, Any]) -> bool:
        # Cache locally in temporary storage for immediate processing
        temp_path = os.path.join(self.temp_docs_dir, filename)
        try:
            with open(temp_path, "wb") as f_out:
                f_out.write(file_bytes)
        except Exception as te:
            logger.warning(f"[SUPABASE ADAPTER] Temporary write failed: {te}")

        # Upload persistently to Supabase Storage
        upload_pdf_bytes(filename, file_bytes)

        # Upsert document metadata in Supabase PostgreSQL
        db_save_document_metadata(doc_entry)
        self._cached_meta = None
        return True

    def get_document_bytes(self, filename: str) -> Optional[bytes]:
        # Check local temporary cache first
        temp_path = os.path.join(self.temp_docs_dir, filename)
        if os.path.exists(temp_path):
            try:
                with open(temp_path, "rb") as f:
                    return f.read()
            except Exception:
                pass
        # Fetch from Supabase Storage
        data = download_pdf_bytes(filename)
        if data:
            try:
                with open(temp_path, "wb") as f_out:
                    f_out.write(data)
            except Exception:
                pass
        return data

    def delete_document(self, filename: str, doc_id: Optional[str] = None) -> bool:
        temp_path = os.path.join(self.temp_docs_dir, filename)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        res = db_delete_document(filename, doc_id)
        self._cached_meta = None
        return res

    def load_metadata(self, active_indexing: Optional[Set[str]] = None) -> Dict[str, Any]:
        meta = db_load_all_metadata()
        if active_indexing:
            for d in meta.get("documents_info", []):
                if d.get("filename") in active_indexing:
                    d["status"] = "Indexing"
        self._cached_meta = meta
        return meta

    def save_metadata(self, meta: Dict[str, Any]) -> bool:
        for d in meta.get("documents_info", []):
            db_save_document_metadata(d)
        self._cached_meta = meta
        return True

    def save_indexing_job(self, job_data: Dict[str, Any]) -> bool:
        return db_save_indexing_job(job_data)

    def get_indexing_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return db_get_indexing_job(job_id)

    def build_or_update_vectorstore(
        self,
        chunks: List[Document],
        embedding_model,
        doc_names: List[str],
        doc_info_map: Dict[str, Any],
        is_rebuild: bool = False
    ) -> Any:
        vs = SupabaseVectorStore(embedding_model=embedding_model, filter_documents=doc_names)
        if chunks:
            # Group chunks by document to insert
            by_doc = {}
            for c in chunks:
                dname = c.metadata.get("document", "doc.pdf")
                by_doc.setdefault(dname, []).append(c)

            for dname, doc_chunks in by_doc.items():
                texts = [c.page_content for c in doc_chunks]
                embeddings = embedding_model.embed_documents(texts)
                doc_id = (doc_info_map.get(dname) or {}).get("document_id") or f"doc_{dname}"
                db_save_chunks(document_id=doc_id, document_name=dname, chunks=doc_chunks, embeddings=embeddings)

        # Update metadata status
        for dname in doc_names:
            if dname in doc_info_map:
                doc_info_map[dname]["status"] = "Ready"
                db_save_document_metadata(doc_info_map[dname])

        self._cached_meta = None
        return vs

    def get_active_vectorstore(self, embedding_model) -> Any:
        meta = self.load_metadata()
        ready_docs = meta.get("indexed_documents", [])
        return SupabaseVectorStore(embedding_model=embedding_model, filter_documents=ready_docs)


class UnconfiguredStorageAdapter(BaseStorageAdapter):
    """
    Storage adapter active on Vercel when Supabase persistent cloud credentials are missing.
    Prevents silent fallback to ephemeral local /tmp storage in production.
    """
    def get_storage_type(self) -> str:
        return "unconfigured"

    def _raise_error(self):
        raise RuntimeError(
            "Persistent cloud storage is not configured in this Vercel deployment. "
            "Please configure SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and DATABASE_URL in Vercel Project Settings."
        )

    def save_document(self, filename: str, file_bytes: bytes, doc_entry: Dict[str, Any]) -> bool:
        self._raise_error()

    def get_document_bytes(self, filename: str) -> Optional[bytes]:
        self._raise_error()

    def delete_document(self, filename: str, doc_id: Optional[str] = None) -> bool:
        self._raise_error()

    def load_metadata(self, active_indexing: Optional[Set[str]] = None) -> Dict[str, Any]:
        return {
            "indexed_documents": [],
            "total_documents": 0,
            "total_indexed": 0,
            "total_pending": 0,
            "total_error": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "documents_info": [],
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "storage_error": "Persistent cloud storage (Supabase) is not configured in this Vercel deployment."
        }

    def save_metadata(self, meta: Dict[str, Any]) -> bool:
        self._raise_error()

    def save_indexing_job(self, job_data: Dict[str, Any]) -> bool:
        self._raise_error()

    def get_indexing_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return None

    def build_or_update_vectorstore(self, *args, **kwargs) -> Any:
        self._raise_error()

    def get_active_vectorstore(self, embedding_model) -> Any:
        return None


# =============================================================================
# SINGLETON STORAGE ADAPTER FACTORY
# =============================================================================

_STORAGE_ADAPTER_INSTANCE = None


def get_storage_adapter(docs_dir: str = "documents", vectorstore_dir: str = "vectorstore") -> BaseStorageAdapter:
    """Returns the storage adapter (Supabase if configured, Unconfigured in Vercel production without Supabase, or Local filesystem)."""
    global _STORAGE_ADAPTER_INSTANCE
    if is_supabase_configured():
        if _STORAGE_ADAPTER_INSTANCE is None or _STORAGE_ADAPTER_INSTANCE.get_storage_type() != "supabase":
            logger.info("[STORAGE] Supabase configuration detected. Initializing SupabaseStorageAdapter.")
            temp_dir = os.path.join("/tmp" if os.environ.get("VERCEL") else docs_dir, "temp_docs")
            _STORAGE_ADAPTER_INSTANCE = SupabaseStorageAdapter(temp_docs_dir=temp_dir)
        return _STORAGE_ADAPTER_INSTANCE

    # If running in Vercel production and Supabase is not configured, do NOT silently fall back to local /tmp
    if os.environ.get("VERCEL") and not os.environ.get("ALLOW_LOCAL_FALLBACK"):
        if _STORAGE_ADAPTER_INSTANCE is None or _STORAGE_ADAPTER_INSTANCE.get_storage_type() != "unconfigured":
            logger.warning("[STORAGE] Running in Vercel environment without Supabase configuration! Initializing UnconfiguredStorageAdapter.")
            _STORAGE_ADAPTER_INSTANCE = UnconfiguredStorageAdapter()
        return _STORAGE_ADAPTER_INSTANCE

    if _STORAGE_ADAPTER_INSTANCE is None or _STORAGE_ADAPTER_INSTANCE.get_storage_type() != "local":
        logger.info("[STORAGE] No Supabase configuration detected. Initializing LocalStorageAdapter.")
        _STORAGE_ADAPTER_INSTANCE = LocalStorageAdapter(docs_dir=docs_dir, vectorstore_dir=vectorstore_dir)
    return _STORAGE_ADAPTER_INSTANCE

