import os
import json
import shutil
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

import hashlib

DEFAULT_VECTORSTORE_DIR = "vectorstore"
METADATA_FILE = "metadata.json"


def compute_file_hash(file_path: str) -> str:
    """Computes SHA-256 hash of a file for content-based uniqueness and duplicate protection."""
    sha256 = hashlib.sha256()
    try:
        if isinstance(file_path, str) and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    sha256.update(chunk)
            return sha256.hexdigest()
        return hashlib.sha256(str(file_path).encode('utf-8')).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute hash for {file_path}: {e}")
        return hashlib.sha256(os.path.basename(str(file_path)).encode('utf-8')).hexdigest()


def _save_metadata(
    folder_path: str,
    doc_names: List[str],
    chunk_count: int,
    documents_info: Optional[List[Dict[str, Any]]] = None,
    total_pages: Optional[int] = None,
    indexed_count: Optional[int] = None,
    pending_count: Optional[int] = None,
    error_count: Optional[int] = None
) -> None:
    """Saves comprehensive indexing metadata and document lifecycle registry to JSON file."""
    try:
        os.makedirs(folder_path, exist_ok=True)
        metadata_path = os.path.join(folder_path, METADATA_FILE)
        
        info_list = documents_info or []
        
        # Determine Ready documents
        ready_docs = [d.get("filename") for d in info_list if d.get("status") == "Ready"]
        if not ready_docs and doc_names and chunk_count > 0 and not any(d.get("status") for d in info_list):
            # Backwards compatibility: if no explicit statuses are set but chunks > 0, consider doc_names as Ready
            ready_docs = sorted(list(set(doc_names)))
            for d in info_list:
                if d.get("filename") in ready_docs:
                    d["status"] = "Ready"

        unique_indexed_docs = sorted(list(set(ready_docs)))
        
        # Calculate totals
        calc_pages = total_pages
        if calc_pages is None and info_list:
            calc_pages = sum(d.get("pages", 0) for d in info_list)
        elif calc_pages is None:
            calc_pages = 0

        tot_indexed = indexed_count if indexed_count is not None else len(unique_indexed_docs)
        tot_pending = pending_count if pending_count is not None else sum(1 for d in info_list if d.get("status") == "Pending")
        tot_error = error_count if error_count is not None else sum(1 for d in info_list if d.get("status") == "Error")

        metadata = {
            "indexed_documents": unique_indexed_docs,
            "total_documents": len(info_list) if info_list else len(doc_names or []),
            "total_indexed": tot_indexed,
            "total_pending": tot_pending,
            "total_error": tot_error,
            "total_pages": calc_pages,
            "total_chunks": chunk_count,
            "documents_info": info_list,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as err:
        logger.warning(f"Could not save vectorstore metadata: {str(err)}")


def load_vectorstore_metadata(folder_path: str = DEFAULT_VECTORSTORE_DIR, docs_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads saved vectorstore metadata and synchronizes with physical files on disk.
    
    Returns:
        Dict with keys: indexed_documents, total_documents, total_indexed, total_pending,
        total_error, total_pages, total_chunks, documents_info, last_updated.
    """
    metadata_path = os.path.join(folder_path, METADATA_FILE)
    data = {
        "indexed_documents": [],
        "total_documents": 0,
        "total_indexed": 0,
        "total_pending": 0,
        "total_error": 0,
        "total_pages": 0,
        "total_chunks": 0,
        "documents_info": [],
        "last_updated": None
    }
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                data.update(loaded)
        except Exception as err:
            logger.warning(f"Failed to read metadata file: {str(err)}")

    # Ensure documents_info items have standardized keys
    doc_info_list = data.get("documents_info", [])
    indexed_set = set(data.get("indexed_documents", []))
    
    for d in doc_info_list:
        fname = d.get("filename", "")
        if "status" not in d:
            d["status"] = "Ready" if fname in indexed_set and data.get("total_chunks", 0) > 0 else "Pending"
        if "document_id" not in d:
            d["document_id"] = f"doc_{d.get('file_hash', fname)[:12]}"
        if "pages" not in d:
            d["pages"] = 1
        if "chunks" not in d:
            d["chunks"] = 0
        if "error_message" not in d:
            d["error_message"] = None
        if "uploaded_at" not in d:
            d["uploaded_at"] = data.get("last_updated") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "indexed_at" not in d:
            d["indexed_at"] = data.get("last_updated") if d["status"] == "Ready" else None

    # Sync with physical documents directory if provided
    if docs_dir and os.path.exists(docs_dir):
        disk_files = set(f for f in sorted(os.listdir(docs_dir)) if f.lower().endswith('.pdf'))
        existing_filenames = set(d.get("filename") for d in doc_info_list)
        
        # Add new disk files as Pending
        for df in sorted(disk_files):
            if df not in existing_filenames:
                fpath = os.path.join(docs_dir, df)
                f_size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
                f_hash = compute_file_hash(fpath)
                doc_info_list.append({
                    "document_id": f"doc_{f_hash[:12]}",
                    "filename": df,
                    "file_hash": f_hash,
                    "status": "Pending",
                    "pages": 1,
                    "chunks": 0,
                    "file_size": f_size,
                    "size_kb": round(f_size / 1024, 1),
                    "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "indexed_at": None,
                    "error_message": None
                })
        
        # Remove metadata for deleted disk files
        doc_info_list = [d for d in doc_info_list if d.get("filename") in disk_files]

    # Recompute summary statistics
    if doc_info_list:
        ready_docs = [d.get("filename") for d in doc_info_list if d.get("status") == "Ready"]
        data["documents_info"] = doc_info_list
        data["indexed_documents"] = ready_docs
        data["total_documents"] = len(doc_info_list)
        data["total_indexed"] = len(ready_docs)
        data["total_pending"] = sum(1 for d in doc_info_list if d.get("status") == "Pending")
        data["total_error"] = sum(1 for d in doc_info_list if d.get("status") == "Error")
        data["total_pages"] = sum(d.get("pages", 0) for d in doc_info_list)
        data["total_chunks"] = sum(d.get("chunks", 0) for d in doc_info_list if d.get("status") == "Ready")
    else:
        indexed = data.get("indexed_documents", [])
        data["total_documents"] = data.get("total_documents") or len(indexed)
        data["total_indexed"] = data.get("total_indexed") or len(indexed)

    return data


def _save_metadata_dict(folder_path: str, data: Dict[str, Any]) -> bool:
    """Directly saves metadata dictionary to metadata.json."""
    try:
        os.makedirs(folder_path, exist_ok=True)
        metadata_path = os.path.join(folder_path, METADATA_FILE)
        data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as err:
        logger.warning(f"Could not save metadata dictionary: {err}")
        return False


def build_vectorstore(
    chunks: List[Document],
    embedding_model,
    doc_names: Optional[List[str]] = None,
    documents_info: Optional[List[Dict[str, Any]]] = None,
    total_pages: Optional[int] = None
) -> Optional[FAISS]:
    """
    Creates a FAISS vector database from document chunks and an embedding model.
    
    Args:
        chunks: List of chunked Document objects containing text and metadata.
        embedding_model: Loaded embedding model instance.
        doc_names: Optional list of document names included in the index.
        documents_info: Optional list of per-document metadata dicts.
        total_pages: Optional count of total pages across indexed documents.
        
    Returns:
        FAISS vectorstore instance or None if input chunks are empty.
    """
    if not chunks:
        logger.warning("No document chunks provided to build vector database.")
        return None

    try:
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        return vectorstore
    except Exception as err:
        logger.error(f"Error building FAISS vectorstore: {str(err)}")
        raise RuntimeError(f"Failed to build vector database: {str(err)}")


def save_vectorstore(
    vectorstore: FAISS,
    folder_path: str = DEFAULT_VECTORSTORE_DIR,
    doc_names: Optional[List[str]] = None,
    total_chunks: Optional[int] = None,
    documents_info: Optional[List[Dict[str, Any]]] = None,
    total_pages: Optional[int] = None
) -> bool:
    """
    Saves the FAISS vector index and metadata to a specified directory.
    
    Args:
        vectorstore: FAISS vectorstore instance.
        folder_path: Target directory path.
        doc_names: Optional list of indexed document names.
        total_chunks: Optional count of chunks in the vectorstore.
        documents_info: Optional list of document info dicts.
        total_pages: Optional count of total pages.
        
    Returns:
        True if successfully saved, False otherwise.
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
        vectorstore.save_local(folder_path)
        
        # Determine document names and chunks if not explicitly passed
        existing_meta = load_vectorstore_metadata(folder_path)
        doc_names = doc_names if doc_names is not None else existing_meta.get("indexed_documents", [])
        total_chunks = total_chunks if total_chunks is not None else existing_meta.get("total_chunks", 0)
        documents_info = documents_info if documents_info is not None else existing_meta.get("documents_info", [])
        total_pages = total_pages if total_pages is not None else existing_meta.get("total_pages", 0)

        _save_metadata(folder_path, doc_names, total_chunks, documents_info, total_pages)
        logger.info(f"Vector database and metadata saved to '{folder_path}'.")
        return True
    except Exception as err:
        logger.error(f"Failed to save vectorstore to '{folder_path}': {str(err)}")
        return False


def load_vectorstore(
    embedding_model,
    folder_path: str = DEFAULT_VECTORSTORE_DIR
) -> Optional[FAISS]:
    """
    Loads a persisted FAISS vector index from disk.
    
    Args:
        embedding_model: Embedding model required to initialize index query engine.
        folder_path: Directory path where index files are stored.
        
    Returns:
        FAISS vectorstore instance or None if not found.
    """
    if not os.path.exists(folder_path):
        logger.info(f"No vector database folder found at '{folder_path}'.")
        return None

    index_file = os.path.join(folder_path, "index.faiss")
    if not os.path.exists(index_file):
        logger.info(f"No FAISS index file found at '{index_file}'.")
        return None

    try:
        vectorstore = FAISS.load_local(
            folder_path=folder_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True  # Required by modern LangChain FAISS
        )
        return vectorstore
    except Exception as err:
        logger.error(f"Error loading vector database from '{folder_path}': {str(err)}")
        return None


def clear_vectorstore(folder_path: str = DEFAULT_VECTORSTORE_DIR) -> bool:
    """
    Clears and deletes the stored FAISS vector database directory.
    
    Args:
        folder_path: Directory path of the vector store.
        
    Returns:
        True if successfully cleared, False otherwise.
    """
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            os.makedirs(folder_path, exist_ok=True)
            _save_metadata(folder_path, [], 0, [], 0)
        logger.info(f"Cleared vector database at '{folder_path}'.")
        return True
    except Exception as err:
        logger.error(f"Error clearing vector database at '{folder_path}': {str(err)}")
        return False


def add_documents_to_vectorstore(
    vectorstore: FAISS,
    new_chunks: List[Document],
    new_doc_names: Optional[List[str]] = None,
    folder_path: str = DEFAULT_VECTORSTORE_DIR,
    documents_info: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """Adds new chunks to an existing FAISS vectorstore and updates metadata."""
    if not vectorstore or not new_chunks:
        return False
    try:
        vectorstore.add_documents(new_chunks)
        existing_meta = load_vectorstore_metadata(folder_path)
        existing_docs = set(existing_meta.get("indexed_documents", []))
        if new_doc_names:
            existing_docs.update(new_doc_names)
        total_chunks = existing_meta.get("total_chunks", 0) + len(new_chunks)
        save_vectorstore(
            vectorstore,
            folder_path=folder_path,
            doc_names=sorted(list(existing_docs)),
            total_chunks=total_chunks,
            documents_info=documents_info
        )
        return True
    except Exception as err:
        logger.error(f"Failed to add documents to vectorstore: {err}")
        return False



