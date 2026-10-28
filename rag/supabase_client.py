import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import requests
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Supabase Environment Configuration Dynamic Accessors
def get_supabase_url() -> str:
    return os.environ.get("SUPABASE_URL", "").strip().rstrip("/")

def get_supabase_key() -> str:
    return (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.environ.get("SUPABASE_KEY", "").strip()
        or os.environ.get("SUPABASE_ANON_KEY", "").strip()
    )

def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()

def get_storage_bucket() -> str:
    return os.environ.get("SUPABASE_STORAGE_BUCKET", "rag-documents").strip()


def is_supabase_configured() -> bool:
    """Returns True if valid Supabase or Postgres configuration is available."""
    url = get_supabase_url()
    key = get_supabase_key()
    db = get_database_url()
    has_api = bool(url and key and url.startswith("http"))
    has_db = bool(db and db.startswith("postgres"))
    return has_api or has_db


def get_pg_connection():
    """Returns a direct psycopg2 database connection if DATABASE_URL is configured."""
    db_url = get_database_url()
    if not db_url:
        return None
    try:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(db_url, connect_timeout=10)
        return conn
    except Exception as e:
        logger.warning(f"[SUPABASE] Direct Postgres connection failed: {e}")
        return None


# =============================================================================
# SUPABASE STORAGE CLIENT (REST API)
# =============================================================================

def _get_storage_headers() -> Dict[str, str]:
    key = get_supabase_key()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}"
    }


def ensure_storage_bucket(bucket_name: Optional[str] = None) -> bool:
    """Ensures that the designated private storage bucket exists."""
    bucket_name = bucket_name or get_storage_bucket()
    url = get_supabase_url()
    key = get_supabase_key()
    if not (url and key):
        return False
    try:
        b_url = f"{url}/storage/v1/bucket/{bucket_name}"
        res = requests.get(b_url, headers=_get_storage_headers(), timeout=10)
        if res.status_code == 200:
            return True
        if res.status_code == 404:
            create_url = f"{url}/storage/v1/bucket"
            create_res = requests.post(
                create_url,
                headers=_get_storage_headers(),
                json={"id": bucket_name, "name": bucket_name, "public": False},
                timeout=10
            )
            return create_res.status_code in (200, 201)
        return False
    except Exception as e:
        logger.warning(f"[SUPABASE] Failed to check/create storage bucket '{bucket_name}': {e}")
        return False


def upload_pdf_bytes(filename: str, file_bytes: bytes, bucket_name: Optional[str] = None) -> bool:
    """Uploads PDF raw binary to Supabase Storage."""
    bucket_name = bucket_name or get_storage_bucket()
    url = get_supabase_url()
    key = get_supabase_key()
    if not (url and key):
        return False
    try:
        ensure_storage_bucket(bucket_name)
        obj_url = f"{url}/storage/v1/object/{bucket_name}/{filename}"
        headers = _get_storage_headers()
        headers["Content-Type"] = "application/pdf"
        headers["x-upsert"] = "true"
        res = requests.post(obj_url, headers=headers, data=file_bytes, timeout=30)
        if res.status_code in (200, 201):
            logger.info(f"[SUPABASE STORAGE] Successfully uploaded '{filename}' ({len(file_bytes)} bytes).")
            return True
        logger.warning(f"[SUPABASE STORAGE] Upload returned HTTP {res.status_code}: {res.text}")
        return False
    except Exception as e:
        logger.error(f"[SUPABASE STORAGE] Upload exception for '{filename}': {e}")
        return False


def download_pdf_bytes(filename: str, bucket_name: Optional[str] = None) -> Optional[bytes]:
    """Downloads PDF binary from Supabase Storage."""
    bucket_name = bucket_name or get_storage_bucket()
    url = get_supabase_url()
    key = get_supabase_key()
    if not (url and key):
        return None
    try:
        # Try authenticated download endpoint
        auth_url = f"{url}/storage/v1/object/authenticated/{bucket_name}/{filename}"
        res = requests.get(auth_url, headers=_get_storage_headers(), timeout=30)
        if res.status_code == 200:
            return res.content

        # Fallback to signing URL
        sign_url = f"{url}/storage/v1/object/sign/{bucket_name}/{filename}"
        sign_res = requests.post(sign_url, headers=_get_storage_headers(), json={"expiresIn": 300}, timeout=10)
        if sign_res.status_code == 200:
            signed_path = sign_res.json().get("signedURL")
            if signed_path:
                dl_url = f"{url}/storage/v1{signed_path}"
                dl_res = requests.get(dl_url, timeout=30)
                if dl_res.status_code == 200:
                    return dl_res.content
        logger.warning(f"[SUPABASE STORAGE] Download failed for '{filename}' (HTTP {res.status_code}).")
        return None
    except Exception as e:
        logger.error(f"[SUPABASE STORAGE] Download exception for '{filename}': {e}")
        return None


def delete_pdf_file(filename: str, bucket_name: Optional[str] = None) -> bool:
    """Deletes a PDF from Supabase Storage."""
    bucket_name = bucket_name or get_storage_bucket()
    url = get_supabase_url()
    key = get_supabase_key()
    if not (url and key):
        return False
    try:
        del_url = f"{url}/storage/v1/object/{bucket_name}"
        headers = _get_storage_headers()
        res = requests.delete(del_url, headers=headers, json={"prefixes": [filename]}, timeout=15)
        return res.status_code in (200, 204)
    except Exception as e:
        logger.warning(f"[SUPABASE STORAGE] Delete exception for '{filename}': {e}")
        return False


# =============================================================================
# SUPABASE POSTGRESQL & PGVECTOR ADAPTER
# =============================================================================

def init_supabase_schema():
    """Applies schema.sql migrations to ensure all pgvector tables and functions exist."""
    conn = get_pg_connection()
    if not conn:
        return False
    try:
        schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                sql = f.read()
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info("[SUPABASE] Successfully initialized pgvector schema and tables.")
            return True
        return False
    except Exception as e:
        logger.error(f"[SUPABASE] Schema initialization failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def db_save_document_metadata(doc_entry: Dict[str, Any]) -> bool:
    """Upserts document metadata into documents_metadata table."""
    conn = get_pg_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents_metadata (
                        document_id, filename, file_hash, status, pages, chunks, size_kb, file_size, uploaded_at, indexed_at, error_message
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (filename) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        file_hash = EXCLUDED.file_hash,
                        status = EXCLUDED.status,
                        pages = EXCLUDED.pages,
                        chunks = EXCLUDED.chunks,
                        size_kb = EXCLUDED.size_kb,
                        file_size = EXCLUDED.file_size,
                        indexed_at = EXCLUDED.indexed_at,
                        error_message = EXCLUDED.error_message;
                """, (
                    doc_entry.get("document_id"),
                    doc_entry.get("filename"),
                    doc_entry.get("file_hash"),
                    doc_entry.get("status", "Pending"),
                    doc_entry.get("pages", 0),
                    doc_entry.get("chunks", 0),
                    doc_entry.get("size_kb", 0.0),
                    doc_entry.get("file_size", 0),
                    doc_entry.get("uploaded_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    doc_entry.get("indexed_at"),
                    doc_entry.get("error_message")
                ))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"[SUPABASE] Metadata upsert failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    # REST PostgREST fallback
    s_url = get_supabase_url()
    s_key = get_supabase_key()
    if s_url and s_key:
        try:
            url = f"{s_url}/rest/v1/documents_metadata"
            headers = _get_storage_headers()
            headers["Prefer"] = "resolution=merge-duplicates"
            res = requests.post(url, headers=headers, json=doc_entry, timeout=10)
            return res.status_code in (200, 201, 204)
        except Exception as e:
            logger.error(f"[SUPABASE REST] Metadata upsert failed: {e}")
    return False


def db_load_all_metadata() -> Dict[str, Any]:
    """Loads all document metadata records and formats as standard registry dictionary."""
    data = {
        "indexed_documents": [],
        "total_documents": 0,
        "total_indexed": 0,
        "total_pending": 0,
        "total_error": 0,
        "total_pages": 0,
        "total_chunks": 0,
        "documents_info": [],
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    conn = get_pg_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT document_id, filename, file_hash, status, pages, chunks, size_kb, file_size, uploaded_at, indexed_at, error_message
                    FROM documents_metadata
                    ORDER BY uploaded_at DESC;
                """)
                rows = cur.fetchall()
                docs = []
                for r in rows:
                    docs.append({
                        "document_id": r[0],
                        "filename": r[1],
                        "file_hash": r[2],
                        "status": r[3],
                        "pages": r[4],
                        "chunks": r[5],
                        "size_kb": float(r[6] or 0),
                        "file_size": r[7],
                        "uploaded_at": r[8],
                        "indexed_at": r[9],
                        "error_message": r[10]
                    })
                ready_list = [d["filename"] for d in docs if d.get("status") == "Ready"]
                data["documents_info"] = docs
                data["total_documents"] = len(docs)
                data["indexed_documents"] = ready_list
                data["total_indexed"] = len(ready_list)
                data["total_pending"] = sum(1 for d in docs if d.get("status") == "Pending")
                data["total_error"] = sum(1 for d in docs if d.get("status") == "Error")
                data["total_pages"] = sum(d.get("pages", 0) for d in docs)
                data["total_chunks"] = sum(d.get("chunks", 0) for d in docs if d.get("status") == "Ready")
                return data
        except Exception as e:
            logger.warning(f"[SUPABASE] Metadata query failed: {e}")
        finally:
            conn.close()

    # REST PostgREST query
    s_url = get_supabase_url()
    s_key = get_supabase_key()
    if s_url and s_key:
        try:
            url = f"{s_url}/rest/v1/documents_metadata?select=*&order=uploaded_at.desc"
            res = requests.get(url, headers=_get_storage_headers(), timeout=10)
            if res.status_code == 200:
                docs = res.json()
                ready_list = [d["filename"] for d in docs if d.get("status") == "Ready"]
                data["documents_info"] = docs
                data["total_documents"] = len(docs)
                data["indexed_documents"] = ready_list
                data["total_indexed"] = len(ready_list)
                data["total_pending"] = sum(1 for d in docs if d.get("status") == "Pending")
                data["total_error"] = sum(1 for d in docs if d.get("status") == "Error")
                data["total_pages"] = sum(d.get("pages", 0) for d in docs)
                data["total_chunks"] = sum(d.get("chunks", 0) for d in docs if d.get("status") == "Ready")
                return data
        except Exception as e:
            logger.warning(f"[SUPABASE REST] Metadata query failed: {e}")

    return data


def db_save_chunks(document_id: str, document_name: str, chunks: List[Document], embeddings: List[List[float]]) -> bool:
    """Inserts document chunks and their 384-dimensional embeddings into document_chunks table."""
    conn = get_pg_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                # Remove stale chunks for document first
                cur.execute("DELETE FROM document_chunks WHERE document_id = %s;", (document_id,))
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    page_num = chunk.metadata.get("page", 1)
                    cur.execute("""
                        INSERT INTO document_chunks (
                            document_id, document_name, chunk_index, page_number, content, embedding, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """, (
                        document_id,
                        document_name,
                        idx,
                        page_num,
                        chunk.page_content,
                        f"[{','.join(str(v) for v in emb)}]",
                        json.dumps(chunk.metadata)
                    ))
            conn.commit()
            logger.info(f"[SUPABASE PGVECTOR] Saved {len(chunks)} chunks with 384-dim embeddings for '{document_name}'.")
            return True
        except Exception as e:
            logger.error(f"[SUPABASE PGVECTOR] Chunks insert failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    # REST PostgREST RPC / batch insert
    s_url = get_supabase_url()
    s_key = get_supabase_key()
    if s_url and s_key:
        try:
            # Delete stale
            del_url = f"{s_url}/rest/v1/document_chunks?document_id=eq.{document_id}"
            requests.delete(del_url, headers=_get_storage_headers(), timeout=10)

            # Insert batch
            ins_url = f"{s_url}/rest/v1/document_chunks"
            records = []
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                records.append({
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_index": idx,
                    "page_number": chunk.metadata.get("page", 1),
                    "content": chunk.page_content,
                    "embedding": f"[{','.join(str(v) for v in emb)}]",
                    "metadata": chunk.metadata
                })
            res = requests.post(ins_url, headers=_get_storage_headers(), json=records, timeout=30)
            return res.status_code in (200, 201)
        except Exception as e:
            logger.error(f"[SUPABASE REST] Chunks insert failed: {e}")

    return False


def db_similarity_search(query_vector: List[float], k: int = 5, filter_documents: Optional[List[str]] = None) -> List[Tuple[Document, float]]:
    """Executes dense vector cosine similarity search in Supabase pgvector."""
    results = []
    vector_str = f"[{','.join(str(v) for v in query_vector)}]"

    conn = get_pg_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                if filter_documents:
                    cur.execute("""
                        SELECT document_name, page_number, content, metadata, 1 - (embedding <=> %s::vector) AS similarity
                        FROM document_chunks
                        WHERE document_name = ANY(%s)
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """, (vector_str, filter_documents, vector_str, k))
                else:
                    cur.execute("""
                        SELECT document_name, page_number, content, metadata, 1 - (embedding <=> %s::vector) AS similarity
                        FROM document_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """, (vector_str, vector_str, k))
                rows = cur.fetchall()
                for r in rows:
                    meta = r[3] if isinstance(r[3], dict) else json.loads(r[3] or "{}")
                    meta["document"] = r[0]
                    meta["page"] = r[1]
                    doc = Document(page_content=r[2], metadata=meta)
                    similarity = float(r[4] or 0.0)
                    results.append((doc, similarity))
                return results
        except Exception as e:
            logger.warning(f"[SUPABASE PGVECTOR] Similarity search query failed: {e}")
        finally:
            conn.close()

    # REST PostgREST RPC call
    s_url = get_supabase_url()
    s_key = get_supabase_key()
    if s_url and s_key:
        try:
            rpc_url = f"{s_url}/rest/v1/rpc/match_chunks"
            payload = {
                "query_embedding": vector_str,
                "match_count": k,
                "filter_documents": filter_documents
            }
            res = requests.post(rpc_url, headers=_get_storage_headers(), json=payload, timeout=15)
            if res.status_code == 200:
                rows = res.json()
                for r in rows:
                    meta = r.get("metadata") or {}
                    meta["document"] = r.get("document_name")
                    meta["page"] = r.get("page_number", 1)
                    doc = Document(page_content=r.get("content", ""), metadata=meta)
                    results.append((doc, float(r.get("similarity", 0.0))))
                return results
        except Exception as e:
            logger.warning(f"[SUPABASE REST RPC] Vector search failed: {e}")

    return results


def db_lexical_search(query_text: str, k: int = 5, filter_documents: Optional[List[str]] = None) -> List[Tuple[Document, float]]:
    """Executes full-text keyword search in PostgreSQL for hybrid retrieval."""
    results = []
    conn = get_pg_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                if filter_documents:
                    cur.execute("""
                        SELECT document_name, page_number, content, metadata,
                               ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) AS rank
                        FROM document_chunks
                        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                          AND document_name = ANY(%s)
                        ORDER BY rank DESC
                        LIMIT %s;
                    """, (query_text, query_text, filter_documents, k))
                else:
                    cur.execute("""
                        SELECT document_name, page_number, content, metadata,
                               ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) AS rank
                        FROM document_chunks
                        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s;
                    """, (query_text, query_text, k))
                rows = cur.fetchall()
                for r in rows:
                    meta = r[3] if isinstance(r[3], dict) else json.loads(r[3] or "{}")
                    meta["document"] = r[0]
                    meta["page"] = r[1]
                    doc = Document(page_content=r[2], metadata=meta)
                    results.append((doc, float(r[4] or 0.0)))
                return results
        except Exception as e:
            logger.warning(f"[SUPABASE PG] Lexical search query failed: {e}")
        finally:
            conn.close()
    return results


def db_get_all_chunks(filter_documents: Optional[List[str]] = None) -> List[Document]:
    """Retrieves all chunk documents from Postgres to populate docstore for BM25 and fallback."""
    chunks = []
    conn = get_pg_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                if filter_documents:
                    cur.execute("""
                        SELECT document_name, page_number, content, metadata
                        FROM document_chunks
                        WHERE document_name = ANY(%s)
                        ORDER BY document_name, chunk_index;
                    """, (filter_documents,))
                else:
                    cur.execute("""
                        SELECT document_name, page_number, content, metadata
                        FROM document_chunks
                        ORDER BY document_name, chunk_index;
                    """)
                rows = cur.fetchall()
                for r in rows:
                    meta = r[3] if isinstance(r[3], dict) else json.loads(r[3] or "{}")
                    meta["document"] = r[0]
                    meta["page"] = r[1]
                    chunks.append(Document(page_content=r[2], metadata=meta))
                return chunks
        except Exception as e:
            logger.warning(f"[SUPABASE] Get all chunks query failed: {e}")
        finally:
            conn.close()

    # REST PostgREST fallback
    s_url = get_supabase_url()
    s_key = get_supabase_key()
    if s_url and s_key:
        try:
            url = f"{s_url}/rest/v1/document_chunks?select=document_name,page_number,content,metadata&order=chunk_index.asc"
            res = requests.get(url, headers=_get_storage_headers(), timeout=15)
            if res.status_code == 200:
                for r in res.json():
                    meta = r.get("metadata") or {}
                    meta["document"] = r.get("document_name")
                    meta["page"] = r.get("page_number", 1)
                    chunks.append(Document(page_content=r.get("content", ""), metadata=meta))
        except Exception as e:
            logger.warning(f"[SUPABASE REST] Get chunks failed: {e}")

    return chunks


def db_delete_document(filename: str, document_id: Optional[str] = None) -> bool:
    """Deletes document record and cascades chunks deletion in PostgreSQL."""
    delete_pdf_file(filename)
    conn = get_pg_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                if document_id:
                    cur.execute("DELETE FROM documents_metadata WHERE document_id = %s OR filename = %s;", (document_id, filename))
                    cur.execute("DELETE FROM document_chunks WHERE document_id = %s OR document_name = %s;", (document_id, filename))
                else:
                    cur.execute("DELETE FROM documents_metadata WHERE filename = %s;", (filename,))
                    cur.execute("DELETE FROM document_chunks WHERE document_name = %s;", (filename,))
            conn.commit()
            return True
        except Exception as e:
            logger.warning(f"[SUPABASE] Document deletion failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    s_url = get_supabase_url()
    s_key = get_supabase_key()
    if s_url and s_key:
        try:
            url = f"{s_url}/rest/v1/documents_metadata?filename=eq.{filename}"
            requests.delete(url, headers=_get_storage_headers(), timeout=10)
            url_chunks = f"{s_url}/rest/v1/document_chunks?document_name=eq.{filename}"
            requests.delete(url_chunks, headers=_get_storage_headers(), timeout=10)
            return True
        except Exception as e:
            logger.warning(f"[SUPABASE REST] Document delete failed: {e}")
    return False


def db_save_indexing_job(job_record: Dict[str, Any]) -> bool:
    """Saves indexing job state durably into indexing_jobs table."""
    conn = get_pg_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO indexing_jobs (
                        job_id, status, stage, progress, message, documents, indexed_count, failed_count, failed_details, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        stage = EXCLUDED.stage,
                        progress = EXCLUDED.progress,
                        message = EXCLUDED.message,
                        documents = EXCLUDED.documents,
                        indexed_count = EXCLUDED.indexed_count,
                        failed_count = EXCLUDED.failed_count,
                        failed_details = EXCLUDED.failed_details,
                        updated_at = EXCLUDED.updated_at;
                """, (
                    job_record.get("job_id"),
                    job_record.get("status", "indexing"),
                    job_record.get("stage", "starting"),
                    job_record.get("progress", 0),
                    job_record.get("message", ""),
                    json.dumps(job_record.get("documents", [])),
                    job_record.get("indexed_count", 0),
                    job_record.get("failed_count", 0),
                    json.dumps(job_record.get("failed_details", [])),
                    job_record.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    job_record.get("updated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                ))
            conn.commit()
            return True
        except Exception as e:
            logger.warning(f"[SUPABASE] Job record upsert failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    s_url = get_supabase_url()
    s_key = get_supabase_key()
    if s_url and s_key:
        try:
            url = f"{s_url}/rest/v1/indexing_jobs"
            headers = _get_storage_headers()
            headers["Prefer"] = "resolution=merge-duplicates"
            res = requests.post(url, headers=headers, json=job_record, timeout=10)
            return res.status_code in (200, 201, 204)
        except Exception as e:
            logger.warning(f"[SUPABASE REST] Job record save failed: {e}")
    return False


def db_get_indexing_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Loads indexing job state from PostgreSQL."""
    conn = get_pg_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT job_id, status, stage, progress, message, documents, indexed_count, failed_count, failed_details, created_at, updated_at
                    FROM indexing_jobs
                    WHERE job_id = %s;
                """, (job_id,))
                r = cur.fetchone()
                if r:
                    return {
                        "job_id": r[0],
                        "status": r[1],
                        "stage": r[2],
                        "progress": r[3],
                        "message": r[4],
                        "documents": r[5] if isinstance(r[5], list) else json.loads(r[5] or "[]"),
                        "indexed_count": r[6],
                        "failed_count": r[7],
                        "failed_details": r[8] if isinstance(r[8], list) else json.loads(r[8] or "[]"),
                        "created_at": r[9],
                        "updated_at": r[10]
                    }
        except Exception as e:
            logger.warning(f"[SUPABASE] Get job query failed: {e}")
        finally:
            conn.close()

    s_url = get_supabase_url()
    s_key = get_supabase_key()
    if s_url and s_key:
        try:
            url = f"{s_url}/rest/v1/indexing_jobs?job_id=eq.{job_id}"
            res = requests.get(url, headers=_get_storage_headers(), timeout=10)
            if res.status_code == 200:
                rows = res.json()
                if rows:
                    return rows[0]
        except Exception as e:
            logger.warning(f"[SUPABASE REST] Get job failed: {e}")
    return None
    return None

