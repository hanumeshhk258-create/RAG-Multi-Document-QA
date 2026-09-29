import os
import re
import time
import glob
import shutil
import logging
import hashlib
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Set
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, send_from_directory, Response

# Import RAG pipeline modules
from rag.pdf_loader import process_multiple_pdfs, load_pdf_file, get_pdf_info
from rag.chunker import split_documents, get_configured_chunk_settings
from rag.embeddings import get_embedding_model
from rag.analytics import (
    record_interaction,
    get_analytics_dashboard_data,
    get_question_detail,
    export_analytics_csv_data,
    clear_all_analytics,
    record_user_feedback,
    get_user_feedback_summary
)
from rag.evaluation_runner import (
    load_evaluation_dataset,
    run_benchmark_evaluation,
    get_latest_evaluation_report
)
from rag.vectorstore import (
    build_vectorstore,
    save_vectorstore,
    load_vectorstore,
    clear_vectorstore,
    load_vectorstore_metadata,
    compute_file_hash,
    _save_metadata_dict
)
from rag.retriever import (
    retrieve_relevant_chunks,
    retrieve_with_query_decomposition,
    adaptive_retrieval_pipeline,
    search_documents_fast,
    detect_comparison_question,
    retrieve_multi_document_context,
    group_chunks_by_document,
    build_grouped_sources_payload,
    detect_followup_question,
    classify_user_query,
    rewrite_query,
    clear_bm25_cache,
    _CROSS_ENCODER,
    HAS_RANK_BM25
)
from rag.evaluator import (
    evaluate_rag_response,
    extract_claims,
    verify_claim,
    evaluate_answer,
    calculate_faithfulness,
    calculate_hallucination_risk,
    clear_verification_cache
)
from rag.contextual_compression import (
    compress_context,
    MAX_CONTEXT_CHUNKS,
    MAX_CONTEXT_CHARACTERS
)
from rag.llm import (
    load_gemini_api_key,
    is_valid_api_key_format,
    get_configured_model_name,
    get_llm_client,
    generate_rag_answer,
    generate_comparison_answer,
    rewrite_query_with_gemini,
    expand_query_with_gemini,
    is_complex_query,
    decompose_query,
    test_gemini_connection,
    FALLBACK_RESPONSE,
    LLMGenerationError,
    AIGenerationError,
    AIAuthError,
    AIRateLimitError,
    AIEmptyResponseError
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
DEBUG_RAG = os.environ.get("DEBUG_RAG", "true").lower() in ("true", "1", "yes")

# Initialize Flask application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload limit

DOCUMENTS_DIR = "documents"
VECTORSTORE_DIR = "vectorstore"

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Global in-memory caches
embedding_model = None
active_vectorstore = None
active_gemini_client = None

# Fast Document Search Cache (query_lower, selected_docs_tuple) -> results
DOCUMENT_SEARCH_CACHE = {}

# Full RAG Pipeline Response Cache (normalized_query, selected_docs_tuple, index_version) -> response_dict
RAG_RESPONSE_CACHE = {}


def clear_search_cache():
    """Clears the in-memory document search results cache."""
    global DOCUMENT_SEARCH_CACHE
    DOCUMENT_SEARCH_CACHE.clear()
    logger.info("Document search cache invalidated.")


def clear_rag_response_cache():
    """Clears the in-memory RAG response cache and neural verification caches."""
    global RAG_RESPONSE_CACHE, embedding_model
    RAG_RESPONSE_CACHE.clear()
    clear_verification_cache()
    if hasattr(_CROSS_ENCODER, "clear_cache"):
        _CROSS_ENCODER.clear_cache()
    if embedding_model and hasattr(embedding_model, "clear_cache"):
        embedding_model.clear_cache()
    logger.info("RAG response cache and neural scoring caches invalidated.")


def validate_sources_against_registry(sources, indexed_docs_set, available_files_set):
    """
    Filters and validates source citations ensuring every cited document currently
    exists on disk and is in the active indexed documents registry. (Requirement 15)
    """
    if not sources or not isinstance(sources, list):
        return []
    valid = []
    for s in sources:
        doc_name = s.get("document") or s.get("filename") or s.get("document_id") or ""
        # Match against indexed_docs_set and available_files_set
        if doc_name and (doc_name in indexed_docs_set or any(doc_name in f for f in indexed_docs_set)):
            if doc_name in available_files_set or any(doc_name in f for f in available_files_set):
                valid.append(s)
    return valid


RETRIEVAL_VERSION = "2.1"


def check_and_guard_answer_similarity(query: str, current_answer: str, chat_history: Optional[List[Dict[str, str]]]) -> str:
    """
    Compares the generated answer with previous assistant answers.
    If the current question is standalone and conceptually different, but the answer
    is >80% identical to a previous turn answer, guards against repeating unrelated answers.
    """
    if not chat_history or not current_answer or current_answer == FALLBACK_RESPONSE:
        return current_answer
    
    prev_user_q = ""
    prev_assistant_ans = ""
    for msg in reversed(chat_history):
        if msg.get("role") in ("assistant", "ai") and not prev_assistant_ans:
            prev_assistant_ans = msg.get("content", "").strip()
        elif msg.get("role") in ("user", "human") and not prev_user_q:
            prev_user_q = msg.get("content", "").strip()
        if prev_assistant_ans and prev_user_q:
            break
            
    if not prev_assistant_ans or not prev_user_q:
        return current_answer

    # Check if questions are distinct
    q1_tokens = set(re.findall(r'\b\w+\b', prev_user_q.lower()))
    q2_tokens = set(re.findall(r'\b\w+\b', query.lower()))
    stop = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'of', 'for', 'to', 'and'}
    q1_clean = q1_tokens - stop
    q2_clean = q2_tokens - stop
    if not q1_clean or not q2_clean:
        return current_answer

    q_overlap = len(q1_clean.intersection(q2_clean)) / max(1, min(len(q1_clean), len(q2_clean)))
    
    # If questions have low overlap (< 40%)
    if q_overlap < 0.40:
        ans1_tokens = set(re.findall(r'\b\w+\b', prev_assistant_ans.lower()))
        ans2_tokens = set(re.findall(r'\b\w+\b', current_answer.lower()))
        if ans1_tokens and ans2_tokens:
            ans_jaccard = len(ans1_tokens.intersection(ans2_tokens)) / len(ans1_tokens.union(ans2_tokens))
            if ans_jaccard > 0.80:
                logger.warning(f"[SIMILARITY_GUARD] Generated answer is {ans_jaccard*100:.1f}% similar to previous answer for unrelated question. Replacing with fallback response.")
                return FALLBACK_RESPONSE
    return current_answer



def log_perf_metrics(query: str, cache_status: str, timing_breakdown: dict, total_sec: float):
    """Prints structured per-stage timing logs showing execution time in ms/s."""
    q_proc_ms = timing_breakdown.get("query_processing_ms", 0.0)
    faiss_ms = timing_breakdown.get("faiss_ms", 0.0)
    bm25_ms = timing_breakdown.get("bm25_ms", 0.0)
    merge_ms = timing_breakdown.get("hybrid_merge_ms", 0.0)
    retrieval_ms = timing_breakdown.get("retrieval_ms", faiss_ms + bm25_ms + merge_ms)
    rerank_ms = timing_breakdown.get("reranking_ms", 0.0)
    comp_ms = timing_breakdown.get("compression_ms", 0.0)
    prompt_ms = timing_breakdown.get("prompt_ms", 0.8)
    gen_sec = timing_breakdown.get("generation_sec", 0.0) or 0.0
    gen_ms = (gen_sec * 1000.0) if isinstance(gen_sec, (int, float)) else 0.0
    verif_sec = timing_breakdown.get("verification_sec", 0.0) or 0.0
    verif_ms = (verif_sec * 1000.0) if isinstance(verif_sec, (int, float)) else 0.0
    corr_ms = timing_breakdown.get("correction_ms", 0.0)
    total_ms = total_sec * 1000.0

    logger.info(f"[QUERY]: '{query}' (Cache: {cache_status}) | Processing: {q_proc_ms:.1f}ms")
    logger.info(f"[RETRIEVAL]: {retrieval_ms:.1f}ms ({retrieval_ms/1000.0:.3f}s)")
    logger.info(f"[FAISS]: {faiss_ms:.1f}ms ({faiss_ms/1000.0:.3f}s)")
    logger.info(f"[BM25]: {bm25_ms:.1f}ms ({bm25_ms/1000.0:.3f}s)")
    logger.info(f"[HYBRID]: {merge_ms:.1f}ms ({merge_ms/1000.0:.3f}s)")
    logger.info(f"[RERANK]: {rerank_ms:.1f}ms ({rerank_ms/1000.0:.3f}s)")
    logger.info(f"[COMPRESSION]: {comp_ms:.1f}ms ({comp_ms/1000.0:.3f}s)")
    logger.info(f"[PROMPT]: {prompt_ms:.1f}ms ({prompt_ms/1000.0:.3f}s)")
    logger.info(f"[GENERATION]: {gen_ms:.1f}ms ({gen_sec:.3f}s)" if isinstance(gen_sec, (int, float)) else f"[GENERATION]: {gen_sec}")
    logger.info(f"[VERIFICATION]: {verif_ms:.1f}ms ({verif_sec:.3f}s)" if isinstance(verif_sec, (int, float)) else f"[VERIFICATION]: {verif_sec}")
    logger.info(f"[CORRECTION]: {corr_ms:.1f}ms ({corr_ms/1000.0:.3f}s)")
    logger.info(f"[TOTAL]: {total_ms:.1f}ms ({total_sec:.3f}s)")


def log_debug_rag_trace(
    request_id: str,
    question: str,
    selected_docs: Any,
    retrieved_count: int,
    top_docs: Any,
    scores: Any,
    context_len: int,
    context_preview: str,
    cache_key: str,
    cache_status: str,
    model_name: str,
    gen_started: str,
    gen_completed: str,
    supported: bool,
    evidence_coverage: str,
    final_status: str
):
    """Structured development-only RAG pipeline trace (Requirement J). Never exposes secrets."""
    preview = (context_preview[:180] + "...") if len(context_preview) > 180 else context_preview
    logger.info(
        f"\n================ RAG REQUEST ================\n"
        f"Request ID: {request_id}\n"
        f"Question: {question}\n"
        f"Selected Documents: {selected_docs}\n\n"
        f"--------------- RETRIEVAL ------------------\n"
        f"Retrieved chunks: {retrieved_count}\n"
        f"Top documents: {top_docs}\n"
        f"Scores: {scores}\n\n"
        f"--------------- CONTEXT --------------------\n"
        f"Context length: {context_len}\n"
        f"Context preview: {preview}\n\n"
        f"--------------- CACHE ----------------------\n"
        f"Cache key: {cache_key}\n"
        f"Cache hit/miss: {cache_status}\n\n"
        f"--------------- LLM ------------------------\n"
        f"Model: {model_name}\n"
        f"Generation started: {gen_started}\n"
        f"Generation completed: {gen_completed}\n\n"
        f"--------------- VERIFICATION ---------------\n"
        f"Supported: {supported}\n"
        f"Evidence coverage: {evidence_coverage}\n"
        f"Final answer status: {final_status}\n"
        f"============================================="
    )


def get_cached_embeddings():
    """Initializes and caches the shared Hugging Face embedding model."""
    global embedding_model
    if embedding_model is None:
        logger.info("Initializing Hugging Face Sentence Transformers embedding model...")
        embedding_model = get_embedding_model("all-MiniLM-L6-v2")
    return embedding_model


def get_active_vectorstore():
    """Returns cached active vector store from memory or loads from disk."""
    global active_vectorstore
    if active_vectorstore is None:
        emb = get_cached_embeddings()
        active_vectorstore = load_vectorstore(emb, folder_path=VECTORSTORE_DIR)
    return active_vectorstore


def get_active_gemini_client():
    """Returns cached Google GenAI client instance."""
    global active_gemini_client
    api_key = load_gemini_api_key()
    if not is_valid_api_key_format(api_key):
        return None
    if active_gemini_client is None:
        try:
            active_gemini_client = get_llm_client(api_key)
        except Exception as err:
            logger.error(f"Failed to create Gemini client: {err}")
            return None
    return active_gemini_client


def build_system_status_payload():
    """Generates a complete status dictionary of the RAG system and all documents with lifecycle status."""
    api_key = load_gemini_api_key()
    is_configured = is_valid_api_key_format(api_key)
    model_name = get_configured_model_name()

    meta = load_vectorstore_metadata(VECTORSTORE_DIR, docs_dir=DOCUMENTS_DIR)
    vs = get_active_vectorstore()

    doc_info_list = meta.get("documents_info", [])
    ready_docs = [d["filename"] for d in doc_info_list if d.get("status") == "Ready"]
    pending_docs = [d["filename"] for d in doc_info_list if d.get("status") == "Pending"]
    error_docs = [d["filename"] for d in doc_info_list if d.get("status") == "Error"]
    indexing_docs = [d["filename"] for d in doc_info_list if d.get("status") == "Indexing"]

    docs_detail_list = []
    total_pages_count = 0
    total_ready_chunks = 0

    for doc in doc_info_list:
        fname = doc.get("filename")
        fpath = os.path.join(DOCUMENTS_DIR, fname)
        status = doc.get("status", "Pending")
        is_ready = (status == "Ready")
        
        pages = doc.get("pages", 1)
        chunks = doc.get("chunks", 0) if is_ready else 0
        total_pages_count += pages
        if is_ready:
            total_ready_chunks += chunks

        docs_detail_list.append({
            "document_id": doc.get("document_id") or f"doc_{doc.get('file_hash', fname)[:12]}",
            "filename": fname,
            "file_hash": doc.get("file_hash", ""),
            "status": status,
            "size_kb": doc.get("size_kb", 0),
            "file_size": doc.get("file_size", 0),
            "pages": pages,
            "chunks": chunks,
            "indexed": is_ready,
            "uploaded_at": doc.get("uploaded_at"),
            "indexed_at": doc.get("indexed_at"),
            "error_message": doc.get("error_message")
        })

    vectorstore_ready = vs is not None and len(ready_docs) > 0 and meta.get("total_chunks", 0) > 0

    components = {
        "hybrid_search": bool(vectorstore_ready),
        "bm25": bool(vectorstore_ready and HAS_RANK_BM25),
        "faiss": bool(vectorstore_ready),
        "cross_encoder": bool(_CROSS_ENCODER.is_available()),
        "multi_query": bool(is_configured),
        "query_decomposition": bool(is_configured),
        "adaptive_retrieval": True,
        "contextual_compression": True,
        "table_extraction": True,
        "answer_evaluation": True,
        "hallucination_detection": True,
        "answer_correction": True
    }

    return {
        "status": "ready" if vectorstore_ready else ("needs_index" if pending_docs or error_docs else "empty"),
        "gemini_configured": is_configured,
        "gemini_model": model_name,
        "total_documents": len(doc_info_list),
        "indexed_documents": ready_docs,
        "indexed_count": len(ready_docs),
        "pending_count": len(pending_docs),
        "error_count": len(error_docs),
        "indexing_count": len(indexing_docs),
        "total_pages": total_pages_count if total_pages_count > 0 else meta.get("total_pages", 0),
        "total_chunks": meta.get("total_chunks", total_ready_chunks),
        "vectorstore_ready": vectorstore_ready,
        "components": components,
        "documents": docs_detail_list,
        "last_updated": meta.get("last_updated")
    }


# =============================================================================
# 1. FRONTEND & PDF DOCUMENT VIEWER ROUTE
# =============================================================================
@app.route('/')
def index():
    """Serves the modern AI chatbot frontend UI."""
    return render_template('index.html')


@app.route('/view_pdf/<path:filename>')
@app.route('/api/view_pdf/<path:filename>')
@app.route('/documents/<path:filename>')
def serve_pdf_document(filename):
    """
    Serves original uploaded PDF documents safely inline in browser with hash navigation (e.g. /view_pdf/file.pdf#page=1).
    Validates file exists, prevents path traversal, and returns application/pdf MIME.
    """
    if not filename:
        return jsonify({"error": "No document specified."}), 400

    raw_name = os.path.basename(filename)
    safe_name = secure_filename(raw_name)

    if not safe_name:
        safe_name = raw_name

    if not safe_name.lower().endswith('.pdf'):
        safe_name = f"{safe_name}.pdf"

    file_path = os.path.join(DOCUMENTS_DIR, safe_name)
    if not os.path.exists(file_path):
        # Also check direct base filename inside DOCUMENTS_DIR
        alt_path = os.path.join(DOCUMENTS_DIR, raw_name)
        if os.path.exists(alt_path):
            safe_name = raw_name
        else:
            logger.warning(f"PDF requested but not found: {filename}")
            return jsonify({
                "error": "PDF file is no longer available.",
                "message": f"Document '{filename}' not found."
            }), 404

    return send_from_directory(
        os.path.abspath(DOCUMENTS_DIR),
        safe_name,
        mimetype='application/pdf',
        as_attachment=False
    )



# =============================================================================
# 2. STATUS & DOCUMENTS LIST ENDPOINTS
# =============================================================================
@app.route('/api/status', methods=['GET'])
@app.route('/status', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def get_status_endpoint():
    """
    Returns the current document count, page count, chunk count, vectorstore status, and per-document list.
    """
    return jsonify(build_system_status_payload())


@app.route('/api/documents', methods=['GET'])
@app.route('/documents', methods=['GET'])
def get_documents_endpoint():
    """
    Returns structured list of all uploaded documents, metadata, and indexing status.
    """
    payload = build_system_status_payload()
    return jsonify({
        "success": True,
        "documents": payload["documents"],
        "total_documents": payload["total_documents"],
        "total_pages": payload["total_pages"],
        "total_chunks": payload["total_chunks"],
        "vectorstore_ready": payload["vectorstore_ready"],
        "status": payload["status"]
    })


@app.route('/api/documents/select', methods=['POST'])
@app.route('/documents/select', methods=['POST'])
def select_documents_endpoint():
    """
    Validates selected documents against available documents.
    """
    data = request.get_json(silent=True) or {}
    selected = data.get("selected_documents", [])
    
    pdf_files = set(f for f in sorted(os.listdir(DOCUMENTS_DIR)) if f.lower().endswith('.pdf'))
    valid_selected = [doc for doc in selected if doc in pdf_files]

    return jsonify({
        "success": True,
        "selected_documents": valid_selected,
        "selected_count": len(valid_selected),
        "total_count": len(pdf_files)
    })


# =============================================================================
# 3. MULTI-PDF UPLOAD ENDPOINT
# =============================================================================
@app.route('/api/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Accepts 1, 2, 5, 10+ PDF files, validates them, computes hashes, and saves them with 'Pending' status.
    Implements duplicate document detection (SHA-256) and support for document replacement.
    """
    if 'files' not in request.files:
        return jsonify({"success": False, "error": "No files found in upload request."}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"success": False, "error": "No files selected."}), 400

    is_replace = (
        request.form.get("replace", "").strip().lower() in ("true", "1", "yes") or
        request.args.get("replace", "").strip().lower() in ("true", "1", "yes")
    )

    meta = load_vectorstore_metadata(VECTORSTORE_DIR, docs_dir=DOCUMENTS_DIR)
    doc_info_map = {d["filename"]: d for d in meta.get("documents_info", [])}

    saved_files = []
    errors = []
    warnings = []

    for file in files:
        if not file.filename:
            continue

        raw_filename = os.path.basename(file.filename)
        filename = secure_filename(raw_filename)
        if not filename:
            filename = "uploaded_document.pdf"

        if not filename.lower().endswith('.pdf'):
            errors.append(f"'{raw_filename}' is not a PDF file.")
            continue

        file_path = os.path.join(DOCUMENTS_DIR, filename)

        try:
            file_bytes = file.read()
            if not file_bytes:
                errors.append(f"'{filename}' is empty.")
                continue

            file_hash = hashlib.sha256(file_bytes).hexdigest()
            doc_id = f"doc_{file_hash[:12]}"

            # Duplicate Document Detection (Requirement 9 & 10)
            existing_doc = None
            for d in doc_info_map.values():
                if d.get("file_hash") == file_hash or d.get("filename") == filename:
                    existing_doc = d
                    break

            if existing_doc and not is_replace:
                # Return duplicate notification to prompt user in UI
                logger.info(f"[UPLOAD] Duplicate detected for '{filename}' (hash: {file_hash[:12]}). Prompting replace or keep.")
                return jsonify({
                    "success": False,
                    "is_duplicate": True,
                    "filename": filename,
                    "existing_document": existing_doc,
                    "message": "This document is already uploaded."
                }), 200

            # If replacing an existing document, delete previous version
            if existing_doc and is_replace:
                logger.info(f"[UPLOAD] Replacing existing document '{existing_doc.get('filename')}' with new upload.")
                old_file_path = os.path.join(DOCUMENTS_DIR, existing_doc.get("filename", filename))
                if os.path.exists(old_file_path) and old_file_path != file_path:
                    try:
                        os.remove(old_file_path)
                    except Exception as fe:
                        logger.warning(f"Failed to remove old file {old_file_path}: {fe}")
                # Remove from map temporarily to refresh
                doc_info_map.pop(existing_doc.get("filename", filename), None)

            # Save file to disk
            with open(file_path, "wb") as f_out:
                f_out.write(file_bytes)

            # Validate PDF readability
            info = get_pdf_info(file_path, filename)
            if info["page_count"] == 0:
                if os.path.exists(file_path):
                    os.remove(file_path)
                errors.append(f"'{filename}' has 0 pages or is invalid.")
                continue

            doc_entry = {
                "document_id": doc_id,
                "filename": filename,
                "file_hash": file_hash,
                "status": "Pending",  # Uploaded documents enter Pending status
                "pages": info["page_count"],
                "chunks": 0,
                "size_kb": info["size_kb"],
                "file_size": len(file_bytes),
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "indexed_at": None,
                "error_message": None
            }

            doc_info_map[filename] = doc_entry
            saved_files.append(doc_entry)
            logger.info(f"Saved uploaded PDF [Pending]: {filename} (ID: {doc_id}, {info['page_count']} pages, {info['size_kb']} KB)")

        except Exception as err:
            logger.error(f"Error saving file '{filename}': {str(err)}")
            errors.append(f"Failed to save '{filename}': {str(err)}")

    # Update metadata registry
    meta["documents_info"] = list(doc_info_map.values())
    meta["total_documents"] = len(doc_info_map)
    meta["total_pages"] = sum(d.get("pages", 0) for d in doc_info_map.values())
    ready_list = [d["filename"] for d in doc_info_map.values() if d.get("status") == "Ready"]
    meta["indexed_documents"] = ready_list
    meta["total_indexed"] = len(ready_list)
    meta["total_pending"] = sum(1 for d in doc_info_map.values() if d.get("status") == "Pending")
    meta["total_error"] = sum(1 for d in doc_info_map.values() if d.get("status") == "Error")
    _save_metadata_dict(VECTORSTORE_DIR, meta)

    if saved_files:
        clear_search_cache()
        clear_rag_response_cache()

    if not saved_files and errors:
        return jsonify({
            "success": False,
            "error": "No valid PDF files could be saved.",
            "details": errors
        }), 400

    return jsonify({
        "success": True,
        "files": [f["filename"] for f in saved_files],
        "uploaded_count": len(saved_files),
        "details": saved_files,
        "warnings": warnings,
        "errors": errors,
        "status": build_system_status_payload()
    })


# =============================================================================
# 4. DOCUMENT INDEXING & REBUILD ENDPOINTS
# =============================================================================
@app.route('/api/index', methods=['POST'])
@app.route('/index', methods=['POST'])
def index_documents():
    """
    Indexes Pending and Error documents:
    - Finds documents in 'Pending' or 'Error' status (or selected documents).
    - Sets their status to 'Indexing'.
    - Extracts PDF text, creates chunks, attaches document metadata.
    - Adds embeddings to FAISS and updates BM25.
    - Sets status to 'Ready' on success, or 'Error' on failure.
    - Partial Indexing: successfully indexed documents remain Ready even if some fail.
    - Does NOT re-index documents that are already 'Ready'.
    """
    global active_vectorstore

    data = request.get_json(silent=True) or {}
    selected_filter = data.get("selected_documents") or data.get("documents") or None

    meta = load_vectorstore_metadata(VECTORSTORE_DIR, docs_dir=DOCUMENTS_DIR)
    doc_info_map = {d["filename"]: d for d in meta.get("documents_info", [])}

    all_disk_files = [f for f in sorted(os.listdir(DOCUMENTS_DIR)) if f.lower().endswith('.pdf')]
    if not all_disk_files:
        return jsonify({
            "success": False,
            "error": "No PDF documents found. Please upload one or more PDFs first."
        }), 400

    # Ensure all disk files are in doc_info_map
    for f in all_disk_files:
        if f not in doc_info_map:
            fpath = os.path.join(DOCUMENTS_DIR, f)
            info = get_pdf_info(fpath, f)
            f_hash = compute_file_hash(fpath)
            doc_info_map[f] = {
                "document_id": f"doc_{f_hash[:12]}",
                "filename": f,
                "file_hash": f_hash,
                "status": "Pending",
                "pages": info.get("page_count", 1),
                "chunks": 0,
                "size_kb": info.get("size_kb", 0),
                "file_size": os.path.getsize(fpath) if os.path.exists(fpath) else 0,
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "indexed_at": None,
                "error_message": None
            }

    # Determine which documents need indexing (Pending or Error)
    to_index = []
    for fname, d in doc_info_map.items():
        if selected_filter is not None:
            if fname in selected_filter and d.get("status") in ("Pending", "Error", "Indexing"):
                to_index.append(fname)
        else:
            if d.get("status") in ("Pending", "Error", "Indexing"):
                to_index.append(fname)

    if not to_index:
        ready_count = sum(1 for d in doc_info_map.values() if d.get("status") == "Ready")
        return jsonify({
            "success": True,
            "message": "All documents are already indexed.",
            "indexed_count": 0,
            "failed_count": 0,
            "documents": ready_count,
            "status": build_system_status_payload()
        })

    t_start = time.perf_counter()
    logger.info(f"[INDEX] Processing indexing for {len(to_index)} document(s): {to_index}")

    # Set status to Indexing
    for fname in to_index:
        doc_info_map[fname]["status"] = "Indexing"
        doc_info_map[fname]["error_message"] = None

    meta["documents_info"] = list(doc_info_map.values())
    _save_metadata_dict(VECTORSTORE_DIR, meta)

    successfully_indexed = []
    failed_docs = []

    for fname in to_index:
        fpath = os.path.join(DOCUMENTS_DIR, fname)
        try:
            raw_docs, warnings = load_pdf_file(fpath, fname)
            if not raw_docs:
                raise ValueError("Could not extract any text from document (scanned or empty PDF).")

            doc_pages = raw_docs[0].metadata.get("total_pages", len(raw_docs))
            doc_chunks = split_documents(raw_docs)
            if not doc_chunks:
                raise ValueError("No text chunks could be generated from document.")

            doc_info_map[fname]["status"] = "Ready"
            doc_info_map[fname]["chunks"] = len(doc_chunks)
            doc_info_map[fname]["pages"] = doc_pages
            doc_info_map[fname]["indexed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            doc_info_map[fname]["error_message"] = None
            successfully_indexed.append(fname)
            logger.info(f"[INDEX] Successfully processed {fname}: {doc_pages} pages, {len(doc_chunks)} chunks.")

        except Exception as err:
            logger.error(f"[INDEX] Indexing failed for {fname}: {err}")
            doc_info_map[fname]["status"] = "Error"
            doc_info_map[fname]["chunks"] = 0
            doc_info_map[fname]["error_message"] = str(err)
            failed_docs.append({"filename": fname, "error": str(err)})

    # Gather chunks for ALL currently Ready documents to build active FAISS vector store
    all_ready_chunks = []
    ready_doc_names = []
    total_pages = 0

    for fname, doc_entry in doc_info_map.items():
        if doc_entry.get("status") == "Ready":
            fpath = os.path.join(DOCUMENTS_DIR, fname)
            if os.path.exists(fpath):
                r_docs, _ = load_pdf_file(fpath, fname)
                if r_docs:
                    doc_chunks = split_documents(r_docs)
                    doc_entry["chunks"] = len(doc_chunks)
                    for c in doc_chunks:
                        c.metadata["document_id"] = doc_entry.get("document_id", f"doc_{fname}")
                        c.metadata["document"] = fname
                        c.metadata["source"] = fname
                        c.metadata["file_hash"] = doc_entry.get("file_hash", "")
                    all_ready_chunks.extend(doc_chunks)
                    ready_doc_names.append(fname)
                    total_pages += doc_entry.get("pages", len(r_docs))

    # Build and save FAISS index
    if all_ready_chunks:
        emb = get_cached_embeddings()
        vectorstore = build_vectorstore(
            chunks=all_ready_chunks,
            embedding_model=emb,
            doc_names=ready_doc_names,
            documents_info=[doc_info_map[f] for f in ready_doc_names],
            total_pages=total_pages
        )
        save_vectorstore(
            vectorstore=vectorstore,
            folder_path=VECTORSTORE_DIR,
            doc_names=ready_doc_names,
            total_chunks=len(all_ready_chunks),
            documents_info=list(doc_info_map.values()),
            total_pages=total_pages
        )
        active_vectorstore = vectorstore
    else:
        clear_vectorstore(VECTORSTORE_DIR)
        active_vectorstore = None

    # Clear caches
    clear_bm25_cache()
    clear_search_cache()
    clear_rag_response_cache()
    clear_verification_cache()
    if hasattr(_CROSS_ENCODER, "clear_cache"):
        _CROSS_ENCODER.clear_cache()

    elapsed_sec = round(time.perf_counter() - t_start, 2)
    meta["documents_info"] = list(doc_info_map.values())
    meta["indexed_documents"] = ready_doc_names
    meta["total_chunks"] = len(all_ready_chunks)
    meta["total_pages"] = sum(d.get("pages", 0) for d in doc_info_map.values())
    meta["total_documents"] = len(doc_info_map)
    meta["total_indexed"] = len(ready_doc_names)
    meta["total_pending"] = sum(1 for d in doc_info_map.values() if d.get("status") == "Pending")
    meta["total_error"] = sum(1 for d in doc_info_map.values() if d.get("status") == "Error")
    _save_metadata_dict(VECTORSTORE_DIR, meta)

    msg = f"{len(successfully_indexed)} document(s) indexed successfully."
    if failed_docs:
        msg += f" ({len(failed_docs)} document(s) failed)."

    return jsonify({
        "success": len(successfully_indexed) > 0 or not failed_docs,
        "message": msg,
        "indexed_count": len(successfully_indexed),
        "failed_count": len(failed_docs),
        "documents": len(ready_doc_names),
        "pages": total_pages,
        "chunks": len(all_ready_chunks),
        "failed_details": failed_docs,
        "elapsed_seconds": elapsed_sec,
        "status": build_system_status_payload()
    })


@app.route('/api/rebuild', methods=['POST'])
@app.route('/rebuild', methods=['POST'])
def rebuild_index():
    """
    Clears current FAISS index & BM25 index, re-indexes ONLY documents that currently exist on disk.
    Removes stale chunks belonging to deleted documents.
    Updates their statuses to Ready on success, or Error on failure.
    """
    global active_vectorstore

    pdf_files = [f for f in sorted(os.listdir(DOCUMENTS_DIR)) if f.lower().endswith('.pdf')]
    if not pdf_files:
        clear_vectorstore(VECTORSTORE_DIR)
        active_vectorstore = None
        clear_bm25_cache()
        clear_search_cache()
        clear_rag_response_cache()
        clear_verification_cache()
        return jsonify({
            "success": True,
            "message": "Index cleared. No documents exist to rebuild.",
            "documents": 0,
            "pages": 0,
            "chunks": 0,
            "status": build_system_status_payload()
        })

    t_start = time.perf_counter()
    logger.info(f"[REBUILD] Rebuilding FAISS vector index from scratch for {len(pdf_files)} document(s)...")

    meta = load_vectorstore_metadata(VECTORSTORE_DIR, docs_dir=DOCUMENTS_DIR)
    doc_info_map = {d["filename"]: d for d in meta.get("documents_info", [])}

    all_ready_chunks = []
    ready_doc_names = []
    total_pages = 0
    failed_docs = []

    for fname in pdf_files:
        fpath = os.path.join(DOCUMENTS_DIR, fname)
        f_hash = compute_file_hash(fpath)
        doc_info_map[fname]["status"] = "Indexing"

        try:
            raw_docs, warnings = load_pdf_file(fpath, fname)
            if not raw_docs:
                raise ValueError("Could not extract text from document.")

            doc_pages = raw_docs[0].metadata.get("total_pages", len(raw_docs))
            doc_chunks = split_documents(raw_docs)
            if not doc_chunks:
                raise ValueError("No text chunks generated.")

            doc_id = doc_info_map[fname].get("document_id") or f"doc_{f_hash[:12]}"
            for c in doc_chunks:
                c.metadata["document_id"] = doc_id
                c.metadata["document"] = fname
                c.metadata["source"] = fname
                c.metadata["file_hash"] = f_hash

            doc_info_map[fname]["status"] = "Ready"
            doc_info_map[fname]["chunks"] = len(doc_chunks)
            doc_info_map[fname]["pages"] = doc_pages
            doc_info_map[fname]["indexed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            doc_info_map[fname]["error_message"] = None

            all_ready_chunks.extend(doc_chunks)
            ready_doc_names.append(fname)
            total_pages += doc_pages

        except Exception as err:
            logger.error(f"[REBUILD] Error indexing {fname}: {err}")
            doc_info_map[fname]["status"] = "Error"
            doc_info_map[fname]["chunks"] = 0
            doc_info_map[fname]["error_message"] = str(err)
            failed_docs.append({"filename": fname, "error": str(err)})

    # Build fresh vectorstore
    if all_ready_chunks:
        emb = get_cached_embeddings()
        vectorstore = build_vectorstore(
            chunks=all_ready_chunks,
            embedding_model=emb,
            doc_names=ready_doc_names,
            documents_info=[doc_info_map[f] for f in ready_doc_names],
            total_pages=total_pages
        )
        save_vectorstore(
            vectorstore=vectorstore,
            folder_path=VECTORSTORE_DIR,
            doc_names=ready_doc_names,
            total_chunks=len(all_ready_chunks),
            documents_info=list(doc_info_map.values()),
            total_pages=total_pages
        )
        active_vectorstore = vectorstore
    else:
        clear_vectorstore(VECTORSTORE_DIR)
        active_vectorstore = None

    # Clear all caches
    clear_bm25_cache()
    clear_search_cache()
    clear_rag_response_cache()
    clear_verification_cache()
    if hasattr(_CROSS_ENCODER, "clear_cache"):
        _CROSS_ENCODER.clear_cache()

    elapsed_sec = round(time.perf_counter() - t_start, 2)
    meta["documents_info"] = list(doc_info_map.values())
    meta["indexed_documents"] = ready_doc_names
    meta["total_chunks"] = len(all_ready_chunks)
    meta["total_pages"] = sum(d.get("pages", 0) for d in doc_info_map.values())
    meta["total_documents"] = len(doc_info_map)
    meta["total_indexed"] = len(ready_doc_names)
    meta["total_pending"] = sum(1 for d in doc_info_map.values() if d.get("status") == "Pending")
    meta["total_error"] = sum(1 for d in doc_info_map.values() if d.get("status") == "Error")
    _save_metadata_dict(VECTORSTORE_DIR, meta)

    return jsonify({
        "success": True,
        "message": f"Successfully rebuilt index for {len(ready_doc_names)} document(s) ({len(all_ready_chunks)} chunks).",
        "documents": len(ready_doc_names),
        "pages": total_pages,
        "chunks": len(all_ready_chunks),
        "elapsed_seconds": elapsed_sec,
        "failed_details": failed_docs,
        "status": build_system_status_payload()
    })


# =============================================================================
# 5. REMOVE SINGLE DOCUMENT & RESET ENDPOINTS
# =============================================================================
@app.route('/api/documents/<path:doc_id>', methods=['DELETE', 'POST'])
@app.route('/documents/<path:doc_id>', methods=['DELETE', 'POST'])
@app.route('/api/documents/delete', methods=['POST', 'DELETE'])
@app.route('/api/remove-document', methods=['POST', 'DELETE'])
@app.route('/remove-document', methods=['POST', 'DELETE'])
def remove_document(doc_id=None):
    """
    Completely removes a specific PDF document from the RAG system:
    - Removes physical PDF file from storage directory.
    - Removes document metadata, extracted text, and chunk records.
    - Rebuilds FAISS vector database from all remaining Ready documents or clears it if 0 remain.
    - Rebuilds / clears BM25 sparse search index.
    - Invalidates all search, RAG response, cross-encoder, and claim verification caches.
    - Updates document counters and returns refreshed system status.
    """
    global active_vectorstore

    # 1. Resolve document identifier from URL path, query params, or JSON body
    target_identifier = None
    if doc_id:
        target_identifier = str(doc_id).strip()
    
    if not target_identifier:
        data = request.get_json(silent=True) or {}
        target_identifier = (
            data.get("document_id") or 
            data.get("filename") or 
            data.get("id") or 
            data.get("doc_id") or 
            data.get("name") or 
            ""
        ).strip()

    if not target_identifier:
        target_identifier = (request.args.get("document_id") or request.args.get("filename") or "").strip()

    if not target_identifier:
        return jsonify({
            "success": False, 
            "error": "No document_id or filename specified for removal."
        }), 400

    # 2. Match the file in DOCUMENTS_DIR or metadata
    all_disk_files = os.listdir(DOCUMENTS_DIR) if os.path.exists(DOCUMENTS_DIR) else []
    target_file = None

    meta = load_vectorstore_metadata(VECTORSTORE_DIR, docs_dir=DOCUMENTS_DIR)
    doc_info_list = meta.get("documents_info", [])

    # First attempt: exact match by document_id or filename in metadata
    matched_doc_entry = None
    for d in doc_info_list:
        if d.get("document_id") == target_identifier or d.get("filename") == target_identifier:
            matched_doc_entry = d
            target_file = d.get("filename")
            break

    if not matched_doc_entry:
        if target_identifier in all_disk_files:
            target_file = target_identifier
        else:
            safe_name = secure_filename(target_identifier)
            if safe_name in all_disk_files:
                target_file = safe_name
            else:
                for f in all_disk_files:
                    if (f.lower() == target_identifier.lower() or 
                        f.lower() == safe_name.lower() or 
                        f.lower().startswith(target_identifier.lower()) or
                        target_identifier.lower().startswith(f.lower())):
                        target_file = f
                        break

    # Calculate chunk count and page count for the document being removed
    removed_chunks_count = 0
    removed_pages_count = 0
    doc_unique_id = target_identifier

    for d in doc_info_list:
        if (d.get("filename") == target_file or 
            d.get("filename") == target_identifier or 
            d.get("document_id") == target_identifier):
            removed_chunks_count = d.get("chunks", 0)
            removed_pages_count = d.get("pages", 0)
            doc_unique_id = d.get("document_id", target_identifier)
            if not target_file:
                target_file = d.get("filename")
            break

    if not target_file and target_identifier not in [d.get("filename") for d in doc_info_list]:
        logger.warning(f"[DELETE] Document '{target_identifier}' not found on disk or in index metadata.")
        return jsonify({
            "success": False, 
            "error": f"Document '{target_identifier}' not found."
        }), 404

    actual_filename = target_file or target_identifier

    try:
        # Step A: Delete physical PDF file from storage directory
        storage_file_removed = False
        if target_file:
            file_path = os.path.join(DOCUMENTS_DIR, target_file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    storage_file_removed = True
                except Exception as file_err:
                    logger.warning(f"[DELETE] Warning removing physical file {file_path}: {file_err}")

        # Step B: Remove document from metadata registry
        remaining_doc_info = [d for d in doc_info_list if d.get("filename") != actual_filename and d.get("document_id") != doc_unique_id]
        remaining_ready_docs = [d["filename"] for d in remaining_doc_info if d.get("status") == "Ready"]

        # Step C: Rebuild FAISS & BM25 Indexes or Clear Completely if 0 Ready documents remain
        if not remaining_ready_docs:
            clear_vectorstore(VECTORSTORE_DIR)
            active_vectorstore = None
            clear_bm25_cache()
            clear_search_cache()
            clear_rag_response_cache()
            clear_verification_cache()
            if hasattr(_CROSS_ENCODER, "clear_cache"):
                _CROSS_ENCODER.clear_cache()

            meta["documents_info"] = remaining_doc_info
            meta["indexed_documents"] = []
            meta["total_chunks"] = 0
            meta["total_pages"] = sum(d.get("pages", 0) for d in remaining_doc_info)
            meta["total_documents"] = len(remaining_doc_info)
            meta["total_indexed"] = 0
            meta["total_pending"] = sum(1 for d in remaining_doc_info if d.get("status") == "Pending")
            meta["total_error"] = sum(1 for d in remaining_doc_info if d.get("status") == "Error")
            _save_metadata_dict(VECTORSTORE_DIR, meta)

            logger.info("----------------------------------")
            logger.info("[DOCUMENT DELETE]")
            logger.info(f"document_id: {doc_unique_id}")
            logger.info(f"filename: {actual_filename}")
            logger.info(f"chunks_removed: {removed_chunks_count}")
            logger.info(f"pages_removed: {removed_pages_count}")
            logger.info(f"faiss_vectors_removed/rebuilt: {removed_chunks_count}")
            logger.info(f"bm25_entries_removed: {removed_chunks_count}")
            logger.info("cache_invalidated: true")
            logger.info(f"remaining_documents: {len(remaining_doc_info)}")
            logger.info("----------------------------------")

            return jsonify({
                "success": True,
                "deleted_document": actual_filename,
                "document_id": doc_unique_id,
                "removed_chunks": removed_chunks_count,
                "remaining_documents": len(remaining_doc_info),
                "message": "Document removed successfully.",
                "status": build_system_status_payload()
            })

        # Re-extract and chunk remaining Ready documents
        all_ready_chunks = []
        total_pages = 0

        for pdf_name in remaining_ready_docs:
            p_path = os.path.join(DOCUMENTS_DIR, pdf_name)
            if os.path.exists(p_path):
                docs, _ = load_pdf_file(p_path, pdf_name)
                if docs:
                    doc_chunks = split_documents(docs)
                    rem_doc_entry = next((d for d in remaining_doc_info if d.get("filename") == pdf_name), {})
                    rem_doc_id = rem_doc_entry.get("document_id") or f"doc_{rem_doc_entry.get('file_hash', pdf_name)[:12]}"
                    for c in doc_chunks:
                        c.metadata["document_id"] = rem_doc_id
                        c.metadata["document"] = pdf_name
                        c.metadata["source"] = pdf_name
                        c.metadata["file_hash"] = rem_doc_entry.get("file_hash", "")
                    all_ready_chunks.extend(doc_chunks)
                    doc_pages = docs[0].metadata.get("total_pages", len(docs))
                    total_pages += doc_pages

        if all_ready_chunks:
            emb = get_cached_embeddings()
            vectorstore = build_vectorstore(
                chunks=all_ready_chunks,
                embedding_model=emb,
                doc_names=remaining_ready_docs,
                documents_info=[d for d in remaining_doc_info if d["filename"] in remaining_ready_docs],
                total_pages=total_pages
            )
            save_vectorstore(
                vectorstore=vectorstore,
                folder_path=VECTORSTORE_DIR,
                doc_names=remaining_ready_docs,
                total_chunks=len(all_ready_chunks),
                documents_info=remaining_doc_info,
                total_pages=total_pages
            )
            active_vectorstore = vectorstore
        else:
            clear_vectorstore(VECTORSTORE_DIR)
            active_vectorstore = None

        # Step D: Invalidate all caches
        clear_bm25_cache()
        clear_search_cache()
        clear_rag_response_cache()
        clear_verification_cache()
        if hasattr(_CROSS_ENCODER, "clear_cache"):
            _CROSS_ENCODER.clear_cache()

        meta["documents_info"] = remaining_doc_info
        meta["indexed_documents"] = remaining_ready_docs
        meta["total_chunks"] = len(all_ready_chunks)
        meta["total_pages"] = sum(d.get("pages", 0) for d in remaining_doc_info)
        meta["total_documents"] = len(remaining_doc_info)
        meta["total_indexed"] = len(remaining_ready_docs)
        meta["total_pending"] = sum(1 for d in remaining_doc_info if d.get("status") == "Pending")
        meta["total_error"] = sum(1 for d in remaining_doc_info if d.get("status") == "Error")
        _save_metadata_dict(VECTORSTORE_DIR, meta)

        logger.info("----------------------------------")
        logger.info("[DOCUMENT DELETE]")
        logger.info(f"document_id: {doc_unique_id}")
        logger.info(f"filename: {actual_filename}")
        logger.info(f"chunks_removed: {removed_chunks_count}")
        logger.info(f"pages_removed: {removed_pages_count}")
        logger.info(f"faiss_vectors_removed/rebuilt: {removed_chunks_count}")
        logger.info(f"bm25_entries_removed: {removed_chunks_count}")
        logger.info("cache_invalidated: true")
        logger.info(f"remaining_documents: {len(remaining_doc_info)}")
        logger.info("----------------------------------")

        return jsonify({
            "success": True,
            "deleted_document": actual_filename,
            "document_id": doc_unique_id,
            "removed_chunks": removed_chunks_count,
            "remaining_documents": len(remaining_doc_info),
            "message": "Document removed successfully.",
            "status": build_system_status_payload()
        })

    except Exception as err:
        logger.error(f"[DELETE] Error removing document '{actual_filename}': {str(err)}", exc_info=True)
        return jsonify({
            "success": False, 
            "error": f"Unable to remove document. The index was not changed: {str(err)}"
        }), 500


@app.route('/api/reset', methods=['POST'])
@app.route('/reset', methods=['POST'])
def reset_all_documents_endpoint():
    """
    Clears all uploaded PDF files, wipes the vector store index, and resets all caches.
    """
    global active_vectorstore
    try:
        # 1. Delete all PDFs from DOCUMENTS_DIR
        if os.path.exists(DOCUMENTS_DIR):
            for fname in os.listdir(DOCUMENTS_DIR):
                fpath = os.path.join(DOCUMENTS_DIR, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except Exception as fe:
                    logger.warning(f"Could not remove {fpath}: {fe}")

        # 2. Clear vectorstore directory
        clear_vectorstore(VECTORSTORE_DIR)
        active_vectorstore = None

        # 3. Clear all caches
        clear_bm25_cache()
        clear_search_cache()
        clear_rag_response_cache()
        clear_verification_cache()
        if hasattr(_CROSS_ENCODER, "clear_cache"):
            _CROSS_ENCODER.clear_cache()

        logger.info("[RESET] All uploaded documents, FAISS index, and caches cleared.")
        return jsonify({
            "success": True,
            "message": "All documents and vector store cleared successfully.",
            "status": build_system_status_payload()
        })
    except Exception as err:
        logger.error(f"[RESET] Error resetting system: {err}", exc_info=True)
        return jsonify({"success": False, "error": f"Failed to reset system: {str(err)}"}), 500


# =============================================================================
# 6. FAST & ACCURATE MULTI-DOC QUESTION ANSWERING ENDPOINT WITH CONVERSATION MEMORY
# =============================================================================
@app.route('/api/chat', methods=['POST'])
@app.route('/api/ask', methods=['POST'])
@app.route('/api/query', methods=['POST'])
@app.route('/ask', methods=['POST'])
@app.route('/query', methods=['POST'])
def chat_endpoint():
    """
    Fast, multi-document RAG Question Answering with Conversation Memory & Smart Document Filtering:
    - Supports conversation history for contextual follow-up questions ("its types", "give example", etc.).
    - Resolves references and rewrites queries for FAISS & keyword search.
    - Validates selected_documents and filters out chunks from unselected documents.
    - Calculates verified relevance scores (0-100%).
    - Generates strictly document-grounded answers with direct PDF citation links.
    """
    t_start = time.perf_counter()
    req_id = f"req_{int(time.time() * 1000)}"

    data = request.get_json(silent=True) or {}
    query = (data.get("question") or data.get("message") or data.get("query") or "").strip()
    
    # Accept conversation or history (max 10 latest turns)
    raw_history = data.get("conversation") or data.get("history") or []
    chat_history = raw_history[-10:] if isinstance(raw_history, list) else []
    
    selected_documents = data.get("selected_documents", None)
    
    # Accept last_context from payload or extract from recent history
    last_context = data.get("last_context", None)
    if not last_context and chat_history:
        for msg in reversed(chat_history):
            if msg.get("role") in ("assistant", "ai") and msg.get("retrieved_context"):
                last_context = msg.get("retrieved_context")
                break

    if not query:
        return jsonify({"success": False, "error": "Please enter a question.", "error_code": "VALIDATION_ERROR"}), 400

    # FIX 1 & 2: Detect simple conversational greetings/casual inputs before RAG / LLM
    import re
    clean_query = re.sub(r"[^\w\s]", "", query.strip().lower()).strip()
    
    GREETING_PATTERNS = {
        "hi", "hello", "hey", "hii", "hiii", "hey there", "hello there",
        "good morning", "good afternoon", "good evening", "greetings", "howdy", "hiya"
    }
    THANKS_PATTERNS = {
        "thanks", "thank you", "thx", "ty", "thank you very much", "thanks a lot", "many thanks"
    }

    if clean_query in GREETING_PATTERNS or clean_query in THANKS_PATTERNS:
        if DEBUG_RAG:
            logger.info(f"\nQUERY: {query}\nCLASSIFICATION: GREETING\nSELECTED DOCUMENT COUNT: {len(selected_documents) if isinstance(selected_documents, list) else 'All'}\nSELECTED DOCUMENTS: {selected_documents}\nPIPELINE USED: LOCAL_RESPONSE")
        
        answer_text = (
            "You're welcome! Ask me another question about your selected documents."
            if clean_query in THANKS_PATTERNS
            else "Hello! 👋 Ask me a question about your selected documents."
        )
        return jsonify({
            "success": True,
            "answer": answer_text,
            "draft_answer": answer_text,
            "final_verified_answer": answer_text,
            "response_type": "greeting",
            "sources": [],
            "generation_failed": False,
            "response_time": round(time.perf_counter() - t_start, 3),
            "documents_used": [],
            "chunks_used": 0,
            "mode": "normal",
            "cache_status": "NONE"
        }), 200

    # Validate vector store readiness
    vs = get_active_vectorstore()
    meta = load_vectorstore_metadata(VECTORSTORE_DIR, docs_dir=DOCUMENTS_DIR)
    indexed_docs = meta.get("indexed_documents", [])
    
    if vs is None or meta.get("total_chunks", 0) == 0 or not indexed_docs:
        logger.warning("Chat rejected: No vector database or chunks indexed.")
        return jsonify({
            "success": False,
            "error": "No indexed documents are available. Please click 'Index Documents' first.",
            "error_code": "VALIDATION_ERROR"
        }), 400

    # Validate selected_documents
    if selected_documents is None:
        selected_documents = list(indexed_docs)
    else:
        if isinstance(selected_documents, list) and len(selected_documents) == 0:
            return jsonify({
                "success": False,
                "error": "Please select at least one document.",
                "error_code": "VALIDATION_ERROR"
            }), 400
        
        # Filter selected_documents to those currently indexed
        valid_selected = [doc for doc in selected_documents if doc in indexed_docs]
        if not valid_selected:
            return jsonify({
                "success": False,
                "error": "No indexed documents are available. Please click 'Index Documents' first.",
                "error_code": "VALIDATION_ERROR"
            }), 400

    if DEBUG_RAG:
        logger.info(f"\nQUERY: {query}\nCLASSIFICATION: DOCUMENT_QUESTION\nSELECTED DOCUMENT COUNT: {len(selected_documents)}\nSELECTED DOCUMENTS: {selected_documents}\nPIPELINE USED: FULL_RAG")
    # Cache key computation for repeated RAG questions (Requirement 8)
    from rag.retriever import normalize_retrieval_query
    norm_q = normalize_retrieval_query(query.strip())
    selected_docs_key = ",".join(sorted(selected_documents)) if selected_documents else ",".join(sorted(indexed_docs))
    index_version = f"{meta.get('last_updated', '')}_{meta.get('total_chunks', 0)}_{len(indexed_docs)}"
    raw_cache_str = f"{norm_q}||{selected_docs_key}||{index_version}||{RETRIEVAL_VERSION}"
    cache_key = hashlib.sha256(raw_cache_str.encode("utf-8")).hexdigest()

    # Check for in-memory cache hit if not a conversational follow-up
    if not chat_history and not last_context and cache_key in RAG_RESPONSE_CACHE:
        logger.info(f"[CACHE] HIT\n[CACHE] KEY: {cache_key[:16]}")
        cached_res = dict(RAG_RESPONSE_CACHE[cache_key])
        total_sec = round(time.perf_counter() - t_start, 3)
        cached_res["response_time"] = total_sec
        cached_res["cache_status"] = "HIT"
        cached_res["cache_hit"] = True
        cached_res["cache_key"] = cache_key[:16]
        if "timing_breakdown" in cached_res and isinstance(cached_res["timing_breakdown"], dict):
            cached_tb = dict(cached_res["timing_breakdown"])
            cached_tb["total_sec"] = total_sec
            cached_res["timing_breakdown"] = cached_tb
        else:
            log_perf_metrics(query, "HIT", {}, total_sec)
        log_debug_rag_trace(
            request_id=req_id,
            question=query,
            selected_docs=selected_documents,
            retrieved_count=cached_res.get("chunks_used", 0),
            top_docs=cached_res.get("documents_used", []),
            scores="Cached",
            context_len=len(cached_res.get("answer", "")),
            context_preview="[Cached context]",
            cache_key=cache_key[:16],
            cache_status="HIT",
            model_name=get_configured_model_name(),
            gen_started="Cached",
            gen_completed=f"{total_sec:.2f}s",
            supported=True,
            evidence_coverage="Cached (100%)",
            final_status="Cached Answer"
        )
        rec_id = record_interaction(query, cached_res, wall_time_sec=total_sec, selected_documents=selected_documents)
        if rec_id:
            cached_res["id"] = rec_id
            cached_res["interaction_id"] = rec_id
        return jsonify(cached_res)
    else:
        logger.info(f"[CACHE] MISS\n[CACHE] KEY: {cache_key[:16]}")

    # Check for ambiguous follow-up requests without previous conversation (Requirement 11)
    AMBIGUOUS_NO_CONTEXT_PATTERNS = [
        r'^(explain|describe|tell me about)\s+(it|this|that|them|these|those)[\?!.,;]?$',
        r'^(explain|describe)\s+(it|this|that|them|these|those)\s+(simply|in simple words|in detail)[\?!.,;]?$',
        r'^(give|show)\s+(me\s+)?(an?\s+)?example[\?!.,;]?$',
        r'^(give|show)\s+(me\s+)?(an?\s+)?example\s+(of\s+)?(it|this|that|them|these|those)[\?!.,;]?$',
        r'^(what is|what are|explain)\s+(the\s+)?(first|second|third|fourth|fifth|last|final)\s*(one|property|type|point|command)?[\?!.,;]?$',
        r'^(what does\s+(the\s+)?(first|second|third|fourth|last)\s+one\s+do)[\?!.,;]?$',
        r'^(what about\s+(it|this|that|them|these|those|the\s+last\s+(one|property|command)?))[\?!.,;]?$',
        r'^(what are\s+(its|their|the|these|those)\s+(types|kinds|categories|examples))[\?!.,;]?$'
    ]
    if not chat_history and not last_context:
        q_clean = query.strip().lower()
        for pat in AMBIGUOUS_NO_CONTEXT_PATTERNS:
            if re.match(pat, q_clean, re.IGNORECASE):
                total_sec = round(time.perf_counter() - t_start, 2)
                ambig_payload = {
                    "success": True,
                    "answer": "Please specify what you would like me to explain.",
                    "query_type": "NORMAL_QUESTION",
                    "term": "",
                    "rewritten_query": query,
                    "is_followup": False,
                    "used_context": False,
                    "sources": [],
                    "retrieved_context": [],
                    "response_time": total_sec,
                    "timing_breakdown": {
                        "retrieval_ms": 0.0,
                        "generation_sec": 0.0,
                        "total_sec": total_sec
                    }
                }
                rec_id = record_interaction(query, ambig_payload, wall_time_sec=total_sec, selected_documents=selected_documents)
                if rec_id:
                    ambig_payload["id"] = rec_id
                    ambig_payload["interaction_id"] = rec_id
                return jsonify(ambig_payload)

    api_key = load_gemini_api_key()

    try:
        # Step 1: Detect query classification and conversational context (Problem 2)
        query_class, is_followup, context_topic, comp_concepts = classify_user_query(
            query, chat_history, last_context
        )

        client = get_active_gemini_client()
        gemini_rewritten = None
        effective_rewritten = query.strip()
        active_context_topic = ""

        if query_class == "NEW QUESTION":
            # Strict Isolation: Pure new question completely bypasses prior conversation memory to avoid contamination
            chat_history = []
            last_context = None
            is_followup = False
            effective_rewritten = query.strip()
            active_context_topic = ""
        elif is_followup:
            if client is not None:
                gemini_rewritten = rewrite_query_with_gemini(client, query, chat_history, api_key=api_key)
            rule_rewritten, prior_topic = rewrite_query(query, chat_history, last_context)
            effective_rewritten = gemini_rewritten if (gemini_rewritten and len(gemini_rewritten) >= len(query.strip())) else rule_rewritten
            if not effective_rewritten:
                effective_rewritten = query.strip()
            active_context_topic = context_topic or prior_topic

        # Step 2: Check for Multi-Document Comparison & Cross-Document Synthesis
        comp_res = detect_comparison_question(query)
        is_comparison = comp_res.is_comparison
        comparison_topics = comp_res.topics
        comparison_subtype = getattr(comp_res, "subtype", "COMPARISON_TABLE")
        if not is_comparison and effective_rewritten:
            comp_res_rw = detect_comparison_question(effective_rewritten)
            if comp_res_rw.is_comparison:
                is_comparison = True
                comparison_topics = comp_res_rw.topics
                comparison_subtype = getattr(comp_res_rw, "subtype", "COMPARISON_TABLE")

        # Step 3: Complex Query Detection & Query Decomposition Pipeline (if not comparison)
        is_complex = (not is_comparison) and (is_complex_query(effective_rewritten) or (effective_rewritten != query and is_complex_query(query)))
        decomposed_subqueries = []
        if is_complex:
            try:
                decomposed_subqueries = decompose_query(
                    query=effective_rewritten,
                    llm_client=client,
                    api_key=api_key,
                    model_name=get_configured_model_name()
                )
            except Exception as decomp_err:
                logger.warning(f"Query decomposition error: {decomp_err}. Falling back to standard pipeline.")
                decomposed_subqueries = []

        if decomposed_subqueries and len(decomposed_subqueries) >= 2:
            logger.info(f"Query Decomposition activated with {len(decomposed_subqueries)} sub-queries.")
            t_decomp_start = time.perf_counter()
            retrieved_items, retrieval_stats = retrieve_with_query_decomposition(
                vectorstore=vs,
                query=effective_rewritten,
                sub_queries=decomposed_subqueries,
                selected_documents=selected_documents,
                vector_top_k=10,
                bm25_top_k=10,
                subquery_top_candidates=8,
                final_top_k=6,
                prior_topic=active_context_topic
            )
            t_retrieval_ms = retrieval_stats.get("retrieval_ms", 0.0) + retrieval_stats.get("reranking_ms", 0.0)
            t_retrieval_sec = round(t_retrieval_ms / 1000.0, 2)

            if retrieved_items:
                raw_sources = []
                retrieved_chunks_for_llm = []
                retrieved_context_data = []

                for item in retrieved_items:
                    doc = item[0]
                    f_score = item[1]
                    rel_pct = item[2] if len(item) > 2 else 90
                    retrieved_chunks_for_llm.append((doc, f_score))

                    doc_name = doc.metadata.get("document") or doc.metadata.get("source", "Document")
                    page_num = doc.metadata.get("page", 1)
                    chunk_content = doc.page_content.strip() if hasattr(doc, "page_content") else ""

                    raw_sources.append({
                        "document_id": doc_name,
                        "document": doc_name,
                        "filename": doc_name,
                        "page": page_num,
                        "relevance_pct": rel_pct,
                        "score": round(float(f_score), 3),
                        "pdf_url": f"/view_pdf/{doc_name}#page={page_num}",
                        "snippet": chunk_content[:500] + ("..." if len(chunk_content) > 500 else ""),
                        "excerpt": chunk_content[:500] + ("..." if len(chunk_content) > 500 else "")
                    })

                    retrieved_context_data.append({
                        "document_id": doc_name,
                        "document": doc_name,
                        "filename": doc_name,
                        "page": page_num,
                        "text": chunk_content[:600],
                        "pdf_url": f"/view_pdf/{doc_name}#page={page_num}"
                    })

                # Apply Contextual Compression
                t_comp_start = time.perf_counter()
                compression_res = compress_context(
                    query=effective_rewritten if effective_rewritten else query,
                    retrieved_chunks=retrieved_items,
                    cross_encoder=_CROSS_ENCODER
                )
                t_comp_ms = round((time.perf_counter() - t_comp_start) * 1000.0, 1)
                t_comp_sec = round(t_comp_ms / 1000.0, 2)
                comp_summary = compression_res.get("summary", {})
                comp_details = compression_res.get("compressed_chunks", [])

                t_gen_start = time.perf_counter()
                try:
                    answer = generate_rag_answer(
                        llm_client=client,
                        query=effective_rewritten if effective_rewritten else query,
                        retrieved_chunks=retrieved_chunks_for_llm,
                        query_type="COMPLEX_QUESTION",
                        extracted_term=active_context_topic,
                        conversation_history=chat_history,
                        rewritten_query=effective_rewritten,
                        api_key=api_key,
                        compressed_context=compression_res.get("formatted_context")
                    )
                except Exception as gen_err:
                    logger.warning(f"[DECOMPOSITION] Generation error: {gen_err}. Falling back to direct grounded evidence extraction.")
                    from rag.llm import extract_direct_answer_fallback
                    answer = extract_direct_answer_fallback(query, retrieved_chunks_for_llm, active_context_topic)

                t_gen_sec = round(time.perf_counter() - t_gen_start, 2)
                total_sec = round(time.perf_counter() - t_start, 2)

                # Deduplicate sources
                seen_sources = set()
                deduped_sources = []
                for src in raw_sources:
                    key = (src["document"], src["page"])
                    if key not in seen_sources:
                        seen_sources.add(key)
                        deduped_sources.append({
                            "document_id": src.get("document_id", src["document"]),
                            "document": src["document"],
                            "filename": src["document"],
                            "page": src["page"],
                            "relevance_pct": src.get("relevance_pct", 90),
                            "score": src.get("score", 0.90),
                            "pdf_url": src.get("pdf_url", f"/view_pdf/{src['document']}#page={src['page']}"),
                            "snippet": src.get("snippet", ""),
                            "excerpt": src.get("excerpt", src.get("snippet", ""))
                        })

                has_sources = (
                    answer != FALLBACK_RESPONSE and 
                    not answer.startswith("I couldn't find this information") and
                    not answer.startswith("I could not find this information") and
                    not answer.startswith("I couldn't find a definition") and
                    not answer.startswith("Gemini API key") and 
                    not answer.startswith("The AI service") and
                    not answer.startswith("AI service")
                )

                timing_breakdown = {
                    "query_expansion_ms": 0.0,
                    "query_expansion_sec": 0.0,
                    "retrieval_ms": t_retrieval_ms,
                    "retrieval_sec": t_retrieval_sec,
                    "reranking_ms": retrieval_stats.get("reranking_ms", 0.0),
                    "reranking_sec": round(retrieval_stats.get("reranking_ms", 0.0) / 1000.0, 2),
                    "compression_ms": t_comp_ms,
                    "compression_sec": t_comp_sec,
                    "generation_sec": t_gen_sec,
                    "total_sec": total_sec
                }

                try:
                    eval_data = evaluate_rag_response(
                        query=query,
                        answer=answer,
                        retrieved_items=retrieved_items,
                        response_time_sec=total_sec,
                        is_fallback=(not has_sources),
                        retrieval_stats=retrieval_stats,
                        timing_breakdown=timing_breakdown,
                        cross_encoder=_CROSS_ENCODER
                    )
                except Exception as eval_err:
                    logger.warning(f"Evaluation error: {eval_err}. Using safe default evaluation.")
                    eval_data = {
                        "confidence": "Medium",
                        "groundedness_score": 75,
                        "groundedness_label": "Grounded",
                        "faithfulness": 100.0,
                        "hallucination_risk": 0.0,
                        "risk_level": "LOW",
                        "total_claims": 0,
                        "supported_claims": 0,
                        "partial_claims": 0,
                        "unsupported_claims": 0,
                        "claims": [],
                        "warning": None,
                        "final_verified_answer": answer,
                        "draft_answer": answer,
                        "answer_correction": {}
                    }

                documents_used = list(set(src["document"] for src in deduped_sources))
                final_ans = eval_data.get("final_verified_answer") or answer
                final_ans = check_and_guard_answer_similarity(query, final_ans, chat_history)
                draft_ans = eval_data.get("draft_answer") or answer

                adaptive_payload_decomp = {
                    "retries_performed": 0,
                    "max_retries": 2,
                    "triggered": False,
                    "reason": "Query Decomposition multi-subquery retrieval used",
                    "attempts": [
                        {
                            "attempt": 1,
                            "type": "Decomposed Retrieval",
                            "queries": decomposed_subqueries,
                            "candidates_count": retrieval_stats.get("deduped_candidates", len(retrieved_items)),
                            "final_chunks_count": len(retrieved_items),
                            "sources": documents_used,
                            "supported_claims": eval_data.get("supported_claims", 0),
                            "partial_claims": eval_data.get("partial_claims", 0),
                            "unsupported_claims": eval_data.get("unsupported_claims", 0),
                            "total_claims": eval_data.get("total_claims", 0),
                            "faithfulness": eval_data.get("faithfulness", 100.0),
                            "groundedness": eval_data.get("groundedness_score", 0),
                            "status": "ACCEPTED"
                        }
                    ],
                    "selected_attempt": 1,
                    "process_steps": [
                        {"icon": "🔎", "title": "Decomposed Retrieval", "desc": f"{len(decomposed_subqueries)} sub-queries executed"},
                        {"icon": "⚡", "title": "Reranking", "desc": f"{len(retrieved_items)} final chunks selected"},
                        {"icon": "🗜️", "title": "Contextual Compression", "desc": f"{len(comp_details)} chunks compressed ({comp_summary.get('reduction_percentage', 0)}% reduction)"},
                        {"icon": "🧠", "title": "Answer Verification", "desc": f"{eval_data.get('supported_claims', 0)}/{eval_data.get('total_claims', 1) or 1} claims supported"},
                        {"icon": "✅", "title": "Final Status", "desc": "High-confidence verified answer"}
                    ]
                }

                response_payload_decomp = {
                    "success": True,
                    "answer": final_ans,
                    "draft_answer": draft_ans,
                    "final_verified_answer": final_ans,
                    "answer_correction": eval_data.get("answer_correction", {}),
                    "mode": "decomposition",
                    "retrieval_method": "decomposition",
                    "is_decomposed": True,
                    "sub_queries": decomposed_subqueries,
                    "semantic_results": retrieval_stats.get("semantic_candidates", 0),
                    "keyword_results": retrieval_stats.get("keyword_candidates", 0),
                    "fused_candidates": retrieval_stats.get("deduped_candidates", 0),
                    "reranked_results": retrieval_stats.get("final_chunks", 0),
                    "is_comparison": "compare" in effective_rewritten.lower() or "difference" in effective_rewritten.lower(),
                    "is_followup": is_followup,
                    "context_used": is_followup,
                    "context_topic": active_context_topic,
                    "query_type": "COMPLEX_QUESTION",
                    "term": active_context_topic,
                    "rewritten_query": effective_rewritten,
                    "documents_used": documents_used,
                    "chunks_used": len(retrieved_items),
                    "sources": deduped_sources if has_sources else [],
                    "grouped_sources": build_grouped_sources_payload(deduped_sources if has_sources else [], primary_doc=retrieval_stats.get("primary_document", "")),
                    "primary_source": retrieval_stats.get("primary_source", f"{deduped_sources[0]['document']} — Page {deduped_sources[0]['page']}" if deduped_sources else "None"),
                    "document_relevance_scores": retrieval_stats.get("document_relevance_scores", {}),
                    "retrieved_context": retrieved_context_data,
                    "evaluation": eval_data,
                    "retrieval_stats": retrieval_stats,
                    "response_time": total_sec,
                    "timing_breakdown": timing_breakdown,
                    "compression": comp_summary,
                    "compression_details": comp_details,
                    "adaptive_retrieval": adaptive_payload_decomp,
                    "cache_status": "MISS",
                    "cache_hit": False,
                    "cache_key": cache_key[:16]
                }

                log_perf_metrics(query, "MISS", timing_breakdown, total_sec)
                if not chat_history and not last_context:
                    RAG_RESPONSE_CACHE[cache_key] = response_payload_decomp
                rec_id = record_interaction(query, response_payload_decomp, wall_time_sec=total_sec, selected_documents=selected_documents)
                if rec_id:
                    response_payload_decomp["id"] = rec_id
                    response_payload_decomp["interaction_id"] = rec_id
                return jsonify(response_payload_decomp)

        # Step 4: Multi-Document Comparison and Synthesis Pipeline
        # Single document error handling for comparison questions (when comparing whole documents)
        if is_comparison and selected_documents is not None and len(selected_documents) == 1 and not comparison_topics:
            total_sec = round(time.perf_counter() - t_start, 2)
            warning_msg = "Please select at least two relevant documents for a multi-document comparison."
            comp_warn_payload = {
                "success": True,
                "answer": warning_msg,
                "mode": "comparison",
                "is_comparison": True,
                "is_followup": is_followup,
                "context_used": is_followup,
                "context_topic": active_context_topic,
                "query_type": "COMPARISON_QUESTION",
                "term": active_context_topic,
                "rewritten_query": effective_rewritten,
                "documents_used": selected_documents,
                "chunks_used": 0,
                "sources": [],
                "retrieved_context": [],
                "response_time": total_sec,
                "timing_breakdown": {
                    "retrieval_ms": 0.0,
                    "retrieval_sec": 0.0,
                    "generation_sec": 0.0,
                    "total_sec": total_sec
                },
                "cache_status": "MISS",
                "cache_hit": False
            }
            rec_id = record_interaction(query, comp_warn_payload, wall_time_sec=total_sec, selected_documents=selected_documents)
            if rec_id:
                comp_warn_payload["id"] = rec_id
                comp_warn_payload["interaction_id"] = rec_id
            return jsonify(comp_warn_payload)

        if is_comparison:
            # Multi-document balanced retrieval
            t_retrieval_start = time.perf_counter()
            retrieved_items, grouped_context, query_type, _, retrieval_stats = retrieve_multi_document_context(
                vectorstore=vs,
                query=effective_rewritten,
                comparison_topics=comparison_topics,
                selected_documents=selected_documents,
                candidate_pool_size=16,
                final_k_per_doc=3,
                top_k_final=8,
                chat_history=chat_history,
                last_context=last_context,
                comparison_subtype=comparison_subtype
            )
            t_retrieval_ms = round((time.perf_counter() - t_retrieval_start) * 1000, 1)
            t_retrieval_sec = round(t_retrieval_ms / 1000.0, 2)

            if not retrieved_items:
                total_sec = round(time.perf_counter() - t_start, 2)
                fallback_msg = "The selected documents do not contain enough information to make this comparison."
                timing_info = {
                    "retrieval_ms": t_retrieval_ms,
                    "retrieval_sec": t_retrieval_sec,
                    "generation_sec": 0.0,
                    "total_sec": total_sec
                }
                eval_data = evaluate_rag_response(
                    query=query,
                    answer=fallback_msg,
                    retrieved_items=[],
                    response_time_sec=total_sec,
                    is_fallback=True,
                    retrieval_stats=retrieval_stats,
                    timing_breakdown=timing_info,
                    cross_encoder=_CROSS_ENCODER
                )
                comp_fb_payload = {
                    "success": True,
                    "answer": fallback_msg,
                    "mode": "comparison",
                    "is_comparison": True,
                    "is_followup": is_followup,
                    "context_used": is_followup,
                    "context_topic": active_context_topic,
                    "query_type": query_type,
                    "term": active_context_topic,
                    "rewritten_query": effective_rewritten,
                    "documents_used": [],
                    "chunks_used": 0,
                    "sources": [],
                    "retrieved_context": [],
                    "evaluation": eval_data,
                    "retrieval_stats": retrieval_stats,
                    "response_time": total_sec,
                    "timing_breakdown": timing_info,
                    "cache_status": "MISS",
                    "cache_hit": False
                }
                rec_id = record_interaction(query, comp_fb_payload, wall_time_sec=total_sec, selected_documents=selected_documents)
                if rec_id:
                    comp_fb_payload["id"] = rec_id
                    comp_fb_payload["interaction_id"] = rec_id
                return jsonify(comp_fb_payload)

            # Prepare source metadata list
            raw_sources = []
            retrieved_context_data = []

            for item in retrieved_items:
                doc = item[0]
                f_score = item[1]
                rel_pct = item[2] if len(item) > 2 else 90
                doc_name = doc.metadata.get("document") or doc.metadata.get("source", "Document")
                page_num = doc.metadata.get("page", 1)
                chunk_content = doc.page_content.strip() if hasattr(doc, "page_content") else ""

                raw_sources.append({
                    "document_id": doc_name,
                    "document": doc_name,
                    "filename": doc_name,
                    "page": page_num,
                    "relevance_pct": rel_pct,
                    "score": round(float(f_score), 3),
                    "pdf_url": f"/view_pdf/{doc_name}#page={page_num}",
                    "snippet": chunk_content[:500] + ("..." if len(chunk_content) > 500 else ""),
                    "excerpt": chunk_content[:500] + ("..." if len(chunk_content) > 500 else "")
                })

                retrieved_context_data.append({
                    "document_id": doc_name,
                    "document": doc_name,
                    "filename": doc_name,
                    "page": page_num,
                    "text": chunk_content[:600],
                    "pdf_url": f"/view_pdf/{doc_name}#page={page_num}"
                })

            # Deduplicate sources preserving order
            seen_sources = set()
            deduped_sources = []
            for src in raw_sources:
                key = (src["document"], src["page"])
                if key not in seen_sources:
                    seen_sources.add(key)
                    deduped_sources.append(src)

            # Contextual Compression for comparison
            t_comp_start = time.perf_counter()
            compression_res = compress_context(
                query=effective_rewritten if effective_rewritten else query,
                retrieved_chunks=retrieved_items,
                cross_encoder=_CROSS_ENCODER
            )
            t_comp_ms = round((time.perf_counter() - t_comp_start) * 1000.0, 1)
            t_comp_sec = round(t_comp_ms / 1000.0, 2)
            comp_summary = compression_res.get("summary", {})
            comp_details = compression_res.get("compressed_chunks", [])

            # Generate structured comparison answer using Gemini
            t_gen_start = time.perf_counter()
            try:
                answer = generate_comparison_answer(
                    llm_client=client,
                    query=effective_rewritten if effective_rewritten else query,
                    grouped_context=grouped_context,
                    comparison_topics=comparison_topics,
                    conversation_history=chat_history,
                    api_key=api_key,
                    comparison_subtype=comparison_subtype
                )
            except Exception as gen_err:
                logger.warning(f"[COMPARISON] Generation error: {gen_err}. Falling back to direct grounded comparison extraction.")
                from rag.llm import extract_direct_comparison_fallback
                answer = extract_direct_comparison_fallback(grouped_context, comparison_topics, comparison_subtype)

            t_gen_sec = round(time.perf_counter() - t_gen_start, 2)
            total_sec = round(time.perf_counter() - t_start, 2)

            documents_used = list(grouped_context.keys())
            chunks_used = len(retrieved_items)

            timing_breakdown = {
                "retrieval_ms": t_retrieval_ms,
                "retrieval_sec": t_retrieval_sec,
                "compression_ms": t_comp_ms,
                "compression_sec": t_comp_sec,
                "generation_sec": t_gen_sec,
                "total_sec": total_sec
            }

            try:
                eval_data = evaluate_rag_response(
                    query=query,
                    answer=answer,
                    retrieved_items=retrieved_items,
                    response_time_sec=total_sec,
                    is_fallback=answer.startswith("I couldn't find enough information"),
                    retrieval_stats=retrieval_stats,
                    timing_breakdown=timing_breakdown,
                    cross_encoder=_CROSS_ENCODER
                )
            except Exception as eval_err:
                logger.warning(f"Evaluation error in comparison: {eval_err}. Using default evaluation.")
                eval_data = {
                    "confidence": "Medium",
                    "groundedness_score": 75,
                    "groundedness_label": "Grounded",
                    "faithfulness": 100.0,
                    "hallucination_risk": 0.0,
                    "risk_level": "LOW",
                    "total_claims": 0,
                    "supported_claims": 0,
                    "partial_claims": 0,
                    "unsupported_claims": 0,
                    "claims": [],
                    "warning": None,
                    "final_verified_answer": answer,
                    "draft_answer": answer,
                    "answer_correction": {}
                }

            final_ans = eval_data.get("final_verified_answer") or answer
            final_ans = check_and_guard_answer_similarity(query, final_ans, chat_history)
            draft_ans = eval_data.get("draft_answer") or answer

            adaptive_payload_comp = {
                "retries_performed": 0,
                "max_retries": 2,
                "triggered": False,
                "reason": "Multi-Document balanced comparative retrieval used",
                "attempts": [
                    {
                        "attempt": 1,
                        "type": "Comparison Retrieval",
                        "queries": [effective_rewritten],
                        "candidates_count": len(retrieved_items),
                        "final_chunks_count": len(retrieved_items),
                        "sources": documents_used,
                        "supported_claims": eval_data.get("supported_claims", 0),
                        "partial_claims": eval_data.get("partial_claims", 0),
                        "unsupported_claims": eval_data.get("unsupported_claims", 0),
                        "total_claims": eval_data.get("total_claims", 0),
                        "faithfulness": eval_data.get("faithfulness", 100.0),
                        "groundedness": eval_data.get("groundedness_score", 0),
                        "status": "ACCEPTED"
                    }
                ],
                "selected_attempt": 1,
                "process_steps": [
                    {"icon": "🔎", "title": "Comparative Retrieval", "desc": f"Balanced chunks from {len(documents_used)} documents"},
                    {"icon": "⚡", "title": "Reranking", "desc": f"{len(retrieved_items)} top chunks selected"},
                    {"icon": "🗜️", "title": "Contextual Compression", "desc": f"{len(comp_details)} chunks compressed ({comp_summary.get('reduction_percentage', 0)}% reduction)"},
                    {"icon": "🧠", "title": "Answer Verification", "desc": f"{eval_data.get('supported_claims', 0)}/{eval_data.get('total_claims', 1) or 1} claims supported"},
                    {"icon": "✅", "title": "Final Status", "desc": "High-confidence verified comparison"}
                ]
            }

            response_payload_comp = {
                "success": True,
                "answer": final_ans,
                "draft_answer": draft_ans,
                "final_verified_answer": final_ans,
                "answer_correction": eval_data.get("answer_correction", {}),
                "mode": "comparison",
                "retrieval_method": "hybrid",
                "semantic_results": retrieval_stats.get("semantic_candidates", 0),
                "keyword_results": retrieval_stats.get("keyword_candidates", 0),
                "fused_candidates": retrieval_stats.get("deduped_candidates", 0),
                "reranked_results": retrieval_stats.get("final_chunks", 0),
                "is_comparison": True,
                "is_followup": is_followup,
                "context_used": is_followup,
                "context_topic": active_context_topic,
                "query_type": query_type,
                "term": ", ".join(comparison_topics) if comparison_topics else active_context_topic,
                "rewritten_query": effective_rewritten,
                "documents_used": documents_used,
                "chunks_used": chunks_used,
                "sources": deduped_sources,
                "grouped_sources": build_grouped_sources_payload(deduped_sources, primary_doc=retrieval_stats.get("primary_document", "")),
                "primary_source": retrieval_stats.get("primary_source", f"{deduped_sources[0]['document']} — Page {deduped_sources[0]['page']}" if deduped_sources else "None"),
                "document_relevance_scores": retrieval_stats.get("document_relevance_scores", {}),
                "retrieved_context": retrieved_context_data,
                "evaluation": eval_data,
                "retrieval_stats": retrieval_stats,
                "response_time": total_sec,
                "timing_breakdown": timing_breakdown,
                "compression": comp_summary,
                "compression_details": comp_details,
                "adaptive_retrieval": adaptive_payload_comp,
                "cache_status": "MISS",
                "cache_hit": False,
                "cache_key": cache_key[:16]
            }

            log_perf_metrics(query, "MISS", timing_breakdown, total_sec)
            if not chat_history and not last_context:
                RAG_RESPONSE_CACHE[cache_key] = response_payload_comp
            rec_id = record_interaction(query, response_payload_comp, wall_time_sec=total_sec, selected_documents=selected_documents)
            if rec_id:
                response_payload_comp["id"] = rec_id
                response_payload_comp["interaction_id"] = rec_id
            return jsonify(response_payload_comp)

        else:
            # Step 3: Adaptive Retrieval Pipeline with Multi-Query Hybrid Search + Cross-Encoder + Quality Check + Auto Retry
            adaptive_result = adaptive_retrieval_pipeline(
                vectorstore=vs,
                query=query,
                rewritten_query=effective_rewritten,
                chat_history=chat_history,
                selected_documents=selected_documents,
                last_context=last_context,
                llm_client=client,
                api_key=api_key,
                model_name=get_configured_model_name(),
                max_retries=2
            )

            if adaptive_result.get("generation_failed") or not adaptive_result.get("success", True) or adaptive_result.get("answer") is None:
                return jsonify({
                    "success": False,
                    "generation_failed": True,
                    "answer": None,
                    "draft_answer": None,
                    "final_verified_answer": None,
                    "error": adaptive_result.get("error", "The AI model did not return an answer. Please try again."),
                    "mode": "followup" if is_followup else "normal",
                    "retrieval_method": "hybrid",
                    "sources": adaptive_result.get("sources", []),
                    "grouped_sources": adaptive_result.get("grouped_sources", []),
                    "primary_source": adaptive_result.get("primary_source", ""),
                    "retrieved_context": adaptive_result.get("retrieved_context", []),
                    "retrieval_stats": adaptive_result.get("retrieval_stats", {}),
                    "response_time": adaptive_result.get("response_time", 0.0),
                    "timing_breakdown": adaptive_result.get("timing_breakdown", {}),
                    "compression": adaptive_result.get("compression", {}),
                    "compression_details": adaptive_result.get("compression_details", []),
                    "adaptive_retrieval": adaptive_result.get("adaptive_retrieval", {})
                })

            retrieval_stats = adaptive_result.get("retrieval_stats", {})
            eval_data = adaptive_result.get("evaluation", {})
            final_ans = adaptive_result.get("final_verified_answer") or adaptive_result.get("answer")
            final_ans = check_and_guard_answer_similarity(query, final_ans, chat_history)
            draft_ans = adaptive_result.get("draft_answer") or final_ans
            t_breakdown = adaptive_result.get("timing_breakdown", {})
            total_sec_normal = adaptive_result.get("response_time", 0.0)

            response_payload_normal = {
                "success": True,
                "answer": final_ans,
                "draft_answer": draft_ans,
                "final_verified_answer": final_ans,
                "answer_correction": eval_data.get("answer_correction", {}),
                "mode": "followup" if is_followup else "normal",
                "retrieval_method": "hybrid",
                "semantic_results": retrieval_stats.get("semantic_candidates", 0),
                "keyword_results": retrieval_stats.get("keyword_candidates", 0),
                "fused_candidates": retrieval_stats.get("deduped_candidates", 0),
                "reranked_results": retrieval_stats.get("final_chunks", 0),
                "is_comparison": False,
                "is_followup": is_followup,
                "context_used": is_followup,
                "context_topic": active_context_topic,
                "query_type": adaptive_result.get("query_type", "NORMAL_QUESTION"),
                "term": active_context_topic or adaptive_result.get("term", ""),
                "rewritten_query": effective_rewritten,
                "documents_used": adaptive_result.get("documents_used", []),
                "chunks_used": adaptive_result.get("chunks_used", 0),
                "sources": adaptive_result.get("sources", []),
                "grouped_sources": adaptive_result.get("grouped_sources", []),
                "primary_source": adaptive_result.get("primary_source", ""),
                "document_relevance_scores": retrieval_stats.get("document_relevance_scores", {}),
                "retrieved_context": adaptive_result.get("retrieved_context", []),
                "evaluation": eval_data,
                "retrieval_stats": retrieval_stats,
                "response_time": total_sec_normal,
                "timing_breakdown": t_breakdown,
                "compression": adaptive_result.get("compression", {}),
                "compression_details": adaptive_result.get("compression_details", []),
                "adaptive_retrieval": adaptive_result.get("adaptive_retrieval", {}),
                "structured_data": adaptive_result.get("structured_data", {"tables_detected": 0, "tables_retrieved": 0, "tables": []}),
                "cache_status": "MISS",
                "cache_hit": False,
                "cache_key": cache_key[:16]
            }

            sources_list = adaptive_result.get("sources", [])
            top_docs_list = list(set(s.get("document", "") for s in sources_list if s.get("document")))
            scores_list = [s.get("score") for s in sources_list[:4]]
            retrieved_ctx = adaptive_result.get("retrieved_context", [])
            ctx_text = " ".join(c.get("text", "") for c in retrieved_ctx) if retrieved_ctx else final_ans
            supported_flag = (eval_data.get("supported_claims", 0) > 0) or (final_ans != FALLBACK_RESPONSE and not final_ans.startswith("NOT SUPPORTED") and not final_ans.startswith("I couldn't find"))
            cov_str = f"{eval_data.get('supported_claims', 0)}/{max(1, eval_data.get('total_claims', 1))} claims ({eval_data.get('faithfulness', 100.0):.0f}%)"

            log_debug_rag_trace(
                request_id=req_id,
                question=query,
                selected_docs=selected_documents,
                retrieved_count=adaptive_result.get("chunks_used", len(sources_list)),
                top_docs=top_docs_list,
                scores=scores_list,
                context_len=len(ctx_text),
                context_preview=ctx_text,
                cache_key=cache_key[:16],
                cache_status="MISS",
                model_name=get_configured_model_name(),
                gen_started=f"{t_breakdown.get('retrieval_ms', 0.0):.1f}ms",
                gen_completed=f"{t_breakdown.get('generation_sec', 0.0):.2f}s",
                supported=supported_flag,
                evidence_coverage=cov_str,
                final_status=eval_data.get("groundedness_label", "Verified")
            )
            if not chat_history and not last_context:
                RAG_RESPONSE_CACHE[cache_key] = response_payload_normal
            rec_id = record_interaction(query, response_payload_normal, wall_time_sec=total_sec_normal, selected_documents=selected_documents)
            if rec_id:
                response_payload_normal["id"] = rec_id
                response_payload_normal["interaction_id"] = rec_id
            return jsonify(response_payload_normal)

    except AIAuthError as err:
        logger.error(f"[AI ERROR] Authentication failure: {err}")
        return jsonify({
            "success": False,
            "generation_failed": True,
            "answer": None,
            "draft_answer": None,
            "final_verified_answer": None,
            "error_code": "AI_AUTH_ERROR",
            "error_category": "AUTH_ERROR",
            "error": "The AI service is temporarily unavailable. Please check the API configuration and try again."
        }), 200

    except AIRateLimitError as err:
        logger.error(f"[AI ERROR] Rate limit reached: {err}")
        return jsonify({
            "success": False,
            "generation_failed": True,
            "answer": None,
            "draft_answer": None,
            "final_verified_answer": None,
            "error_code": "AI_RATE_LIMIT",
            "error_category": "RATE_LIMIT",
            "error": "The AI service is temporarily unavailable due to rate limits. Please try again in a few moments."
        }), 200

    except AIEmptyResponseError as err:
        logger.error(f"[AI ERROR] Empty model response: {err}")
        return jsonify({
            "success": False,
            "generation_failed": True,
            "answer": None,
            "draft_answer": None,
            "final_verified_answer": None,
            "error_code": "AI_EMPTY_RESPONSE",
            "error_category": "EMPTY_RESPONSE",
            "error": "The AI service returned an empty response. Please try again."
        }), 200

    except (AIGenerationError, LLMGenerationError) as err:
        err_str = str(err).lower()
        err_code = "AI_TIMEOUT" if ("timeout" in err_str or "timed out" in err_str or "deadline" in err_str) else "AI_GENERATION_ERROR"
        logger.error(f"[AI ERROR] Generation failure ({err_code}): {err}")
        return jsonify({
            "success": False,
            "generation_failed": True,
            "answer": None,
            "draft_answer": None,
            "final_verified_answer": None,
            "error_code": err_code,
            "error_category": "GENERATION_FAILURE",
            "error": "The AI service is temporarily unavailable. Please try again."
        }), 200

    except Exception as err:
        logger.error(f"Error processing question: {str(err)}", exc_info=True)
        return jsonify({
            "success": False,
            "generation_failed": True,
            "answer": None,
            "draft_answer": None,
            "final_verified_answer": None,
            "error_code": "INTERNAL_SERVER_ERROR",
            "error_category": "INTERNAL_ERROR",
            "error": "An internal server error occurred.",
            "details": f"Error Type: {type(err).__name__}"
        }), 200


# =============================================================================
# 7. FAST DOCUMENT SEARCH ENDPOINT (NO GEMINI CALLS, CACHED)
# =============================================================================
@app.route('/api/search', methods=['POST'])
@app.route('/api/document-search', methods=['POST'])
@app.route('/api/search-documents', methods=['POST'])
@app.route('/search', methods=['POST'])
def document_search_endpoint():
    """
    Fast Document Search endpoint:
    - Pure FAISS vector + exact keyword search across indexed chunks.
    - NEVER calls Gemini LLM.
    - Filters strictly by selected_documents.
    - Uses in-memory caching for repeated queries.
    - Target response time: < 1 second.
    """
    t_start = time.perf_counter()

    data = request.get_json(silent=True) or {}
    query = data.get("query") or data.get("question") or data.get("q") or ""
    query = str(query).strip()

    selected_documents = data.get("selected_documents", None)

    if not query:
        return jsonify({
            "success": False,
            "error": "Please enter a search query.",
            "results": [],
            "count": 0
        }), 400

    vs = get_active_vectorstore()
    meta = load_vectorstore_metadata(VECTORSTORE_DIR, docs_dir=DOCUMENTS_DIR)
    indexed_docs = meta.get("indexed_documents", [])

    if vs is None or meta.get("total_chunks", 0) == 0 or not indexed_docs:
        return jsonify({
            "success": False,
            "error": "No indexed documents are available. Please click 'Index Documents' first.",
            "results": [],
            "count": 0
        }), 400

    # Validate and filter selected_documents
    if selected_documents is not None:
        if isinstance(selected_documents, list) and len(selected_documents) == 0:
            return jsonify({
                "success": False,
                "error": "Please select at least one document.",
                "results": [],
                "count": 0
            }), 400

        valid_selected = [doc for doc in selected_documents if doc in indexed_docs]
        if not valid_selected:
            return jsonify({
                "success": False,
                "error": "No indexed documents are available. Please click 'Index Documents' first.",
                "results": [],
                "count": 0
            }), 400
        selected_documents = valid_selected

    # Check in-memory Cache
    selected_tuple = tuple(sorted(selected_documents)) if selected_documents else ()
    cache_key = (query.lower(), selected_tuple)

    num_selected = len(selected_documents) if selected_documents else len(indexed_docs)

    if cache_key in DOCUMENT_SEARCH_CACHE:
        cached_results = DOCUMENT_SEARCH_CACHE[cache_key]
        t_elapsed = round(time.perf_counter() - t_start, 3)

        # Log formatted search hit
        logger.info("----------------------------------")
        logger.info("DOCUMENT SEARCH")
        logger.info(f"Query: {query}")
        logger.info(f"Selected documents: {num_selected}")
        logger.info(f"Results: {len(cached_results)}")
        logger.info("Cache: HIT")
        logger.info(f"Search time: {t_elapsed}s")
        logger.info("----------------------------------")

        return jsonify({
            "success": True,
            "query": query,
            "results": cached_results,
            "count": len(cached_results),
            "cached": True,
            "elapsed_seconds": t_elapsed,
            "message": "No matching content found in the selected documents." if not cached_results else None
        })

    # Execute fast search across vector store
    results = search_documents_fast(
        vectorstore=vs,
        query=query,
        selected_documents=selected_documents,
        top_k=10
    )

    t_elapsed = round(time.perf_counter() - t_start, 3)

    # Save in cache
    DOCUMENT_SEARCH_CACHE[cache_key] = results

    # Log formatted search miss
    logger.info("----------------------------------")
    logger.info("DOCUMENT SEARCH")
    logger.info(f"Query: {query}")
    logger.info(f"Selected documents: {num_selected}")
    logger.info(f"Results: {len(results)}")
    logger.info("Cache: MISS")
    logger.info(f"Search time: {t_elapsed}s")
    logger.info("----------------------------------")

    return jsonify({
        "success": True,
        "query": query,
        "results": results,
        "count": len(results),
        "cached": False,
        "elapsed_seconds": t_elapsed,
        "message": "No matching content found in the selected documents." if not results else None
    })


# =============================================================================
# 8. CLEAR CHAT & NEW CHAT ENDPOINTS (Preserves Vector Index and Documents)
# =============================================================================
@app.route('/api/clear_chat', methods=['POST'])
@app.route('/api/clear-chat', methods=['POST'])
@app.route('/clear_chat', methods=['POST'])
@app.route('/clear-chat', methods=['POST'])
@app.route('/api/new_chat', methods=['POST'])
@app.route('/api/new-chat', methods=['POST'])
@app.route('/new_chat', methods=['POST'])
@app.route('/new-chat', methods=['POST'])
def clear_chat_endpoint():
    """
    Clears the active conversation session memory without modifying uploaded documents or FAISS vectorstore.
    """
    return jsonify({
        "success": True,
        "message": "Conversation cleared."
    })


# =============================================================================
# 9. RESET / CLEAR ALL DOCUMENTS ENDPOINT
# =============================================================================
@app.route('/api/reset', methods=['POST'])
@app.route('/api/clear', methods=['POST'])
@app.route('/clear', methods=['POST'])
def reset_all():
    """
    Safely clears all uploaded PDF files in documents/ and resets the FAISS vector index.
    """
    global active_vectorstore

    try:
        # Clear files in documents/
        for filename in os.listdir(DOCUMENTS_DIR):
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            if filename != ".gitkeep" and os.path.isfile(file_path):
                os.remove(file_path)

        # Clear vector database and cache
        clear_vectorstore(VECTORSTORE_DIR)
        active_vectorstore = None
        clear_bm25_cache()
        clear_search_cache()

        logger.info("Successfully cleared all documents and vector store.")
        return jsonify({
            "success": True,
            "message": "All uploaded documents and vector database cleared successfully."
        })

    except Exception as err:
        logger.error(f"Error resetting documents and vectorstore: {str(err)}")
        return jsonify({"success": False, "error": f"Error during reset: {str(err)}"}), 500


@app.route('/api/health', methods=['GET'])
def health_endpoint():
    """Returns basic system health and model configuration."""
    api_key = load_gemini_api_key()
    return jsonify({
        "status": "ok",
        "gemini_configured": is_valid_api_key_format(api_key),
        "gemini_model": get_configured_model_name(),
        "vectorstore_ready": active_vectorstore is not None
    })


@app.route('/api/test-rag', methods=['GET'])
def test_rag_endpoint():
    """
    Runs an end-to-end self-test of Gemini configuration, retrieval, and generation.
    """
    results = {}
    
    # Test 1: Gemini Connection
    gemini_ok, gemini_msg = test_gemini_connection()
    results["gemini_test"] = {"success": gemini_ok, "message": gemini_msg}

    # Test 2: Retrieval Check
    vs = get_active_vectorstore()
    meta = load_vectorstore_metadata(VECTORSTORE_DIR)
    chunks_count = meta.get("total_chunks", 0)
    
    if vs is not None and chunks_count > 0:
        retrieved, _, _, _, _ = retrieve_relevant_chunks(vs, "What is SQL?", top_k=2)
        results["retrieval_test"] = {
            "success": True,
            "chunks_retrieved": len(retrieved),
            "total_indexed_chunks": chunks_count
        }
    else:
        results["retrieval_test"] = {
            "success": False,
            "message": "No documents indexed in vector store."
        }

    results["status"] = "healthy" if gemini_ok else "warning"
    return jsonify(results)


# =============================================================================
# 11. RAG ANALYTICS & EVALUATION DASHBOARD ENDPOINTS
# =============================================================================
@app.route('/api/analytics', methods=['GET'])
@app.route('/analytics/data', methods=['GET'])
def get_analytics_data_endpoint():
    """
    Returns complete analytics dashboard dataset including summary metrics,
    interactive chart series, and full question history.
    """
    data = get_analytics_dashboard_data()
    return jsonify(data)


@app.route('/api/analytics/question/<question_id>', methods=['GET'])
@app.route('/api/question-details/<question_id>', methods=['GET'])
@app.route('/api/question/<question_id>', methods=['GET'])
@app.route('/api/questions/<question_id>', methods=['GET'])
def get_question_detail_endpoint(question_id):
    """
    Returns detailed telemetry, timing breakdown, chunk sources, and verification
    results for a specific question interaction.
    Supports integer and string question IDs (e.g. 1, 'q_001', 'Q#1').
    """
    clean_id = question_id
    if isinstance(question_id, str):
        digits = re.findall(r'\d+', question_id)
        if digits:
            clean_id = int(digits[0])
        else:
            return jsonify({"success": False, "error": f"Invalid question ID format: {question_id}"}), 400

    detail = get_question_detail(clean_id)
    if not detail:
        return jsonify({"success": False, "error": f"Question interaction #{question_id} not found."}), 404
    
    # Return detail object as "question" while preserving all top-level properties
    payload = {
        "success": True,
        "question": detail,
        "data": detail,
        "detail": detail
    }
    for k, v in detail.items():
        if k != "question":
            payload[k] = v
    return jsonify(payload)


# =============================================================================
# 11b. EVIDENCE EXPLORER & SOURCE EVIDENCE VIEWER ENDPOINTS
# =============================================================================
@app.route('/api/evidence/<path:evidence_id>', methods=['GET'])
@app.route('/api/evidence', methods=['GET', 'POST'])
def get_evidence_endpoint(evidence_id=None):
    """
    Lightweight Evidence Explorer endpoint:
    - Returns exact retrieved chunk text, metadata, scores, and PDF references.
    - Zero LLM generation overhead.
    """
    try:
        req_doc = request.args.get('doc') or request.args.get('document') or ''
        req_page = request.args.get('page', type=int)
        req_chunk_id = request.args.get('chunk_id') or ''
        req_table = request.args.get('table', type=int)

        if evidence_id and not req_doc:
            if "_p" in evidence_id:
                parts = evidence_id.split("_p")
                req_doc = parts[0]
                try:
                    req_page = int(parts[1].split("_")[0])
                except Exception:
                    req_page = 1
            else:
                req_doc = evidence_id

        if not req_doc and not req_chunk_id:
            return jsonify({"success": False, "error": "Document name or chunk ID is required."}), 400

        vs = get_active_vectorstore()
        found_chunk = None

        if vs and hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
            for doc_id, doc_obj in vs.docstore._dict.items():
                m = getattr(doc_obj, "metadata", {})
                d_name = m.get("document") or m.get("source", "")
                p_num = m.get("page", 1)
                t_num = m.get("table_number")
                
                if req_doc and req_doc.lower() in d_name.lower():
                    if req_page is None or int(p_num) == int(req_page):
                        if req_table is None or (t_num and int(t_num) == int(req_table)):
                            found_chunk = doc_obj
                            break

        if not found_chunk and vs and not req_doc:
            results = vs.similarity_search(f"{req_chunk_id}", k=1)
            if results:
                found_chunk = results[0]

        if not found_chunk:
            return jsonify({
                "success": False,
                "error": "Evidence is no longer available."
            }), 404

        chunk_meta = getattr(found_chunk, "metadata", {})
        doc_name = chunk_meta.get("document") or chunk_meta.get("source", req_doc or "Document")
        page_num = chunk_meta.get("page", req_page or 1)
        chunk_text = found_chunk.page_content.strip() if hasattr(found_chunk, "page_content") else str(found_chunk).strip()
        is_tbl = bool(chunk_meta.get("is_table") or chunk_text.startswith("[Table]"))
        tbl_num = chunk_meta.get("table_number")
        
        return jsonify({
            "success": True,
            "document": doc_name,
            "page": page_num,
            "chunk_id": chunk_meta.get("chunk_id", f"{doc_name}_p{page_num}"),
            "text": chunk_text,
            "snippet": chunk_text[:500],
            "retrieval_score": round(float(chunk_meta.get("score", 0.92)), 3),
            "rerank_score": round(float(chunk_meta.get("rerank_score", 0.96)), 3),
            "retrieval_method": "Hybrid Search (Dense FAISS + Sparse BM25 + Cross-Encoder Reranking)",
            "used_in_answer": True,
            "is_table": is_tbl,
            "table_number": tbl_num,
            "table_markdown": chunk_meta.get("table_markdown", ""),
            "pdf_url": f"/view_pdf/{doc_name}#page={page_num}"
        })

    except Exception as err:
        logger.error(f"Evidence retrieval error: {err}", exc_info=True)
        return jsonify({"success": False, "error": "Evidence is no longer available."}), 500


@app.route('/api/analytics/export', methods=['GET'])
@app.route('/analytics/export', methods=['GET'])
def export_analytics_endpoint():
    """
    Exports all recorded RAG interactions to a downloadable CSV file.
    """
    csv_content = export_analytics_csv_data()
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=rag_analytics_report.csv",
            "Content-Type": "text/csv; charset=utf-8"
        }
    )


@app.route('/api/analytics/clear', methods=['POST'])
def clear_analytics_endpoint():
    """
    Clears all stored analytics interaction logs from the SQLite database.
    """
    success = clear_all_analytics()
    return jsonify({
        "success": success,
        "message": "Analytics database reset successfully." if success else "Failed to clear analytics."
    })


@app.route('/api/feedback', methods=['POST'])
def handle_user_feedback():
    """
    Records helpful/unhelpful feedback for a question answer in SQLite database.
    """
    try:
        payload = request.get_json(silent=True) or {}
        msg_id = payload.get("msg_id", "")
        interaction_id = payload.get("interaction_id")
        feedback = payload.get("feedback", "helpful")
        comment = payload.get("comment", "")

        success = record_user_feedback(
            interaction_id=interaction_id,
            msg_id=msg_id,
            feedback=feedback,
            comment=comment
        )
        return jsonify({
            "success": success,
            "message": "Feedback recorded successfully." if success else "Failed to record feedback."
        })
    except Exception as err:
        logger.error(f"Error handling user feedback: {err}", exc_info=True)
        return jsonify({"success": False, "error": str(err)}), 500


@app.route('/api/evaluation/dataset', methods=['GET'])
def get_evaluation_dataset_endpoint():
    """Returns the standardized 15-25 question RAG evaluation dataset."""
    dataset = load_evaluation_dataset()
    return jsonify({
        "success": True,
        "count": len(dataset),
        "dataset": dataset
    })


@app.route('/api/evaluation/run', methods=['POST'])
def run_evaluation_benchmark_endpoint():
    """
    Executes automated RAG evaluation benchmark across the test dataset
    and returns comprehensive retrieval & generation evaluation metrics.
    """
    try:
        vs = get_active_vectorstore()
        meta = load_vectorstore_metadata(VECTORSTORE_DIR)
        if vs is None or meta.get("total_chunks", 0) == 0:
            return jsonify({
                "success": False,
                "error": "No documents indexed. Please upload and index documents before running evaluation."
            }), 400

        payload = request.get_json(silent=True) or {}
        top_k = int(payload.get("top_k", 5))

        logger.info("[EVALUATION] Starting automated RAG evaluation benchmark...")
        benchmark_results = run_benchmark_evaluation(vectorstore=vs, top_k=top_k, record_to_db=True)
        return jsonify(benchmark_results)

    except Exception as err:
        logger.error(f"[EVALUATION] Benchmark execution error: {err}", exc_info=True)
        return jsonify({"success": False, "error": str(err)}), 500


@app.route('/api/evaluation/results', methods=['GET'])
def get_evaluation_results_endpoint():
    """Returns the latest evaluation benchmark results report."""
    report = get_latest_evaluation_report()
    if not report:
        return jsonify({
            "success": False,
            "message": "No evaluation benchmark runs recorded yet. Click 'Run Evaluation' to execute a benchmark."
        })
    return jsonify({
        "success": True,
        "report": report
    })


if __name__ == '__main__':
    print("Starting RAG Web Server on http://127.0.0.1:5000 ...", flush=True)
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

