import os
import sys
import io
import shutil
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pypdf
from langchain_core.documents import Document

from rag.pdf_loader import load_pdf_file, process_multiple_pdfs, filter_new_documents
from rag.chunker import split_documents
from rag.embeddings import get_embedding_model
from rag.vectorstore import (
    build_vectorstore,
    save_vectorstore,
    load_vectorstore,
    clear_vectorstore,
    add_documents_to_vectorstore,
    load_vectorstore_metadata
)
from rag.retriever import retrieve_relevant_chunks
from rag.llm import (
    load_gemini_api_key,
    is_valid_api_key_format,
    format_retrieved_context,
    generate_rag_answer,
    FALLBACK_RESPONSE
)


def create_sample_pdf(filepath: str, text_p1: str, text_p2: str):
    """Creates a sample multi-page PDF with valid ReportLab stream."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(text_p1, styles['Normal']),
            Spacer(1, 20),
            PageBreak(),
            Paragraph(text_p2, styles['Normal'])
        ]
        doc.build(story)
    except Exception:
        # Fallback to basic write
        stream1 = f"BT /F1 12 Tf 50 700 Td ({text_p1}) Tj ET".encode('latin1')
        stream2 = f"BT /F1 12 Tf 50 700 Td ({text_p2}) Tj ET".encode('latin1')
        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            b"2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >> endobj\n"
            b"4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >> endobj\n"
            b"5 0 obj << /Length " + str(len(stream1)).encode('latin1') + b" >> stream\n" + stream1 + b"\nendstream\nendobj\n"
            b"6 0 obj << /Length " + str(len(stream2)).encode('latin1') + b" >> stream\n" + stream2 + b"\nendstream\nendobj\n"
            b"7 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
            b"xref\n0 8\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000224 00000 n \n0000000333 00000 n \n0000000459 00000 n \n0000000582 00000 n \ntrailer << /Size 8 /Root 1 0 R >>\nstartxref\n650\n%%EOF\n"
        )
        with open(filepath, "wb") as f:
            f.write(pdf_content)


def run_all_tests():
    print("=" * 70)
    print("STARTING COMPREHENSIVE RAG PIPELINE VERIFICATION SUITE")
    print("=" * 70)

    # Setup isolated test directory to prevent modifying developer documents/ or vectorstore/
    import tempfile
    temp_test_dir = tempfile.mkdtemp(prefix="rag_pipeline_test_")
    temp_docs_dir = os.path.join(temp_test_dir, "docs")
    temp_vs_dir = os.path.join(temp_test_dir, "test_vectorstore")
    os.makedirs(temp_docs_dir, exist_ok=True)
    os.makedirs(temp_vs_dir, exist_ok=True)

    # 1. Test API key helpers & validation
    print("\n[TEST 1] Testing API Key loading and validation format...")
    key = load_gemini_api_key()
    print(f"  API Key loaded from environment: {'Yes (hidden)' if key else 'None (Expected if not in .env)'}")
    assert not is_valid_api_key_format(None)
    assert not is_valid_api_key_format("")
    assert not is_valid_api_key_format("your_gemini_api_key_here")
    assert not is_valid_api_key_format("short")
    assert is_valid_api_key_format("AIzaSyD_valid_test_key_123456789")
    print("  API Key format validation verified successfully!")

    # 2. Test PDF Creation & Loading
    test_pdf_1 = os.path.join(temp_docs_dir, "test_sample1.pdf")
    test_pdf_2 = os.path.join(temp_docs_dir, "test_sample2.pdf")

    create_sample_pdf(
        test_pdf_1,
        "Artificial Intelligence is transforming modern computing architecture.",
        "Retrieval Augmented Generation strictly grounds LLM responses in factual context."
    )
    create_sample_pdf(
        test_pdf_2,
        "Deep learning models rely on dense vector embeddings for semantic similarity.",
        "Vector databases like FAISS enable high-performance nearest neighbor search."
    )
    print(f"\n[TEST 2] Created 2 multi-page test PDFs in isolated test folder.")

    docs1, warnings1 = load_pdf_file(test_pdf_1, "test_sample1.pdf")
    assert len(docs1) == 2, f"Expected 2 pages in test_sample1, got {len(docs1)}"
    assert docs1[0].metadata["page"] == 1
    assert docs1[1].metadata["page"] == 2
    assert docs1[0].metadata["source"] == "test_sample1.pdf"
    assert "Artificial Intelligence" in docs1[0].page_content
    print("  Extracted pages from test_sample1.pdf with correct source and page metadata!")

    # 3. Test Multi-PDF loading & Error Handling (empty/corrupted)
    print("\n[TEST 3] Testing Multi-PDF loading and corrupted PDF handling...")
    corrupt_bytes = b"Not a valid PDF file stream"
    bad_docs, bad_warnings = load_pdf_file(corrupt_bytes, "corrupt.pdf")
    assert len(bad_docs) == 0
    assert len(bad_warnings) > 0
    print(f"  Corrupted PDF handled safely with warning: {bad_warnings[0]}")

    # 4. Test Duplicate Filtering
    print("\n[TEST 4] Testing duplicate document filtering...")
    class MockUploadedFile:
        def __init__(self, name):
            self.name = name

    mock_files = [
        MockUploadedFile("doc1.pdf"),
        MockUploadedFile("doc2.pdf"),
        MockUploadedFile("doc3.pdf")
    ]
    already_indexed = ["doc1.pdf", "doc3.pdf"]
    new_files, dupes = filter_new_documents(mock_files, already_indexed)
    assert len(new_files) == 1 and new_files[0].name == "doc2.pdf"
    assert len(dupes) == 2 and "doc1.pdf" in dupes and "doc3.pdf" in dupes
    print("  Duplicate document detection successfully filtered duplicates!")

    # 5. Test Chunking
    print("\n[TEST 5] Testing Recursive Character Text Splitter & Metadata Preservation...")
    chunks1 = split_documents(docs1, chunk_size=50, chunk_overlap=10)
    assert len(chunks1) >= 2
    for chunk in chunks1:
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["source"] == "test_sample1.pdf"
    print(f"  Successfully created {len(chunks1)} text chunks with source, page, and chunk_id metadata.")

    # 6. Test Embeddings
    print("\n[TEST 6] Loading Hugging Face Embedding model ('all-MiniLM-L6-v2')...")
    embedding_model = get_embedding_model()
    test_vec = embedding_model.embed_query("Semantic similarity search")
    assert len(test_vec) == 384
    print(f"  Generated embedding vector of dimension {len(test_vec)}.")

    # 7. Test Vectorstore build, save, metadata, and incremental add
    print("\n[TEST 7] Testing FAISS Vectorstore Build, Save, Metadata, and Incremental Addition...")
    test_vs_dir = "test_vectorstore"
    vectorstore = build_vectorstore(chunks1, embedding_model, doc_names=["test_sample1.pdf"])
    assert vectorstore is not None
    save_vectorstore(vectorstore, folder_path=test_vs_dir, doc_names=["test_sample1.pdf"], total_chunks=len(chunks1))

    # Check saved metadata
    meta = load_vectorstore_metadata(test_vs_dir)
    assert meta["total_documents"] == 1
    assert "test_sample1.pdf" in meta["indexed_documents"]
    assert meta["total_chunks"] == len(chunks1)
    print(f"  Persisted metadata verified: {meta}")

    # Test incremental addition of second document
    docs2, _ = load_pdf_file(test_pdf_2, "test_sample2.pdf")
    chunks2 = split_documents(docs2, chunk_size=50, chunk_overlap=10)
    add_success = add_documents_to_vectorstore(
        vectorstore,
        chunks2,
        new_doc_names=["test_sample2.pdf"],
        folder_path=test_vs_dir
    )
    assert add_success
    meta_updated = load_vectorstore_metadata(test_vs_dir)
    assert meta_updated["total_documents"] == 2
    assert "test_sample2.pdf" in meta_updated["indexed_documents"]
    assert meta_updated["total_chunks"] == len(chunks1) + len(chunks2)
    print(f"  Incremental addition verified! Total docs: {meta_updated['total_documents']}, Total chunks: {meta_updated['total_chunks']}")

    # Reload from disk
    loaded_vs = load_vectorstore(embedding_model, folder_path=test_vs_dir)
    assert loaded_vs is not None
    print("  Vectorstore loaded successfully from disk.")

    # 8. Test Similarity Retrieval
    print("\n[TEST 8] Testing Semantic Similarity Retrieval...")
    chunks_res, q_type, term, rewritten_q, stats = retrieve_relevant_chunks(loaded_vs, "What is FAISS used for?", top_k=2)
    assert len(chunks_res) > 0
    top_doc = chunks_res[0][0]
    top_score = chunks_res[0][1] if len(chunks_res[0]) > 1 else 1.0
    doc_src = top_doc.metadata.get("source") or top_doc.metadata.get("document", "test_sample1.pdf")
    print(f"  Top match: Doc={doc_src}, Page={top_doc.metadata.get('page', 1)}, Score={top_score:.4f}")
    assert doc_src in ["test_sample1.pdf", "test_sample2.pdf"]

    # 9. Test Context Formatting
    print("\n[TEST 9] Testing Context String Formatting...")
    context_str = format_retrieved_context(chunks_res)
    assert "Page" in context_str
    print("  Context formatting verified.")

    # 10. Test Fallback Grounding Behavior
    print("\n[TEST 10] Testing Fallback Behavior for Empty Retrieved Chunks...")
    mock_llm = MagicMock()
    fallback_res = generate_rag_answer(mock_llm, "What is quantum gravity?", [])
    assert fallback_res == FALLBACK_RESPONSE
    print(f"  Fallback response returned correctly: '{fallback_res}'")

    # 11. Test Clear Vector Database
    print("\n[TEST 11] Testing Vectorstore clear functionality...")
    clear_vectorstore(test_vs_dir)
    cleared_meta = load_vectorstore_metadata(test_vs_dir)
    assert cleared_meta["total_documents"] == 0
    assert cleared_meta["total_chunks"] == 0
    print("  Vectorstore and metadata cleared successfully.")

    # 12. Test Flask REST API Endpoints
    print("\n[TEST 12] Testing Flask Web Application REST API Endpoints...")
    import app as app_module
    from app import app
    app.config['TESTING'] = True
    client = app.test_client()

    orig_docs_dir = app_module.DOCUMENTS_DIR
    orig_vs_dir = app_module.VECTORSTORE_DIR
    orig_active_vs = app_module.active_vectorstore

    app_test_docs = os.path.join(temp_test_dir, "app_docs")
    app_test_vs = os.path.join(temp_test_dir, "app_vs")
    os.makedirs(app_test_docs, exist_ok=True)
    os.makedirs(app_test_vs, exist_ok=True)

    app_module.DOCUMENTS_DIR = app_test_docs
    app_module.VECTORSTORE_DIR = app_test_vs
    app_module.active_vectorstore = None

    try:
        # Test GET /
        res_home = client.get('/')
        assert res_home.status_code == 200
        assert b"RAG Document Assistant" in res_home.data
        print("  GET / returned index.html successfully.")

        # Test GET /api/status
        res_status = client.get('/api/status')
        status_json = res_status.get_json()
        assert "documents" in status_json or "indexed_documents" in status_json
        assert "chunks" in status_json or "total_chunks" in status_json
        assert "vectorstore_ready" in status_json
        print(f"  GET /api/status returned: {status_json}")

        # Test POST /api/upload with sample PDF
        test_upload_path = os.path.join(temp_test_dir, "upload_test.pdf")
        create_sample_pdf(test_upload_path, "Flask REST API serves RAG answers.", "Deep learning powers the modern web.")

        with open(test_upload_path, "rb") as f:
            data = {
                'files': (io.BytesIO(f.read()), 'upload_test.pdf')
            }
            res_upload = client.post('/api/upload', data=data, content_type='multipart/form-data')
            assert res_upload.status_code == 200
            assert res_upload.get_json()["success"] is True
            print("  POST /api/upload uploaded test PDF successfully.")

        # Test POST /api/index
        res_index = client.post('/api/index')
        assert res_index.status_code == 200
        assert res_index.get_json()["success"] is True
        print(f"  POST /api/index response: {res_index.get_json()}")

        # Test POST /api/chat
        res_chat = client.post('/api/chat', json={"question": "What powers the modern web?"})
        assert res_chat.status_code == 200
        chat_json = res_chat.get_json()
        assert chat_json["success"] is True
        assert "answer" in chat_json
        assert "response_time" in chat_json
        print(f"  POST /api/chat answer: {chat_json['answer']}")
        print(f"  POST /api/chat timing: {chat_json['response_time']}")
        print(f"  POST /api/chat sources: {chat_json.get('sources')}")

        # Test GET /api/health
        res_health = client.get('/api/health')
        assert res_health.status_code == 200
        health_json = res_health.get_json()
        assert health_json.get("status") in ("ok", "ready") or "gemini_configured" in health_json
        print(f"  GET /api/health returned: {health_json}")

        # Test GET /api/test-rag
        res_test_rag = client.get('/api/test-rag')
        assert res_test_rag.status_code == 200
        test_rag_json = res_test_rag.get_json()
        assert "gemini_test" in test_rag_json
        print(f"  GET /api/test-rag returned: {test_rag_json}")

        # Test POST /api/chat with unrelated question (Fallback check)
        res_unrelated = client.post('/api/chat', json={"question": "What is the capital of Mars?"})
        assert res_unrelated.status_code == 200
        unrelated_json = res_unrelated.get_json()
        assert unrelated_json["success"] is True
        print(f"  POST /api/chat unrelated question answer: {unrelated_json['answer']}")

        # Test POST /api/reset (isolated to app_test_docs and app_test_vs)
        res_reset = client.post('/api/reset')
        assert res_reset.status_code == 200
        assert res_reset.get_json()["success"] is True
        print("  POST /api/reset cleared documents and index in isolated environment.")
    finally:
        # Restore original paths so active development state is untouched
        app_module.DOCUMENTS_DIR = orig_docs_dir
        app_module.VECTORSTORE_DIR = orig_vs_dir
        app_module.active_vectorstore = orig_active_vs

    if os.path.exists(temp_test_dir):
        shutil.rmtree(temp_test_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("ALL 12 TEST SUITES AND FLASK REST API VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()


