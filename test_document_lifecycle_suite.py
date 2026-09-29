import os
import sys
import time
import json
import requests
import io

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:5000"
DOCS_DIR = "documents"

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_pdf_bytes(title: str, lines: list) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 750
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, title)
    y -= 30
    c.setFont("Helvetica", 10)
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    buf.seek(0)
    return buf.read()

def run_lifecycle_tests():
    print("=" * 80)
    print("RUNNING DOCUMENT LIFECYCLE & STATUS MANAGEMENT TEST SUITE")
    print("=" * 80)

    # Clean initial state
    requests.post(f"{BASE_URL}/api/reset")
    time.sleep(1)

    # -------------------------------------------------------------------------
    # TEST 1: Upload one PDF -> Expected: Pending
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: Upload one PDF (Doc_Alpha) -> Expected status: Pending ---")
    doc_alpha_bytes = create_pdf_bytes(
        "DOCUMENT ALPHA - CIPHER SPECIFICATION",
        [
            "The master security key for Alpha Protocol is CIPHER-77441.",
            "Alpha protocol utilizes AES-GCM-256 authenticated encryption.",
            "All sessions expire after exactly 3600 seconds of inactivity."
        ]
    )

    upload_res = requests.post(
        f"{BASE_URL}/api/upload",
        files={"files": ("Doc_Alpha.pdf", doc_alpha_bytes, "application/pdf")}
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    upload_data = upload_res.json()
    print(f"[OK] Uploaded response: {upload_data}")
    
    # Check status endpoint
    status_res = requests.get(f"{BASE_URL}/api/status").json()
    print(f"[OK] System status after upload: Total={status_res.get('total_documents')}, Pending={status_res.get('pending_count')}, Indexed={status_res.get('indexed_count')}")
    
    doc_alpha = next((d for d in status_res.get("documents", []) if d["filename"] == "Doc_Alpha.pdf"), None)
    assert doc_alpha is not None, "Doc_Alpha.pdf not found in documents list!"
    assert doc_alpha["status"] == "Pending", f"Expected Doc_Alpha status 'Pending', got '{doc_alpha['status']}'"
    assert doc_alpha["indexed"] is False, "Pending document must have indexed=False"
    assert status_res.get("pending_count") == 1, "Pending count should be 1"
    assert status_res.get("indexed_count") == 0, "Indexed count should be 0"
    print("[PASS] TEST 1 PASSED: Uploaded document starts in 'Pending' status and is not searchable.")

    # -------------------------------------------------------------------------
    # TEST 2: Click Index Documents -> Expected: Pending -> Indexing -> Ready
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: Click Index Documents -> Expected status: Ready ---")
    index_res = requests.post(f"{BASE_URL}/api/index", json={"selected_documents": ["Doc_Alpha.pdf"]})
    assert index_res.status_code == 200, f"Indexing failed: {index_res.text}"
    index_data = index_res.json()
    print(f"[OK] Index response: {index_data.get('message')}")
    assert index_data.get("indexed_count") == 1, "Should have indexed 1 document"

    status_res = requests.get(f"{BASE_URL}/api/status").json()
    doc_alpha = next((d for d in status_res.get("documents", []) if d["filename"] == "Doc_Alpha.pdf"), None)
    assert doc_alpha["status"] == "Ready", f"Expected Doc_Alpha status 'Ready', got '{doc_alpha['status']}'"
    assert doc_alpha["indexed"] is True, "Ready document must have indexed=True"
    assert doc_alpha["chunks"] > 0, "Ready document must have chunks > 0"
    assert status_res.get("indexed_count") == 1, "Indexed count should be 1"
    assert status_res.get("pending_count") == 0, "Pending count should be 0"
    assert status_res.get("vectorstore_ready") is True, "Vectorstore must be ready"
    print(f"[PASS] TEST 2 PASSED: Doc_Alpha indexed to {doc_alpha['chunks']} chunks with status 'Ready'.")

    # -------------------------------------------------------------------------
    # TEST 3: Ask a question -> Expected: Answer comes from Ready document
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: Ask a question -> Expected: Answer from Doc_Alpha ---")
    chat_res = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is the master security key for Alpha Protocol?",
        "selected_documents": ["Doc_Alpha.pdf"]
    })
    assert chat_res.status_code == 200, f"Chat failed: {chat_res.text}"
    chat_data = chat_res.json()
    answer = chat_data.get("final_verified_answer") or chat_data.get("answer") or ""
    sources = [s.get("document") for s in chat_data.get("sources", [])]
    print(f"[OK] Retrieved sources: {sources}")
    print(f"[OK] Answer snippet: {answer[:120]}...")
    assert "Doc_Alpha.pdf" in sources, "Doc_Alpha.pdf must be in retrieved sources"
    assert "CIPHER-77441" in answer, "Answer must contain CIPHER-77441 from Doc_Alpha"
    print("[PASS] TEST 3 PASSED: Answer generated correctly using Ready document.")

    # -------------------------------------------------------------------------
    # TEST 4: Upload second PDF but don't index it -> Expected: Remains Pending and excluded from search
    # -------------------------------------------------------------------------
    print("\n--- TEST 4: Upload Doc_Beta without indexing -> Expected: Pending and not in search ---")
    doc_beta_bytes = create_pdf_bytes(
        "DOCUMENT BETA - QUANTUM TELEMETRY",
        [
            "The quantum teleportation frequency is 432.85 MHz.",
            "Beta telemetry transmits sensor packets via entangled photon channels.",
            "The secret password for Beta system is NEBULA-5522."
        ]
    )

    upload_res_2 = requests.post(
        f"{BASE_URL}/api/upload",
        files={"files": ("Doc_Beta.pdf", doc_beta_bytes, "application/pdf")}
    )
    assert upload_res_2.status_code == 200

    status_res = requests.get(f"{BASE_URL}/api/status").json()
    doc_beta = next((d for d in status_res.get("documents", []) if d["filename"] == "Doc_Beta.pdf"), None)
    assert doc_beta["status"] == "Pending", f"Doc_Beta must be Pending, got '{doc_beta['status']}'"
    assert status_res.get("total_documents") == 2
    assert status_res.get("indexed_count") == 1
    assert status_res.get("pending_count") == 1

    # Asking question answered only by Doc_Beta
    chat_res_beta = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is the secret password for Beta system?",
        "selected_documents": ["Doc_Alpha.pdf", "Doc_Beta.pdf"]
    })
    chat_beta_data = chat_res_beta.json()
    sources_beta = [s.get("document") for s in chat_beta_data.get("sources", [])]
    answer_beta = chat_beta_data.get("final_verified_answer") or chat_beta_data.get("answer") or ""
    print(f"[OK] Sources for Beta query before indexing: {sources_beta}")
    print(f"[OK] Answer for Beta query before indexing: {answer_beta}")
    assert not any("Doc_Beta" in s for s in sources_beta), "Pending Doc_Beta must NOT participate in retrieval!"
    assert "NEBULA-5522" not in answer_beta, "Content from pending Doc_Beta must NOT appear in answer!"

    # Testing Search Restriction when only Pending document is selected
    chat_res_only_pending = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is the quantum frequency?",
        "selected_documents": ["Doc_Beta.pdf"]
    })
    assert chat_res_only_pending.status_code == 400
    assert "No indexed documents are available" in chat_res_only_pending.json().get("error", "")
    print(f"[OK] Correct search restriction error returned when selecting only pending documents: {chat_res_only_pending.json().get('error')}")
    print("[PASS] TEST 4 PASSED: Pending document strictly excluded from RAG retrieval.")

    # -------------------------------------------------------------------------
    # TEST 5: Index second PDF -> Expected: Ready and searchable
    # -------------------------------------------------------------------------
    print("\n--- TEST 5: Index second PDF (Doc_Beta) -> Expected: Ready and searchable ---")
    index_res_2 = requests.post(f"{BASE_URL}/api/index", json={"selected_documents": ["Doc_Beta.pdf"]})
    assert index_res_2.status_code == 200
    
    status_res = requests.get(f"{BASE_URL}/api/status").json()
    doc_beta = next((d for d in status_res.get("documents", []) if d["filename"] == "Doc_Beta.pdf"), None)
    assert doc_beta["status"] == "Ready"
    assert status_res.get("indexed_count") == 2
    assert status_res.get("pending_count") == 0

    # Query Beta now
    chat_res_beta_2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is the secret password for Beta system?",
        "selected_documents": ["Doc_Beta.pdf"]
    })
    assert chat_res_beta_2.status_code == 200
    chat_data_b2 = chat_res_beta_2.json()
    sources_b2 = [s.get("document") for s in chat_data_b2.get("sources", [])]
    answer_b2 = chat_data_b2.get("final_verified_answer") or chat_data_b2.get("answer") or ""
    print(f"[OK] Sources after indexing Doc_Beta: {sources_b2}")
    print(f"[OK] Answer after indexing Doc_Beta: {answer_b2}")
    assert any("Doc_Beta" in s for s in sources_b2), "Doc_Beta must now be retrieved"
    assert "NEBULA-5522" in answer_b2, "Answer must contain NEBULA-5522"
    print("[PASS] TEST 5 PASSED: Second PDF successfully indexed and participating in retrieval.")

    # -------------------------------------------------------------------------
    # TEST 6: Delete second PDF using DELETE -> Expected: Purged from search results
    # -------------------------------------------------------------------------
    print("\n--- TEST 6: Delete Doc_Beta -> Expected: Immediately purged from index & search ---")
    del_res = requests.delete(f"{BASE_URL}/api/documents/Doc_Beta.pdf")
    assert del_res.status_code == 200
    print(f"[OK] Delete response: {del_res.json().get('message')}")

    status_res = requests.get(f"{BASE_URL}/api/status").json()
    assert not any(d["filename"] == "Doc_Beta.pdf" for d in status_res.get("documents", []))
    assert status_res.get("total_documents") == 1
    assert status_res.get("indexed_count") == 1

    # Query Beta again -> should not find it
    chat_res_del = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is the secret password for Beta system?"
    })
    chat_data_del = chat_res_del.json()
    sources_del = [s.get("document") for s in chat_data_del.get("sources", [])]
    answer_del = chat_data_del.get("final_verified_answer") or chat_data_del.get("answer") or ""
    assert not any("Doc_Beta" in s for s in sources_del), "Deleted Doc_Beta must NOT appear in sources!"
    assert "NEBULA-5522" not in answer_del, "Deleted Doc_Beta content must NOT appear in answer!"
    print("[PASS] TEST 6 PASSED: Deleted document completely purged from vector store and search.")

    # -------------------------------------------------------------------------
    # TEST 7: Duplicate protection test
    # -------------------------------------------------------------------------
    print("\n--- TEST 7: Duplicate upload protection ---")
    dup_res = requests.post(
        f"{BASE_URL}/api/upload",
        files={"files": ("Doc_Alpha.pdf", doc_alpha_bytes, "application/pdf")}
    )
    assert dup_res.status_code == 200
    dup_data = dup_res.json()
    print(f"[OK] Duplicate upload warnings: {dup_data.get('warnings')}")
    assert any("already indexed" in w.lower() for w in dup_data.get("warnings", [])), "Must warn that document is already indexed"
    print("[PASS] TEST 7 PASSED: Duplicate upload protection prevents duplicate chunk creation.")

    # -------------------------------------------------------------------------
    # TEST 8: Force an indexing error & Partial Indexing
    # -------------------------------------------------------------------------
    print("\n--- TEST 8: Force indexing error -> Expected: Error status, retryable, partial indexing intact ---")
    corrupt_pdf_path = os.path.join(DOCS_DIR, "Corrupt_Doc.pdf")
    with open(corrupt_pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n%corrupted broken bytes without end of stream\n")

    # Refresh status to register corrupt doc as Pending
    status_res = requests.get(f"{BASE_URL}/api/status").json()
    corrupt_doc = next((d for d in status_res.get("documents", []) if d["filename"] == "Corrupt_Doc.pdf"), None)
    assert corrupt_doc is not None
    assert corrupt_doc["status"] == "Pending"

    # Index all pending
    idx_res_err = requests.post(f"{BASE_URL}/api/index")
    assert idx_res_err.status_code == 200
    idx_err_data = idx_res_err.json()
    print(f"[OK] Indexing with error result: indexed={idx_err_data.get('indexed_count')}, failed={idx_err_data.get('failed_count')}")
    assert idx_err_data.get("failed_count") == 1

    status_res = requests.get(f"{BASE_URL}/api/status").json()
    corrupt_doc_status = next((d for d in status_res.get("documents", []) if d["filename"] == "Corrupt_Doc.pdf"), None)
    assert corrupt_doc_status["status"] == "Error"
    assert corrupt_doc_status["error_message"] is not None
    print(f"[OK] Corrupt doc status: {corrupt_doc_status['status']} - Error: {corrupt_doc_status['error_message']}")

    # Check Doc_Alpha is still Ready and searchable (Partial Indexing requirement)
    alpha_status = next((d for d in status_res.get("documents", []) if d["filename"] == "Doc_Alpha.pdf"), None)
    assert alpha_status["status"] == "Ready"
    print("[PASS] TEST 8 PASSED: Failed document gets status 'Error' with error details, while Ready document remains searchable.")

    # Cleanup corrupt doc
    requests.delete(f"{BASE_URL}/api/documents/Corrupt_Doc.pdf")

    print("\n" + "=" * 80)
    print("ALL 8 DOCUMENT LIFECYCLE & STATUS MANAGEMENT TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    run_lifecycle_tests()
