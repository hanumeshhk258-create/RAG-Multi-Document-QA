import os
import sys
import time
import json
import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:5000"
DOCS_DIR = "documents"

def run_tests():
    print("=" * 80)
    print("RUNNING TRUE DOCUMENT DELETION TEST SUITE")
    print("=" * 80)

    # 1. Health check
    res = requests.get(f"{BASE_URL}/api/status")
    assert res.status_code == 200, f"Status check failed: {res.text}"
    print(f"[OK] Health check passed. System status: {res.json().get('status')}")

    # 2. Prepare 3 distinct test PDFs using reportlab
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    doc_a_path = os.path.join(DOCS_DIR, "TestDoc_A.pdf")
    doc_b_path = os.path.join(DOCS_DIR, "TestDoc_B.pdf")
    doc_c_path = os.path.join(DOCS_DIR, "TestDoc_C.pdf")

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    def make_real_pdf(filename, text_lines):
        c = canvas.Canvas(os.path.join(DOCS_DIR, filename), pagesize=letter)
        y = 750
        for line in text_lines:
            c.drawString(72, y, line)
            y -= 25
        c.save()

    make_real_pdf("TestDoc_A.pdf", [
        "DOCUMENT ALPHA - SECRET SPECIFICATION",
        "The secret alpha protocol code is OMEGA-9992.",
        "Alpha protocol ensures end-to-end zero-knowledge key escrow encryption.",
        "This document is uniquely identified as Document Alpha."
    ])
    make_real_pdf("TestDoc_B.pdf", [
        "DOCUMENT BETA - DATABASE ARCHITECTURE",
        "The primary architecture of Beta is relational database normalization.",
        "Third Normal Form (3NF) eliminates transitive functional dependencies.",
        "Beta system utilizes B-Tree indexes for fast point queries."
    ])
    make_real_pdf("TestDoc_C.pdf", [
        "DOCUMENT GAMMA - CLOUD DEPLOYMENT",
        "Gamma deployment runs Kubernetes clusters across three availability zones.",
        "Gamma services scale automatically based on CPU and memory thresholds."
    ])

    # 3. Index documents
    print("\n--- Step 1: Indexing 3 Test Documents ---")
    idx_res = requests.post(f"{BASE_URL}/api/index")
    assert idx_res.status_code == 200, f"Indexing failed: {idx_res.text}"
    idx_data = idx_res.json()
    print(f"[OK] Indexed {idx_data.get('documents')} documents ({idx_data.get('chunks')} chunks).")

    # 4. Ask question about TestDoc_A
    print("\n--- Step 2: Asking question answered ONLY by TestDoc_A ---")
    chat_res = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is the secret alpha protocol code in Document Alpha?"
    })
    assert chat_res.status_code == 200, f"Chat failed: {chat_res.text}"
    chat_data = chat_res.json()
    sources = [s.get("document") for s in chat_data.get("sources", [])]
    answer = chat_data.get("final_verified_answer") or chat_data.get("answer") or ""
    print(f"Sources retrieved: {sources}")
    print(f"Answer snippet: {answer[:120]}...")
    assert any("TestDoc_A" in s for s in sources), "TestDoc_A must be in retrieved sources!"
    print("[OK] TestDoc_A correctly retrieved and cited.")

    # 5. Delete TestDoc_A using DELETE /api/documents/<doc_id>
    print("\n--- Step 3: Deleting TestDoc_A via DELETE /api/documents/TestDoc_A.pdf ---")
    del_res = requests.delete(f"{BASE_URL}/api/documents/TestDoc_A.pdf")
    assert del_res.status_code == 200, f"Delete failed: {del_res.text}"
    del_data = del_res.json()
    print(f"Delete response: {del_data.get('message')}")
    assert del_data.get("success") is True, "Delete response success must be true"

    # Verify physical file deletion
    assert not os.path.exists(doc_a_path), "Physical PDF file for TestDoc_A must be deleted from disk!"
    print("[OK] Physical file 'TestDoc_A.pdf' deleted from storage directory.")

    # 6. Ask the exact same question again
    print("\n--- Step 4: Asking the SAME question again after TestDoc_A deletion ---")
    chat_res_2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is the secret alpha protocol code in Document Alpha?"
    })
    chat_data_2 = chat_res_2.json()
    sources_2 = [s.get("document") for s in chat_data_2.get("sources", [])]
    answer_2 = chat_data_2.get("final_verified_answer") or chat_data_2.get("answer") or ""
    print(f"Sources retrieved after deletion: {sources_2}")
    print(f"Answer after deletion: {answer_2}")
    
    # TestDoc_A must NEVER appear in sources or citations
    assert not any("TestDoc_A" in s for s in sources_2), "Deleted document TestDoc_A must NOT appear in sources!"
    assert "OMEGA-9992" not in answer_2, "Deleted document content must NOT appear in generated answer!"
    print("[OK] Verified: Deleted document is completely purged from FAISS, BM25, and Answer citations.")

    # 7. Ask question about TestDoc_B to verify remaining documents work
    print("\n--- Step 5: Asking question about remaining Document B ---")
    chat_res_3 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is the primary architecture of Beta?"
    })
    assert chat_res_3.status_code == 200
    chat_data_3 = chat_res_3.json()
    sources_3 = [s.get("document") for s in chat_data_3.get("sources", [])]
    print(f"Sources for Beta query: {sources_3}")
    assert any("TestDoc_B" in s for s in sources_3), "TestDoc_B must be retrieved!"
    print("[OK] Remaining Document B functions normally.")

    # 8. Clean up test documents
    requests.delete(f"{BASE_URL}/api/documents/TestDoc_B.pdf")
    requests.delete(f"{BASE_URL}/api/documents/TestDoc_C.pdf")
    print("\n--- Step 6: All test documents cleaned up. ---")

    # 9. Re-index base documents
    requests.post(f"{BASE_URL}/api/index")

    print("\n" + "=" * 80)
    print("ALL DOCUMENT DELETION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
