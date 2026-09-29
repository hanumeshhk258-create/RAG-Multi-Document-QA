import os
import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_empty_state():
    print("=" * 80)
    print("TESTING FULL REMOVAL TO ZERO DOCUMENTS & RE-INGESTION")
    print("=" * 80)

    # 1. Get initial status
    status = requests.get(f"{BASE_URL}/api/status").json()
    docs = [d["filename"] for d in status.get("documents", [])]
    print(f"Starting with {len(docs)} documents: {docs}")

    # 2. Remove all documents one-by-one
    for d in docs:
        del_res = requests.post(f"{BASE_URL}/api/remove-document", json={"filename": d}, timeout=15).json()
        assert del_res.get("success") is True, f"Failed to remove {d}"
        print(f"  Removed {d}. Remaining: {del_res.get('remaining_documents')}")

    # 3. Check status at 0 documents
    status_empty = requests.get(f"{BASE_URL}/api/status").json()
    print(f"Empty state status: total_documents={status_empty.get('total_documents')}, chunks={status_empty.get('total_chunks')}, ready={status_empty.get('vectorstore_ready')}")
    assert status_empty.get("total_documents") == 0, "Expected 0 documents"
    assert status_empty.get("total_chunks") == 0, "Expected 0 chunks"
    assert status_empty.get("vectorstore_ready") is False, "Expected vectorstore_ready=False"

    # 4. Querying at 0 documents should gracefully reject without crash
    q_res = requests.post(f"{BASE_URL}/api/chat", json={"question": "What is SQL?"}, timeout=10)
    print(f"Query at 0 documents response code: {q_res.status_code}")
    assert q_res.status_code == 400, "Expected 400 error when querying with 0 documents"
    print(f"  [OK] Graceful rejection: {q_res.json().get('error')}")

    # 5. Restore and re-index all sample documents
    print("\n--- Restoring sample PDFs ---")
    from create_sample_pdfs import create_simple_pdf
    import create_sample_pdfs
    os.system("python create_sample_pdfs.py")
    idx_res = requests.post(f"{BASE_URL}/api/index", timeout=30).json()
    print(f"Re-indexed {idx_res.get('documents')} documents into {idx_res.get('chunks')} chunks.")
    assert idx_res.get("success") is True

    status_restored = requests.get(f"{BASE_URL}/api/status").json()
    assert status_restored.get("total_documents") >= 4
    print("  [OK] System fully restored to healthy indexed state.")
    print("=" * 80)
    print("EMPTY STATE & RE-INGESTION TEST PASSED 100%!")
    print("=" * 80)

if __name__ == "__main__":
    test_empty_state()
