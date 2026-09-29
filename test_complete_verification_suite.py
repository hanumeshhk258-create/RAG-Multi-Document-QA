import os
import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_full_pipeline():
    print("=" * 80)
    print("COMPREHENSIVE RAG PIPELINE & DOCUMENT DELETION TEST SUITE")
    print("=" * 80)

    # 1. System Status
    status = requests.get(f"{BASE_URL}/api/status", timeout=10).json()
    docs = [d["filename"] for d in status.get("documents", [])]
    print(f"System status: {status.get('status')} | {len(docs)} documents | {status.get('total_chunks')} chunks")
    assert status.get("vectorstore_ready") is True, "Vector store is not ready"

    # 2. Test Question 1: "What is the Waterfall Model?"
    print("\n--- TEST 1: 'What is the Waterfall Model?' ---")
    t0 = time.perf_counter()
    r1 = requests.post(f"{BASE_URL}/api/chat", json={"question": "What is the Waterfall Model?"}, timeout=25).json()
    t1 = time.perf_counter() - t0
    ans1 = r1.get("answer", "")
    sources1 = r1.get("sources", [])
    eval1 = r1.get("evaluation", {})
    print(f"  Time: {t1:.2f}s")
    print(f"  Final Answer Preview: {ans1[:180]}...")
    print(f"  Sources: {[s.get('document') + ' (P.' + str(s.get('page')) + ')' for s in sources1]}")
    print(f"  Confidence: {eval1.get('confidence')} | Grounding: {eval1.get('groundedness_label')} | Supported Claims: {eval1.get('supported_claims')}")
    assert len(ans1) > 20, "Final answer is empty or too short!"
    assert any("CommerceOS" in s.get("document", "") for s in sources1), "Expected CommerceOS source for Waterfall model"
    assert not ans1.startswith("- "), "Answer should not be a raw fragmented bullet list!"
    print("  [OK] Test 1 Passed.")

    # 3. Test Question 2: "What is ACID?"
    print("\n--- TEST 2: 'What is ACID?' ---")
    t0 = time.perf_counter()
    r2 = requests.post(f"{BASE_URL}/api/chat", json={"question": "What is ACID in DBMS?"}, timeout=25).json()
    t2 = time.perf_counter() - t0
    ans2 = r2.get("answer", "")
    sources2 = r2.get("sources", [])
    print(f"  Time: {t2:.2f}s")
    print(f"  Final Answer Preview: {ans2[:180]}...")
    print(f"  Sources: {[s.get('document') for s in sources2]}")
    assert "DBMS_Notes.pdf" in [s.get("document") for s in sources2]
    print("  [OK] Test 2 Passed.")

    # 4. Test Question 3: "What is Key Escrow?"
    print("\n--- TEST 3: 'What is Key Escrow?' (from BIS703-module-2-pdf.pdf) ---")
    t0 = time.perf_counter()
    r3 = requests.post(f"{BASE_URL}/api/chat", json={"question": "What is Key Escrow?"}, timeout=25).json()
    t3 = time.perf_counter() - t0
    ans3 = r3.get("answer", "")
    sources3 = r3.get("sources", [])
    print(f"  Time: {t3:.2f}s")
    print(f"  Final Answer Preview: {ans3[:180]}...")
    print(f"  Sources: {[s.get('document') for s in sources3]}")
    assert any("BIS703" in s.get("document", "") for s in sources3)
    print("  [OK] Test 3 Passed.")

    # 5. Test Question 4: Out-of-domain question
    print("\n--- TEST 4: Out-of-domain question ('What is the recipe for baking chocolate cheesecake?') ---")
    r4 = requests.post(f"{BASE_URL}/api/chat", json={"question": "What is the recipe for baking chocolate cheesecake?"}, timeout=25).json()
    ans4 = r4.get("answer", "")
    print(f"  Response: {ans4}")
    assert "not contain enough information" in ans4.lower() or "not find sufficient information" in ans4.lower() or "could not find" in ans4.lower()
    print("  [OK] Handled out-of-domain query cleanly without hallucination.")

    # 6. Test Document Deletion & Retrieval Purge
    print("\n--- TEST 5: Complete Document Deletion Flow (SQL_CheatSheet.pdf) ---")
    del_res = requests.post(f"{BASE_URL}/api/remove-document", json={"filename": "SQL_CheatSheet.pdf"}, timeout=20).json()
    assert del_res.get("success") is True, f"Deletion failed: {del_res}"
    print(f"  Deleted 'SQL_CheatSheet.pdf'. Remaining documents: {del_res.get('remaining_documents')}")

    # Querying after deletion: SQL_CheatSheet.pdf MUST NOT be retrieved
    r_after = requests.post(f"{BASE_URL}/api/chat", json={"question": "What is SQL DDL and DML commands?"}, timeout=25).json()
    sources_after = [s.get("document") for s in r_after.get("sources", [])]
    print(f"  Sources after deletion: {set(sources_after)}")
    assert "SQL_CheatSheet.pdf" not in sources_after, "Deleted document was still retrieved!"
    print("  [OK] Deleted document was completely purged from FAISS and retrieval.")

    # Restore sample
    from create_sample_pdfs import create_simple_pdf
    import create_sample_pdfs
    os.system("python create_sample_pdfs.py")
    requests.post(f"{BASE_URL}/api/index", timeout=30)
    print("  [OK] Re-indexed test environment.")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    test_full_pipeline()
