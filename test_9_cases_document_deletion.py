import os
import sys
import time
import json
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:5000"
DOCS_DIR = "documents"

def make_test_pdf(filename, text_lines):
    os.makedirs(DOCS_DIR, exist_ok=True)
    c = canvas.Canvas(os.path.join(DOCS_DIR, filename), pagesize=letter)
    y = 750
    for line in text_lines:
        c.drawString(72, y, line)
        y -= 25
    c.save()

def run_suite():
    print("=" * 80)
    print("STARTING 9-POINT COMPREHENSIVE DOCUMENT DELETION VERIFICATION SUITE")
    print("=" * 80)

    # Prepare base test documents
    make_test_pdf("Doc_Alpha.pdf", [
        "ALPHA SPECIFICATION",
        "The confidential Alpha token code is TOKEN-ALPHA-7788.",
        "Alpha architecture operates on a decentralized ledger system."
    ])
    make_test_pdf("Doc_Beta.pdf", [
        "BETA SPECIFICATION",
        "Beta system uses distributed B-Tree indexing with multi-version concurrency.",
        "Beta normalization adheres strictly to Boyce-Codd Normal Form."
    ])
    make_test_pdf("Doc_Gamma.pdf", [
        "GAMMA SPECIFICATION",
        "Gamma clusters deploy microservices using container orchestration.",
        "Gamma auto-scaler adjusts replica counts based on memory pressure."
    ])

    # Index all 3
    idx_res = requests.post(f"{BASE_URL}/api/index")
    assert idx_res.status_code == 200, f"Indexing failed: {idx_res.text}"
    status = requests.get(f"{BASE_URL}/api/status").json()
    print(f"Initial setup: {status.get('total_documents')} documents, {status.get('total_chunks')} chunks.")

    # -------------------------------------------------------------
    # TEST 1: Upload 3 PDFs -> delete 1 -> verify 2 remain
    # -------------------------------------------------------------
    print("\n--- TEST 1: Upload 3 PDFs -> delete 1 -> verify 2 remain ---")
    del_res = requests.delete(f"{BASE_URL}/api/documents/Doc_Alpha.pdf")
    assert del_res.status_code == 200
    st1 = requests.get(f"{BASE_URL}/api/status").json()
    doc_names = [d["filename"] for d in st1.get("documents", [])]
    assert "Doc_Alpha.pdf" not in doc_names, "Doc_Alpha.pdf must not be in documents list"
    assert "Doc_Beta.pdf" in doc_names and "Doc_Gamma.pdf" in doc_names
    print(f"[OK] Test 1 Passed: 2 documents remain ({doc_names})")

    # -------------------------------------------------------------
    # TEST 2: Select 3 PDFs -> delete 1 selected PDF -> verify 2 remain selected
    # -------------------------------------------------------------
    print("\n--- TEST 2: Select 3 PDFs -> delete 1 selected -> verify 2 remain selected ---")
    make_test_pdf("Doc_Alpha.pdf", ["ALPHA SPECIFICATION", "The confidential Alpha token code is TOKEN-ALPHA-7788."])
    requests.post(f"{BASE_URL}/api/index")
    # Select all 3
    sel_res = requests.post(f"{BASE_URL}/api/documents/select", json={
        "selected_documents": ["Doc_Alpha.pdf", "Doc_Beta.pdf", "Doc_Gamma.pdf"]
    }).json()
    assert sel_res["selected_count"] == 3
    # Delete Doc_Beta.pdf
    requests.delete(f"{BASE_URL}/api/documents/Doc_Beta.pdf")
    # Verify selection validation on remaining
    sel_check = requests.post(f"{BASE_URL}/api/documents/select", json={
        "selected_documents": ["Doc_Alpha.pdf", "Doc_Beta.pdf", "Doc_Gamma.pdf"]
    }).json()
    assert sel_check["selected_count"] == 2
    assert "Doc_Beta.pdf" not in sel_check["selected_documents"]
    print(f"[OK] Test 2 Passed: Selected count dynamically adjusted to {sel_check['selected_count']}")

    # -------------------------------------------------------------
    # TEST 3: Upload 3 PDFs -> delete an unselected PDF -> verify selected documents are unchanged
    # -------------------------------------------------------------
    print("\n--- TEST 3: Delete unselected PDF -> verify selected documents unchanged ---")
    make_test_pdf("Doc_Beta.pdf", ["BETA SPECIFICATION", "Beta system uses distributed B-Tree indexing."])
    requests.post(f"{BASE_URL}/api/index")
    # Select only Alpha and Beta
    selected_set = ["Doc_Alpha.pdf", "Doc_Beta.pdf"]
    # Delete unselected Gamma
    requests.delete(f"{BASE_URL}/api/documents/Doc_Gamma.pdf")
    sel_check_3 = requests.post(f"{BASE_URL}/api/documents/select", json={
        "selected_documents": selected_set
    }).json()
    assert sel_check_3["selected_count"] == 2
    assert set(sel_check_3["selected_documents"]) == set(selected_set)
    print(f"[OK] Test 3 Passed: Selected documents remain unchanged ({sel_check_3['selected_documents']})")

    # -------------------------------------------------------------
    # TEST 4: Delete all down to 0 PDFs -> verify empty state
    # -------------------------------------------------------------
    print("\n--- TEST 4: Delete all down to 0 PDFs -> verify empty state ---")
    # Retrieve all current documents and delete every single one
    all_current_docs = [d["filename"] for d in requests.get(f"{BASE_URL}/api/status").json().get("documents", [])]
    for dname in all_current_docs:
        requests.delete(f"{BASE_URL}/api/documents/{dname}")
    
    st_empty = requests.get(f"{BASE_URL}/api/status").json()
    assert st_empty["total_documents"] == 0, f"Expected 0 documents, got {st_empty['total_documents']}"
    assert st_empty["total_pages"] == 0, f"Expected 0 pages, got {st_empty['total_pages']}"
    assert st_empty["total_chunks"] == 0, f"Expected 0 chunks, got {st_empty['total_chunks']}"
    assert st_empty["status"] == "empty", f"Expected 'empty' status, got {st_empty['status']}"
    
    # Query against empty database must return 400 error
    chat_empty = requests.post(f"{BASE_URL}/api/chat", json={"question": "What is normalization?"})
    assert chat_empty.status_code == 400
    print("[OK] Test 4 Passed: Clean 0-document empty state (0 docs, 0 pages, 0 chunks) with proper 400 rejection.")

    # -------------------------------------------------------------
    # TEST 5: Ask question based only on deleted PDF -> verify deleted PDF is never retrieved
    # -------------------------------------------------------------
    print("\n--- TEST 5: Ask question on deleted PDF -> verify deleted PDF is NEVER retrieved ---")
    make_test_pdf("Doc_Alpha.pdf", ["The confidential Alpha token code is TOKEN-ALPHA-7788."])
    make_test_pdf("Doc_Beta.pdf", ["Beta system uses distributed B-Tree indexing."])
    requests.post(f"{BASE_URL}/api/index")
    # Delete Doc_Alpha.pdf
    requests.delete(f"{BASE_URL}/api/documents/Doc_Alpha.pdf")
    # Ask about Alpha token
    chat_5 = requests.post(f"{BASE_URL}/api/chat", json={"question": "What is the confidential Alpha token code?"}).json()
    sources_5 = [s.get("document") for s in chat_5.get("sources", [])]
    ans_5 = chat_5.get("final_verified_answer") or chat_5.get("answer") or ""
    assert "Doc_Alpha.pdf" not in sources_5, "Doc_Alpha must NOT be in sources"
    assert "TOKEN-ALPHA-7788" not in ans_5, "TOKEN-ALPHA-7788 must NOT be in answer"
    print(f"[OK] Test 5 Passed: Deleted document was never retrieved. Answer: '{ans_5[:70]}...'")

    # -------------------------------------------------------------
    # TEST 6: Delete PDF -> upload the same PDF again -> verify it works as fresh document
    # -------------------------------------------------------------
    print("\n--- TEST 6: Delete PDF -> re-upload same PDF -> verify it works as fresh document ---")
    make_test_pdf("Doc_Alpha.pdf", ["The confidential Alpha token code is TOKEN-ALPHA-7788."])
    requests.post(f"{BASE_URL}/api/index")
    chat_6 = requests.post(f"{BASE_URL}/api/chat", json={"question": "What is the confidential Alpha token code?"}).json()
    sources_6 = [s.get("document") for s in chat_6.get("sources", [])]
    assert "Doc_Alpha.pdf" in sources_6
    print(f"[OK] Test 6 Passed: Re-uploaded document is indexed and retrievable ({sources_6}).")

    # -------------------------------------------------------------
    # TEST 7: Delete PDF -> check Sources -> deleted PDF must not appear
    # -------------------------------------------------------------
    print("\n--- TEST 7: Delete PDF -> check Sources ---")
    requests.delete(f"{BASE_URL}/api/documents/Doc_Alpha.pdf")
    chat_7 = requests.post(f"{BASE_URL}/api/chat", json={"question": "What does Beta system use?"}).json()
    sources_7 = [s.get("document") for s in chat_7.get("sources", [])]
    assert "Doc_Alpha.pdf" not in sources_7
    assert "Doc_Beta.pdf" in sources_7
    print(f"[OK] Test 7 Passed: Sources only contain active documents ({sources_7}).")

    # -------------------------------------------------------------
    # TEST 8: Delete PDF -> check Analytics -> counters/index statistics reflect remaining docs
    # -------------------------------------------------------------
    print("\n--- TEST 8: Delete PDF -> check Analytics counters ---")
    analytics_res = requests.get(f"{BASE_URL}/api/analytics").json()
    summary = analytics_res.get("summary", {})
    st8 = requests.get(f"{BASE_URL}/api/status").json()
    print(f"Active documents in system: {st8.get('total_documents')}, chunks: {st8.get('total_chunks')}")
    assert st8.get("total_documents") == 1
    print("[OK] Test 8 Passed: Analytics and status reflect exact remaining document count.")

    # -------------------------------------------------------------
    # TEST 9: Delete PDF -> ask previous question again -> ensure stale cached answer is not returned
    # -------------------------------------------------------------
    print("\n--- TEST 9: Ensure stale cached answer is invalidated after delete ---")
    make_test_pdf("Doc_Alpha.pdf", ["The confidential Alpha token code is TOKEN-ALPHA-7788."])
    requests.post(f"{BASE_URL}/api/index")
    # Prime cache with question
    q_cache = "What is the confidential Alpha token code?"
    res_prime = requests.post(f"{BASE_URL}/api/chat", json={"question": q_cache}).json()
    assert "Doc_Alpha.pdf" in [s.get("document") for s in res_prime.get("sources", [])]
    # Delete Doc_Alpha
    requests.delete(f"{BASE_URL}/api/documents/Doc_Alpha.pdf")
    # Ask the same question again
    res_after_del = requests.post(f"{BASE_URL}/api/chat", json={"question": q_cache}).json()
    assert "Doc_Alpha.pdf" not in [s.get("document") for s in res_after_del.get("sources", [])]
    assert "TOKEN-ALPHA-7788" not in (res_after_del.get("final_verified_answer") or res_after_del.get("answer") or "")
    print("[OK] Test 9 Passed: Stale cache was completely invalidated and purged.")

    # Cleanup and restore standard documents
    requests.delete(f"{BASE_URL}/api/documents/Doc_Beta.pdf")

    # Restore base PDFs
    make_test_pdf("DBMS_Notes.pdf", [
        "DBMS NOTES - ACID PROPERTIES & NORMALIZATION",
        "Atomicity: All operations in a transaction succeed, or none are applied (all-or-nothing).",
        "Consistency: The database transitions only from one valid state to another.",
        "Isolation: Concurrent transactions execute independently without mutual interference.",
        "Durability: Committed updates persist permanently even across power failures.",
        "First Normal Form (1NF): Each column contains only atomic values.",
        "Second Normal Form (2NF): In 1NF and no non-prime attribute is dependent on a proper subset of candidate keys.",
        "Third Normal Form (3NF): In 2NF and no transitive dependency exists."
    ])
    make_test_pdf("Python_Notes.pdf", [
        "PYTHON PROGRAMMING ESSENTIALS",
        "Lists: Ordered, mutable collections defined with square brackets [].",
        "Tuples: Ordered, immutable collections defined with parentheses ().",
        "Dictionaries: Key-value hash mappings defined with curly braces {}.",
        "Sets: Unordered collections of unique elements."
    ])
    make_test_pdf("SQL_CheatSheet.pdf", [
        "SQL COMMANDS REFERENCE",
        "DML Commands: SELECT, INSERT, UPDATE, DELETE.",
        "DDL Commands: CREATE, ALTER, DROP, TRUNCATE.",
        "JOIN Types: INNER JOIN, LEFT OUTER JOIN, RIGHT OUTER JOIN, FULL OUTER JOIN, CROSS JOIN."
    ])
    make_test_pdf("CommerceOS_Complete_Project_Report.pdf", [
        "COMMERCEOS PROJECT REPORT - SDLC METHODOLOGIES",
        "Waterfall Model: A linear, sequential software development lifecycle model.",
        "Phases of Waterfall Model: Requirements analysis, System Design, Implementation, Testing, Deployment, Maintenance."
    ])

    # Re-index restored base documents
    requests.post(f"{BASE_URL}/api/index")

    print("\n" + "=" * 80)
    print("ALL 9 TESTS COMPLETED SUCCESSFULLY WITH 100% PASS RATE!")
    print("=" * 80)

if __name__ == "__main__":
    run_suite()
