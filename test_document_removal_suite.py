import os
import time
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def run_test():
    print("=" * 80)
    print("STARTING DOCUMENT REMOVAL & RETRIEVAL PURGE TEST SUITE")
    print("=" * 80)

    # 1. Check initial status
    status_res = requests.get(f"{BASE_URL}/api/status", timeout=10).json()
    init_docs = [d["filename"] for d in status_res.get("documents", [])]
    init_total = status_res.get("total_documents", 0)
    print(f"[STATUS] Initial total documents: {init_total} ({init_docs})")
    assert init_total > 0, "No documents initially indexed"

    target_doc = "SQL_CheatSheet.pdf"
    if target_doc not in init_docs:
        target_doc = init_docs[0]

    print(f"\n--- TEST 1: Query before removal of '{target_doc}' ---")
    chat_res_1 = requests.post(
        f"{BASE_URL}/api/chat",
        json={"question": "What is SQL and relational database queries?", "selected_documents": init_docs},
        timeout=25
    ).json()
    sources_1 = chat_res_1.get("sources", [])
    src_docs_1 = [s.get("document") for s in sources_1]
    print(f"  Response received: {len(chat_res_1.get('answer', ''))} chars")
    print(f"  Retrieved sources: {set(src_docs_1)}")

    # 2. Call /api/remove-document
    print(f"\n--- TEST 2: Removing '{target_doc}' via API ---")
    t0 = time.perf_counter()
    del_res = requests.post(
        f"{BASE_URL}/api/remove-document",
        json={"filename": target_doc},
        timeout=30
    )
    t_del = time.perf_counter() - t0
    print(f"  Delete API response status: {del_res.status_code} in {t_del:.2f}s")
    del_data = del_res.json()
    assert del_data.get("success") is True, f"Delete failed: {del_data}"
    assert del_data.get("deleted_document") == target_doc, "Mismatch in deleted_document"
    assert del_data.get("remaining_documents") == init_total - 1, "Mismatch in remaining_documents"
    print(f"  [OK] Success: deleted_document={del_data.get('deleted_document')}, remaining={del_data.get('remaining_documents')}")

    # 3. Check physical file deletion & metadata purge
    print(f"\n--- TEST 3: Verifying disk & metadata purge ---")
    doc_path = os.path.join("documents", target_doc)
    assert not os.path.exists(doc_path), f"File still exists on disk at {doc_path}!"
    print(f"  [OK] File '{target_doc}' confirmed deleted from disk.")

    with open(os.path.join("vectorstore", "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert target_doc not in meta.get("indexed_documents", []), f"'{target_doc}' still in metadata.json!"
    print(f"  [OK] Metadata purged: indexed_documents = {meta.get('indexed_documents')}")

    # 4. Query after removal - verify target_doc is NEVER returned
    print(f"\n--- TEST 4: Querying after removal - zero trace of '{target_doc}' ---")
    status_res_2 = requests.get(f"{BASE_URL}/api/status", timeout=10).json()
    curr_docs = [d["filename"] for d in status_res_2.get("documents", [])]

    chat_res_2 = requests.post(
        f"{BASE_URL}/api/chat",
        json={"question": "What is SQL and relational database queries?", "selected_documents": curr_docs},
        timeout=25
    ).json()
    sources_2 = chat_res_2.get("sources", [])
    src_docs_2 = [s.get("document") for s in sources_2]
    print(f"  Sources returned after deletion: {set(src_docs_2)}")
    assert target_doc not in src_docs_2, f"CRITICAL: Deleted document '{target_doc}' was still retrieved in sources!"
    print(f"  [OK] Confirmed '{target_doc}' is NOT present in any retrieved chunk or citation.")

    # 5. Test removing a non-existent document
    print(f"\n--- TEST 5: Error handling for non-existent document ---")
    err_res = requests.post(
        f"{BASE_URL}/api/remove-document",
        json={"filename": "Fake_Doc_9999.pdf"},
        timeout=10
    )
    print(f"  Non-existent delete status: {err_res.status_code}")
    assert err_res.status_code == 404, "Expected 404 for non-existent document"
    print("  [OK] Handled gracefully with 404 status.")

    # 6. Restore deleted file to verify re-indexing flow
    if target_doc == "SQL_CheatSheet.pdf":
        print("\n--- TEST 6: Restoring sample PDF to verify re-indexing flow ---")
        from create_sample_pdfs import create_simple_pdf
        sql_p1 = (
            "SQL Overview and Relational Database Queries\n"
            "Structured Query Language (SQL) is the standard declarative language for interacting with relational databases.\n\n"
            "SQL Sub-Languages:\n"
            "1. DDL (Data Definition Language): CREATE, ALTER, DROP, TRUNCATE.\n"
            "2. DML (Data Manipulation Language): SELECT, INSERT, UPDATE, DELETE.\n"
            "3. DCL (Data Control Language): GRANT, REVOKE permissions.\n"
            "4. TCL (Transaction Control Language): COMMIT, ROLLBACK, SAVEPOINT."
        )
        sql_p2 = (
            "SQL Joins and Query Optimization\n"
            "Types of Joins:\n"
            "- INNER JOIN: Returns records that have matching values in both tables.\n"
            "- LEFT (OUTER) JOIN: Returns all records from left table and matched from right.\n"
            "- RIGHT (OUTER) JOIN: Returns all records from right table and matched from left.\n"
            "- FULL (OUTER) JOIN: Returns all records when there is a match in either left or right table.\n\n"
            "Query Optimization Best Practices:\n"
            "- Use appropriate indexes on frequently filtered columns (WHERE clause).\n"
            "- Avoid SELECT *; explicitly name needed columns.\n"
            "- Use EXPLAIN query plan to inspect execution bottlenecks."
        )
        create_simple_pdf("documents/SQL_CheatSheet.pdf", [sql_p1, sql_p2])
        idx_res = requests.post(f"{BASE_URL}/api/index", timeout=30).json()
        print(f"  Re-index result: indexed {idx_res.get('documents')} docs, {idx_res.get('chunks')} chunks.")
        assert idx_res.get("success") is True, "Re-indexing failed"
        print("  [OK] Re-indexing completed successfully.")

    print("\n" + "=" * 80)
    print("ALL DOCUMENT REMOVAL & RETRIEVAL PURGE TESTS PASSED SUCCESSFULLY! (100%)")
    print("=" * 80)

if __name__ == "__main__":
    run_test()
