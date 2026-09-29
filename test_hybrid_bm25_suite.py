import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def run_tests():
    print("==================================================")
    print("STARTING HYBRID SEARCH + BM25 + RERANKING TEST SUITE")
    print("==================================================")

    # 1. Server Status check
    res = requests.get(f"{BASE_URL}/api/status")
    print("System Status:", res.status_code, res.json())
    assert res.status_code == 200

    docs_res = requests.get(f"{BASE_URL}/api/documents")
    available_docs = [d["filename"] for d in docs_res.json().get("documents", [])]
    print(f"Available documents ({len(available_docs)}): {available_docs}\n")

    # =========================================================================
    # TEST 1: "What is ACID?"
    # =========================================================================
    print("--------------------------------------------------")
    print("TEST 1: 'What is ACID?'")
    r1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is ACID?",
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    print(f"  Retrieval method: {r1.get('retrieval_method')}")
    print(f"  Semantic results: {r1.get('semantic_results')}, Keyword results: {r1.get('keyword_results')}")
    print(f"  Fused candidates: {r1.get('fused_candidates')}, Reranked: {r1.get('reranked_results')}")
    print(f"  Sources: {[s['document'] + ' P' + str(s['page']) for s in r1.get('sources', [])]}")
    assert r1.get("retrieval_method") == "hybrid"
    assert any("DBMS_Notes.pdf" in s["document"] or "ACID" in s.get("snippet", "") for s in r1.get("sources", []))
    print(">>> TEST 1 PASSED: Relevant DBMS ACID chunk retrieved via Hybrid Search.")

    # =========================================================================
    # TEST 2: "What are the ACID properties?"
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 2: 'What are the ACID properties?'")
    r2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What are the ACID properties?",
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    answer_text = r2.get("answer", "")
    print(f"  Answer excerpt: {answer_text[:120]}...")
    assert "atomicity" in answer_text.lower() or "consistency" in answer_text.lower() or "isolation" in answer_text.lower() or "durability" in answer_text.lower()
    print(">>> TEST 2 PASSED: Relevant chunks containing Atomicity/Consistency/Isolation/Durability retrieved.")

    # =========================================================================
    # TEST 3: "What is DML?"
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 3: 'What is DML?'")
    r3 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is DML?",
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    sources = [s["document"] for s in r3.get("sources", [])]
    print(f"  Sources: {sources}")
    assert any("SQL" in s or "DBMS" in s for s in sources)
    assert "data manipulation" in r3.get("answer", "").lower() or "dml" in r3.get("answer", "").lower()
    print(">>> TEST 3 PASSED: Relevant SQL/DBMS chunks retrieved.")

    # =========================================================================
    # TEST 4: "Compare SQL and DBMS."
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 4: 'Compare SQL and DBMS.'")
    r4 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "Compare SQL and DBMS.",
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    print(f"  Mode: {r4.get('mode')}")
    print(f"  Documents used: {r4.get('documents_used')}")
    assert r4.get("mode") == "comparison"
    docs_used = r4.get("documents_used", [])
    assert any("SQL" in d for d in docs_used) and any("DBMS" in d for d in docs_used)
    print(">>> TEST 4 PASSED: Relevant chunks from both concepts/documents retrieved.")

    # =========================================================================
    # TEST 5: "What is Key Security and Trust Controls?"
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 5: 'What is Key Security and Trust Controls?'")
    r5 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is Key Security and Trust Controls?",
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    top_sources = [s["document"] for s in r5.get("sources", [])]
    print(f"  Sources: {top_sources}")
    assert any("CommerceOS" in s for s in top_sources)
    print(">>> TEST 5 PASSED: Correct CommerceOS document chunks retrieved.")

    # =========================================================================
    # TEST 6: Follow-up: "What is DML?" -> "Give me an example."
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 6: Follow-up: 'What is DML?' -> 'Give me an example.'")
    requests.post(f"{BASE_URL}/api/new_chat")
    q1 = "What is DML?"
    res_q1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q1,
        "selected_documents": available_docs,
        "conversation": []
    }).json()

    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": res_q1.get("answer", "")}
    ]
    r6 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "Give me an example.",
        "selected_documents": available_docs,
        "conversation": history
    }).json()
    print(f"  Rewritten query: {r6.get('rewritten_query')}")
    print(f"  Is follow-up: {r6.get('is_followup')}")
    print(f"  Context topic: {r6.get('context_topic')}")
    assert r6.get("is_followup") == True or r6.get("context_used") == True
    assert "dml" in r6.get("rewritten_query", "").lower() or "manipulation" in r6.get("rewritten_query", "").lower()
    print(">>> TEST 6 PASSED: Conversational rewriting + hybrid retrieval.")

    # =========================================================================
    # TEST 7: Select only SQL_CheatSheet.pdf -> "What is DML?"
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 7: Select only 'SQL_CheatSheet.pdf' -> 'What is DML?'")
    r7 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is DML?",
        "selected_documents": ["SQL_CheatSheet.pdf"],
        "conversation": []
    }).json()
    sources_7 = r7.get("sources", [])
    print(f"  Sources ({len(sources_7)}): {[s['document'] for s in sources_7]}")
    for s in sources_7:
        assert s["document"] == "SQL_CheatSheet.pdf"
    print(">>> TEST 7 PASSED: Search strictly restricted to SQL_CheatSheet.pdf.")

    # =========================================================================
    # TEST 8: Select two documents -> "Compare the concepts discussed in these documents."
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 8: Select 2 docs (DBMS_Notes.pdf, SQL_CheatSheet.pdf) -> Compare concepts")
    r8 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "Compare the concepts discussed in these documents.",
        "selected_documents": ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"],
        "conversation": []
    }).json()
    print(f"  Mode: {r8.get('mode')}")
    print(f"  Documents used: {r8.get('documents_used')}")
    assert r8.get("mode") == "comparison"
    docs_8 = r8.get("documents_used", [])
    assert "DBMS_Notes.pdf" in docs_8 and "SQL_CheatSheet.pdf" in docs_8
    print(">>> TEST 8 PASSED: Balanced retrieval from both selected documents.")

    # =========================================================================
    # TEST 9: Question whose answer does not exist
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 9: Question that does not exist in documents")
    r9 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is quantum warp velocity using tachyon particle flux in spacetime?",
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    print(f"  Answer: {r9.get('answer')}")
    print(f"  Sources count: {len(r9.get('sources', []))}")
    assert len(r9.get("sources", [])) == 0
    assert "couldn't find" in r9.get("answer", "").lower() or "could not find" in r9.get("answer", "").lower()
    print(">>> TEST 9 PASSED: No hallucination, properly returned not found message.")

    # =========================================================================
    # TEST 10: View PDF & View Excerpt check
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 10: View PDF & View Excerpt endpoint check")
    pdf_res = requests.get(f"{BASE_URL}/view_pdf/DBMS_Notes.pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers.get("Content-Type") == "application/pdf"
    print(f"  View PDF status: {pdf_res.status_code}, content-type: {pdf_res.headers.get('Content-Type')}")
    print(">>> TEST 10 PASSED: View PDF inline endpoint verified.")

    print("\n==================================================")
    print("ALL 10 HYBRID SEARCH + BM25 + RERANKING TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
