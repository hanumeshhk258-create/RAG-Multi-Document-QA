import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_api():
    print("=== STARTING MULTI-DOCUMENT COMPARISON & SYNTHESIS TESTS ===\n")
    
    # 0. Health / Status check
    res = requests.get(f"{BASE_URL}/api/status")
    print(f"Status check: {res.status_code}, total_docs={res.json().get('total_documents')}, chunks={res.json().get('total_chunks')}")
    assert res.status_code == 200
    
    # 1. Test "What is DML?" (Normal Mode)
    print("\n--- TEST 1: 'What is DML?' (Normal mode) ---")
    payload = {"question": "What is DML?"}
    r1 = requests.post(f"{BASE_URL}/api/chat", json=payload).json()
    print(f"Mode: {r1.get('mode')}")
    print(f"Sources count: {len(r1.get('sources', []))}")
    print(f"Answer excerpt:\n{r1.get('answer')[:200]}...")
    assert r1.get('mode') == 'normal'
    assert len(r1.get('sources', [])) > 0

    # 2. Test "What is ACID?" (Normal Mode)
    print("\n--- TEST 2: 'What is ACID?' (Normal mode) ---")
    payload = {"question": "What is ACID?"}
    r2 = requests.post(f"{BASE_URL}/api/chat", json=payload).json()
    print(f"Mode: {r2.get('mode')}")
    print(f"Sources count: {len(r2.get('sources', []))}")
    print(f"Answer excerpt:\n{r2.get('answer')[:200]}...")
    assert r2.get('mode') == 'normal'
    assert len(r2.get('sources', [])) > 0

    # 3. Test "Compare SQL and DBMS" (Comparison Mode)
    print("\n--- TEST 3: 'Compare SQL and DBMS' (Comparison mode) ---")
    payload = {"question": "Compare SQL and DBMS"}
    r3 = requests.post(f"{BASE_URL}/api/chat", json=payload).json()
    print(f"Mode: {r3.get('mode')}")
    print(f"Documents used: {r3.get('documents_used')}")
    print(f"Chunks used: {r3.get('chunks_used')}")
    print(f"Sources count: {len(r3.get('sources', []))}")
    print(f"Answer:\n{r3.get('answer')}")
    assert r3.get('mode') == 'comparison'
    assert r3.get('chunks_used', 0) > 0
    assert "|" in r3.get('answer', '')  # contains markdown table

    # 4. Test "Difference between DDL and DML" (Comparison Mode)
    print("\n--- TEST 4: 'Difference between DDL and DML' (Comparison mode) ---")
    payload = {"question": "Difference between DDL and DML"}
    r4 = requests.post(f"{BASE_URL}/api/chat", json=payload).json()
    print(f"Mode: {r4.get('mode')}")
    print(f"Documents used: {r4.get('documents_used')}")
    print(f"Answer excerpt:\n{r4.get('answer')[:300]}...")
    assert r4.get('mode') == 'comparison'
    assert "|" in r4.get('answer', '')

    # 5. Test "What are the similarities between these documents?" (Cross-document Synthesis)
    print("\n--- TEST 5: 'What are the similarities between these documents?' ---")
    payload = {"question": "What are the similarities between these documents?"}
    r5 = requests.post(f"{BASE_URL}/api/chat", json=payload).json()
    print(f"Mode: {r5.get('mode')}")
    print(f"Documents used: {r5.get('documents_used')}")
    print(f"Answer excerpt:\n{r5.get('answer')[:300]}...")
    assert r5.get('mode') == 'comparison'
    assert len(r5.get('documents_used', [])) >= 2

    # 6. Test "Compare normalization concepts" (Comparison Mode)
    print("\n--- TEST 6: 'Compare normalization concepts' ---")
    payload = {"question": "Compare normalization concepts"}
    r6 = requests.post(f"{BASE_URL}/api/chat", json=payload).json()
    print(f"Mode: {r6.get('mode')}")
    print(f"Answer excerpt:\n{r6.get('answer')[:300]}...")
    assert r6.get('mode') == 'comparison'

    # 7. Test Select only one PDF and ask a comparison question
    print("\n--- TEST 7: Single PDF selected with comparison query ---")
    payload = {
        "question": "Compare SQL and DBMS",
        "selected_documents": ["DBMS_Notes.pdf"]
    }
    r7 = requests.post(f"{BASE_URL}/api/chat", json=payload).json()
    print(f"Mode: {r7.get('mode')}")
    print(f"Answer: {r7.get('answer')}")
    assert "Please select at least two relevant documents" in r7.get('answer', '')

    # 8. Test question whose answer does not exist (Strict Grounding)
    print("\n--- TEST 8: Question whose answer does not exist ---")
    payload = {"question": "Compare quantum teleportation and warp drives in interstellar space"}
    r8 = requests.post(f"{BASE_URL}/api/chat", json=payload).json()
    print(f"Mode: {r8.get('mode')}")
    print(f"Answer: {r8.get('answer')}")
    assert "couldn't find" in r8.get('answer', '').lower() or "not find" in r8.get('answer', '').lower()

    # 9. Test Follow-up question after a comparison
    print("\n--- TEST 9: Follow-up question after a comparison ---")
    history = [
        {"role": "user", "content": "Compare SQL and DBMS"},
        {"role": "assistant", "content": r3.get('answer', ''), "retrieved_context": r3.get('retrieved_context', [])}
    ]
    payload = {
        "question": "What is the key difference between them?",
        "history": history
    }
    r9 = requests.post(f"{BASE_URL}/api/chat", json=payload).json()
    print(f"Mode: {r9.get('mode')}")
    print(f"Is followup: {r9.get('is_followup')}")
    print(f"Answer excerpt:\n{r9.get('answer')[:300]}...")

    # 10. Test View PDF endpoint
    print("\n--- TEST 10: View PDF endpoint test ---")
    r10 = requests.get(f"{BASE_URL}/view_pdf/DBMS_Notes.pdf")
    print(f"View PDF status: {r10.status_code}, content-type: {r10.headers.get('content-type')}")
    assert r10.status_code == 200
    assert "application/pdf" in r10.headers.get('content-type')

    # 11. Test View Excerpt and page metadata integrity
    print("\n--- TEST 11: View Excerpt and page metadata check ---")
    sources = r3.get('sources', [])
    assert len(sources) > 0
    top_src = sources[0]
    print(f"Top source document: {top_src.get('document')}, page: {top_src.get('page')}, score: {top_src.get('score')}")
    assert top_src.get('document') is not None
    assert top_src.get('page') is not None
    assert top_src.get('pdf_url') is not None

    print("\n=== ALL 11 MULTI-DOCUMENT COMPARISON TESTS PASSED PERFECTLY! ===")

if __name__ == "__main__":
    test_api()
