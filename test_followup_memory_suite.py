import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_api():
    print("==================================================")
    print("STARTING COMPREHENSIVE FOLLOW-UP MEMORY TEST SUITE")
    print("==================================================")

    # 1. Health check
    res = requests.get(f"{BASE_URL}/api/status")
    print("System Status:", res.status_code, res.json())
    assert res.status_code == 200

    # Ensure documents are indexed
    docs_res = requests.get(f"{BASE_URL}/api/documents")
    docs_data = docs_res.json()
    available_docs = [d["filename"] for d in docs_data.get("documents", [])]
    print(f"Available documents ({len(available_docs)}): {available_docs}")

    # =========================================================================
    # TEST 1: What is DML? -> Give me an example.
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 1: 'What is DML?' -> 'Give me an example.'")
    # Reset chat
    requests.post(f"{BASE_URL}/api/new_chat")
    
    # Q1
    q1 = "What is DML?"
    r1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q1,
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    print(f"Q1: {q1} -> mode: {r1.get('mode')}, is_followup: {r1.get('is_followup')}")
    assert r1.get("mode") in ["normal", "general"], f"Expected normal mode for standalone Q1, got {r1.get('mode')}"

    # Follow-up
    q2 = "Give me an example."
    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1.get("answer", "")}
    ]
    r2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q2,
        "selected_documents": available_docs,
        "conversation": history
    }).json()
    print(f"Q2: {q2}")
    print(f"  Mode: {r2.get('mode')}, is_followup: {r2.get('is_followup')}, context_topic: {r2.get('context_topic')}")
    print(f"  Rewritten: {r2.get('rewritten_query')}")
    print(f"  Answer snippet: {r2.get('answer', '')[:120]}...")
    assert r2.get("is_followup") == True or r2.get("context_used") == True, "TEST 1 Failed: Expected follow-up to be detected"
    assert "dml" in r2.get("rewritten_query", "").lower() or "manipulation" in r2.get("rewritten_query", "").lower(), "TEST 1 Failed: Rewritten query should mention DML"
    print(">>> TEST 1 PASSED: Assistant understands 'example' refers to DML.")

    # =========================================================================
    # TEST 2: What is normalization? -> What are its advantages?
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 2: 'What is normalization?' -> 'What are its advantages?'")
    requests.post(f"{BASE_URL}/api/new_chat")
    
    q1 = "What is normalization?"
    r1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q1,
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    
    q2 = "What are its advantages?"
    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1.get("answer", "")}
    ]
    r2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q2,
        "selected_documents": available_docs,
        "conversation": history
    }).json()
    print(f"Q2: {q2}")
    print(f"  Mode: {r2.get('mode')}, is_followup: {r2.get('is_followup')}, context_topic: {r2.get('context_topic')}")
    print(f"  Rewritten: {r2.get('rewritten_query')}")
    assert r2.get("is_followup") == True or r2.get("context_used") == True, "TEST 2 Failed: Expected follow-up"
    assert "normalization" in r2.get("rewritten_query", "").lower(), "TEST 2 Failed: Rewritten query should mention normalization"
    print(">>> TEST 2 PASSED: Assistant understands 'its' = normalization.")

    # =========================================================================
    # TEST 3: Compare SQL and DBMS. -> What are their similarities?
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 3: 'Compare SQL and DBMS.' -> 'What are their similarities?'")
    requests.post(f"{BASE_URL}/api/new_chat")
    
    q1 = "Compare SQL and DBMS."
    r1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q1,
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    
    q2 = "What are their similarities?"
    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1.get("answer", "")}
    ]
    r2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q2,
        "selected_documents": available_docs,
        "conversation": history
    }).json()
    print(f"Q2: {q2}")
    print(f"  Mode: {r2.get('mode')}, is_followup: {r2.get('is_followup')}, context_topic: {r2.get('context_topic')}")
    print(f"  Rewritten: {r2.get('rewritten_query')}")
    assert r2.get("is_followup") == True or r2.get("context_used") == True, "TEST 3 Failed: Expected follow-up"
    rewritten_lower = r2.get("rewritten_query", "").lower()
    assert "sql" in rewritten_lower and "dbms" in rewritten_lower, "TEST 3 Failed: Rewritten query should mention both SQL and DBMS"
    print(">>> TEST 3 PASSED: Assistant understands comparison follow-up with both concepts.")

    # =========================================================================
    # TEST 4: What is DDL? -> Explain it simply.
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 4: 'What is DDL?' -> 'Explain it simply.'")
    requests.post(f"{BASE_URL}/api/new_chat")
    
    q1 = "What is DDL?"
    r1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q1,
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    
    q2 = "Explain it simply."
    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1.get("answer", "")}
    ]
    r2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q2,
        "selected_documents": available_docs,
        "conversation": history
    }).json()
    print(f"Q2: {q2}")
    print(f"  Mode: {r2.get('mode')}, is_followup: {r2.get('is_followup')}, context_topic: {r2.get('context_topic')}")
    print(f"  Rewritten: {r2.get('rewritten_query')}")
    assert r2.get("is_followup") == True or r2.get("context_used") == True, "TEST 4 Failed: Expected follow-up"
    assert "ddl" in r2.get("rewritten_query", "").lower() or "definition" in r2.get("rewritten_query", "").lower(), "TEST 4 Failed: Rewritten query should mention DDL"
    print(">>> TEST 4 PASSED: Assistant understands simple explanation of DDL.")

    # =========================================================================
    # TEST 5: What is DML? -> What about DDL?
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 5: 'What is DML?' -> 'What about DDL?'")
    requests.post(f"{BASE_URL}/api/new_chat")
    
    q1 = "What is DML?"
    r1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q1,
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    
    q2 = "What about DDL?"
    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1.get("answer", "")}
    ]
    r2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q2,
        "selected_documents": available_docs,
        "conversation": history
    }).json()
    print(f"Q2: {q2}")
    print(f"  Mode: {r2.get('mode')}, is_followup: {r2.get('is_followup')}, context_topic: {r2.get('context_topic')}")
    print(f"  Rewritten: {r2.get('rewritten_query')}")
    assert "ddl" in r2.get("rewritten_query", "").lower(), "TEST 5 Failed: Rewritten query should focus on DDL"
    print(">>> TEST 5 PASSED: Assistant shifts focus to DDL.")

    # =========================================================================
    # TEST 6: What is ACID? -> Explain atomicity.
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 6: 'What is ACID?' -> 'Explain atomicity.'")
    requests.post(f"{BASE_URL}/api/new_chat")
    
    q1 = "What is ACID?"
    r1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q1,
        "selected_documents": available_docs,
        "conversation": []
    }).json()
    
    q2 = "Explain atomicity."
    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1.get("answer", "")}
    ]
    r2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q2,
        "selected_documents": available_docs,
        "conversation": history
    }).json()
    print(f"Q2: {q2}")
    print(f"  Mode: {r2.get('mode')}, is_followup: {r2.get('is_followup')}, context_topic: {r2.get('context_topic')}")
    print(f"  Rewritten: {r2.get('rewritten_query')}")
    assert "atomicity" in r2.get("rewritten_query", "").lower(), "TEST 6 Failed: Rewritten query should explain atomicity"
    print(">>> TEST 6 PASSED: Assistant explains atomicity in context of ACID.")

    # =========================================================================
    # TEST 7: Start New Chat -> What about its advantages?
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 7: New Chat -> 'What about its advantages?'")
    requests.post(f"{BASE_URL}/api/new_chat")
    
    q1 = "What about its advantages?"
    r1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q1,
        "selected_documents": available_docs,
        "conversation": []  # Empty history after New Chat
    }).json()
    print(f"Q1: {q1}")
    print(f"  Answer: {r1.get('answer', '')}")
    print(f"  Context used: {r1.get('context_used')}, mode: {r1.get('mode')}")
    assert r1.get("context_used") == False or r1.get("is_followup") == False, "TEST 7 Failed: Context should NOT be used after New Chat"
    assert r1.get("mode") != "followup", "TEST 7 Failed: Mode should not be followup"
    print(">>> TEST 7 PASSED: Does not use previous context after New Chat.")

    # =========================================================================
    # TEST 8: Select only one document -> Ask follow-up -> Restricted
    # =========================================================================
    print("\n--------------------------------------------------")
    print("TEST 8: Select single document ('DBMS_Notes.pdf') -> Ask follow-up")
    requests.post(f"{BASE_URL}/api/new_chat")
    
    single_doc = ["DBMS_Notes.pdf"]
    q1 = "What is normalization?"
    r1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q1,
        "selected_documents": single_doc,
        "conversation": []
    }).json()
    
    q2 = "Give me an example."
    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1.get("answer", "")}
    ]
    r2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q2,
        "selected_documents": single_doc,
        "conversation": history
    }).json()
    print(f"Q2: {q2}")
    sources = r2.get("sources", [])
    print(f"  Sources returned ({len(sources)}): {[s.get('document') for s in sources]}")
    for s in sources:
        assert s.get("document") == "DBMS_Notes.pdf", f"TEST 8 Failed: Expected only DBMS_Notes.pdf, got {s.get('document')}"
    print(">>> TEST 8 PASSED: Retrieval strictly restricted to selected document.")

    print("\n==================================================")
    print("ALL 8 FOLLOW-UP CONVERSATIONAL MEMORY TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test_api()
