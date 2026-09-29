import json
import time
import urllib.request
import os

BASE_URL = "http://127.0.0.1:5000"

def chat(question, history=None, selected_docs=None):
    payload = {
        "question": question,
        "conversation": history or [],
        "history": history or []
    }
    if selected_docs is not None:
        payload["selected_documents"] = selected_docs

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode())
            res["client_elapsed"] = round(time.time() - t0, 2)
            return res
    except Exception as e:
        return {"success": False, "error": str(e), "client_elapsed": round(time.time() - t0, 2)}


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING QUERY DECOMPOSITION / COMPLEX QUERY RETRIEVAL VERIFICATION SUITE")
    print("="*70 + "\n")

    # TEST 1: Simple query - Primary Key
    print("--- TEST 1: Simple Question ('What is a primary key?') ---")
    t1 = chat("What is a primary key?")
    print(f"Success: {t1.get('success')}")
    print(f"Is Decomposed: {t1.get('is_decomposed', False)}")
    print(f"Query Type: {t1.get('query_type')}")
    print(f"Mode: {t1.get('mode')}")
    print(f"Answer Preview: {t1.get('answer', '')[:120]}...")
    print(f"Sources Count: {len(t1.get('sources', []))}")
    assert t1.get("is_decomposed", False) is False, "TEST 1 Failed: Simple question should NOT be decomposed!"
    print(">> TEST 1 PASSED: Simple retrieval without decomposition.\n")

    # TEST 2: Normal query - Security controls in CommerceOS
    print("--- TEST 2: Normal Retrieval ('Explain the security controls in CommerceOS.') ---")
    t2 = chat("Explain the security controls in CommerceOS.")
    print(f"Success: {t2.get('success')}")
    print(f"Is Decomposed: {t2.get('is_decomposed', False)}")
    print(f"Mode: {t2.get('mode')}")
    print(f"Answer Preview: {t2.get('answer', '')[:120]}...")
    print(f"Sources Count: {len(t2.get('sources', []))}")
    assert t2.get("is_decomposed", False) is False, "TEST 2 Failed: Normal single-topic query should NOT be decomposed!"
    print(">> TEST 2 PASSED: Normal retrieval without decomposition.\n")

    # TEST 3: Complex Multi-Document Comparative Query
    print("--- TEST 3: Complex Query ('Compare the security and trust controls in CommerceOS with the security concepts in the assignment PDF and explain the similarities and differences.') ---")
    q3 = "Compare the security and trust controls in CommerceOS with the security concepts in the assignment PDF and explain the similarities and differences."
    t3 = chat(q3)
    print(f"Success: {t3.get('success')}")
    print(f"Is Decomposed: {t3.get('is_decomposed', False)}")
    print(f"Sub-queries ({len(t3.get('sub_queries', []))}):")
    for i, sq in enumerate(t3.get("sub_queries", []), 1):
        print(f"  [{i}] {sq}")
    print(f"Documents used: {t3.get('documents_used', [])}")
    print(f"Chunks used: {t3.get('chunks_used')}")
    print(f"Sources Count: {len(t3.get('sources', []))}")
    print(f"Answer Preview:\n{t3.get('answer', '')[:250]}...\n")
    assert t3.get("is_decomposed", False) is True, "TEST 3 Failed: Complex comparative question MUST be decomposed!"
    assert 2 <= len(t3.get("sub_queries", [])) <= 5, "TEST 3 Failed: Sub-queries must be between 2 and 5!"
    print(">> TEST 3 PASSED: Query decomposition activates with 2-5 sub-queries.\n")

    # TEST 4: Complex Multi-Document Query across SQL & DBMS
    print("--- TEST 4: Complex Multi-Document Query ('Compare SQL querying commands with the concepts explained in the DBMS notes.') ---")
    q4 = "Compare SQL querying commands with the concepts explained in the DBMS notes."
    t4 = chat(q4)
    print(f"Success: {t4.get('success')}")
    print(f"Is Decomposed: {t4.get('is_decomposed', False)}")
    print(f"Sub-queries ({len(t4.get('sub_queries', []))}):")
    for i, sq in enumerate(t4.get("sub_queries", []), 1):
        print(f"  [{i}] {sq}")
    print(f"Documents used: {t4.get('documents_used', [])}")
    print(f"Sources Count: {len(t4.get('sources', []))}")
    print(f"Answer Preview:\n{t4.get('answer', '')[:250]}...\n")
    assert t4.get("is_decomposed", False) is True or t4.get("is_comparison", False) is True, "TEST 4 Failed: Multi-document query must be decomposed or comparative!"
    print(">> TEST 4 PASSED: Multiple sub-queries and multi-document retrieval.\n")

    # TEST 5: Conversational Follow-up
    print("--- TEST 5: Conversational Follow-up ('What about their differences?') ---")
    history5 = [
        {"role": "user", "content": q4},
        {"role": "assistant", "content": t4.get("answer", "")}
    ]
    t5 = chat("What about their differences?", history=history5)
    print(f"Success: {t5.get('success')}")
    print(f"Rewritten query: {t5.get('rewritten_query')}")
    print(f"Is Follow-up: {t5.get('is_followup')}")
    print(f"Context Topic: {t5.get('context_topic')}")
    print(f"Is Decomposed: {t5.get('is_decomposed', False)}")
    print(f"Answer Preview:\n{t5.get('answer', '')[:200]}...\n")
    assert t5.get("is_followup", False) is True or "sql" in str(t5.get("rewritten_query", "")).lower() or "dbms" in str(t5.get("rewritten_query", "")).lower(), "TEST 5 Failed: Follow-up question must resolve context!"
    print(">> TEST 5 PASSED: Conversation memory resolves 'their' using previous context.\n")

    print("="*70)
    print("ALL 5 QUERY DECOMPOSITION & COMPLEX RETRIEVAL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
