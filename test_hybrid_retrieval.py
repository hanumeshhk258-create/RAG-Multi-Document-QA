import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def wait_for_server():
    print("Waiting for server to be ready...")
    for _ in range(30):
        try:
            res = requests.get(f"{BASE_URL}/api/status", timeout=2)
            if res.status_code == 200:
                print("Server is ready!")
                return True
        except Exception:
            time.sleep(1)
    raise RuntimeError("Server failed to start")

def test_hybrid_pipeline():
    wait_for_server()
    
    # 1. Status & loaded documents
    status_res = requests.get(f"{BASE_URL}/api/status").json()
    docs = [d["filename"] for d in status_res.get("documents", [])]
    print(f"Available indexed documents: {docs}")
    assert len(docs) > 0, "No indexed documents found!"

    # 2. Reset conversation
    requests.post(f"{BASE_URL}/api/new_chat")

    # 3. Test exact technical terms
    exact_terms = [
        "What is ACID?",
        "What is DML?",
        "What is SQL?",
        "What is JOIN?",
        "What is a primary key?"
    ]

    print("\n=======================================================")
    print("1. TESTING EXACT TECHNICAL TERMS (HYBRID KEYWORD + SEMANTIC)")
    print("=======================================================")
    for q in exact_terms:
        payload = {"question": q, "selected_documents": docs, "conversation": []}
        res = requests.post(f"{BASE_URL}/api/chat", json=payload)
        assert res.status_code == 200, f"Failed chat for query: {q}"
        data = res.json()
        
        answer = data.get("answer", "")
        eval_info = data.get("evaluation", {})
        stats = data.get("retrieval_stats") or eval_info.get("retrieval_stats", {})
        timing = data.get("timing_breakdown", {})
        sources = data.get("sources", [])

        print(f"\n[QUERY]: {q}")
        print(f"  * Answer: {answer[:90]}...")
        print(f"  * Sources count: {len(sources)}")
        print(f"  * Groundedness: {eval_info.get('groundedness_score')}% ({eval_info.get('groundedness_label')})")
        print(f"  * Retrieval Method: {eval_info.get('retrieval_method')}")
        print(f"  * Hybrid Stats: Semantic={stats.get('semantic_candidates')}, Keyword={stats.get('keyword_candidates')}, Deduped={stats.get('deduped_candidates')}, Final={stats.get('final_chunks')}")
        print(f"  * Timing: Retrieval={timing.get('retrieval_sec')}s | Generation={timing.get('generation_sec')}s | Total={timing.get('total_sec')}s")

        assert eval_info.get("retrieval_method") == "Hybrid Search"
        assert stats.get("semantic_candidates", 0) > 0 or stats.get("keyword_candidates", 0) > 0
        assert len(sources) > 0

    # 4. Test natural language queries
    nl_queries = [
        "Explain database transactions.",
        "Explain ACID properties in simple words.",
        "How does JOIN work?"
    ]

    print("\n=======================================================")
    print("2. TESTING NATURAL LANGUAGE QUERIES")
    print("=======================================================")
    for q in nl_queries:
        payload = {"question": q, "selected_documents": docs, "conversation": []}
        res = requests.post(f"{BASE_URL}/api/chat", json=payload)
        assert res.status_code == 200
        data = res.json()
        eval_info = data.get("evaluation", {})
        print(f"\n[QUERY]: {q}")
        print(f"  * Groundedness: {eval_info.get('groundedness_score')}% ({eval_info.get('groundedness_label')})")
        print(f"  * Confidence: {eval_info.get('confidence')}")
        print(f"  * Sources: {[s['document'] for s in data.get('sources', [])]}")

    # 5. Test multi-turn follow-up queries
    print("\n=======================================================")
    print("3. TESTING MULTI-TURN CONVERSATION FOLLOW-UP QUERIES")
    print("=======================================================")
    conv = []
    
    # Turn 1
    q1 = "What is ACID?"
    res1 = requests.post(f"{BASE_URL}/api/chat", json={"question": q1, "selected_documents": docs, "conversation": conv}).json()
    conv.append({"role": "user", "content": q1})
    conv.append({"role": "assistant", "content": res1["answer"]})
    print(f"\nTurn 1: {q1} -> Sources: {[s['document'] for s in res1['sources']]}")

    # Turn 2
    q2 = "Give an example."
    res2 = requests.post(f"{BASE_URL}/api/chat", json={"question": q2, "selected_documents": docs, "conversation": conv}).json()
    conv.append({"role": "user", "content": q2})
    conv.append({"role": "assistant", "content": res2["answer"]})
    print(f"Turn 2: {q2} -> Rewritten: {res2.get('rewritten_query')} -> Groundedness: {res2.get('evaluation', {}).get('groundedness_score')}%")

    # Turn 3
    q3 = "What about the last property?"
    res3 = requests.post(f"{BASE_URL}/api/chat", json={"question": q3, "selected_documents": docs, "conversation": conv}).json()
    print(f"Turn 3: {q3} -> Rewritten: {res3.get('rewritten_query')} -> Answer: {res3['answer'][:80]}...")
    assert "durability" in res3["answer"].lower() or "durability" in res3.get("rewritten_query", "").lower()

    # 6. Test missing information (out-of-domain)
    print("\n=======================================================")
    print("4. TESTING MISSING / OUT-OF-DOMAIN QUERY")
    print("=======================================================")
    q_out = "What is quantum computing with superconducting qubits?"
    res_out = requests.post(f"{BASE_URL}/api/chat", json={"question": q_out, "selected_documents": docs, "conversation": []}).json()
    print(f"Out-of-domain query: {q_out}")
    print(f"  * Answer: {res_out['answer']}")
    print(f"  * Groundedness: {res_out.get('evaluation', {}).get('groundedness_score')}% ({res_out.get('evaluation', {}).get('groundedness_label')})")
    assert "couldn't find" in res_out["answer"].lower() or "could not find" in res_out["answer"].lower()
    assert res_out.get("evaluation", {}).get("groundedness_score") == 0

    # 7. Test strict document filtering
    print("\n=======================================================")
    print("5. TESTING STRICT DOCUMENT SELECTION / FILTERING")
    print("=======================================================")
    filtered_doc = "DBMS_Notes.pdf"
    res_filtered = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is SQL?",
        "selected_documents": [filtered_doc],
        "conversation": []
    }).json()
    print(f"Query: 'What is SQL?' with only [{filtered_doc}] selected")
    sources_used = [s["document"] for s in res_filtered.get("sources", [])]
    print(f"  * Sources returned: {sources_used}")
    for s in sources_used:
        assert s == filtered_doc, f"Unselected document returned! {s} != {filtered_doc}"

    print("\n=======================================================")
    print("ALL HYBRID RETRIEVAL & RERANKING TESTS PASSED SUCCESSFULLY!")
    print("=======================================================\n")

if __name__ == "__main__":
    test_hybrid_pipeline()
