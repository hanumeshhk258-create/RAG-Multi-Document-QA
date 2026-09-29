import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def run_tests():
    # 1. Check status
    time.sleep(1)
    status_res = requests.get(f"{BASE_URL}/api/status")
    print(f"Status check: {status_res.status_code}")
    data = status_res.json()
    docs = data.get("documents", [])
    print(f"Loaded documents ({len(docs)}): {docs}")
    selected_docs = docs if docs else ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"]

    # 2. Reset chat session
    requests.post(f"{BASE_URL}/api/new_chat")

    test_queries = [
        ("What are ACID properties?", False),
        ("What is DML?", False),
        ("Explain JOIN.", False),
        ("Give an example of normalization.", False),
        ("What is the third property?", True),
        ("What is quantum teleportation using tachyons and dark energy flux in database indexing?", False)
    ]

    print("\n=================== TESTING RAG EVALUATION & QUALITY DASHBOARD ===================")
    conversation_history = []
    
    for q, is_followup in test_queries:
        print(f"\n[QUERY]: '{q}' (follow-up: {is_followup})")
        payload = {
            "question": q,
            "selected_documents": [d["filename"] if isinstance(d, dict) else d for d in selected_docs],
            "conversation": conversation_history
        }
        res = requests.post(f"{BASE_URL}/api/chat", json=payload)
        assert res.status_code == 200, f"Failed chat request ({res.status_code}): {res.text}"
        res_data = res.json()
        
        answer = res_data.get("answer", "")
        eval_data = res_data.get("evaluation")
        sources = res_data.get("sources", [])
        resp_time = res_data.get("response_time")

        # Update history
        conversation_history.append({"role": "user", "content": q})
        conversation_history.append({"role": "assistant", "content": answer})

        print(f"-> Response Time: {resp_time}s")
        print(f"-> Answer snippet: {answer[:120]}...")
        print(f"-> Sources count: {len(sources)}")
        
        assert eval_data is not None, "Evaluation data missing in response!"
        print(f"-> EVALUATION METRICS:")
        print(f"   * Groundedness Score: {eval_data.get('groundedness_score')}%")
        print(f"   * Groundedness Label: {eval_data.get('groundedness_label')}")
        print(f"   * Retrieval Relevance: {eval_data.get('retrieval_relevance')}%")
        print(f"   * Source Coverage: {eval_data.get('source_coverage')}%")
        print(f"   * Confidence: {eval_data.get('confidence')}")
        print(f"   * Sources Used: {eval_data.get('sources_count')}")
        print(f"   * Pages Used: {eval_data.get('pages_count')}")
        print(f"   * Warning: {eval_data.get('warning')}")
        print(f"   * Retrieval Details items: {len(eval_data.get('retrieval_details', []))}")

        # Check retrieval details format
        for item in eval_data.get("retrieval_details", []):
            assert "rank" in item
            assert "document" in item
            assert "page" in item
            assert "relevance_pct" in item

    print("\n=================== ALL EVALUATION TESTS PASSED SUCCESSFULLY! ===================")

if __name__ == "__main__":
    run_tests()
