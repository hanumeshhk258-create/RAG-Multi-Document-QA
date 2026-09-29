import time
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_status():
    print("Testing /api/status...")
    res = requests.get(f"{BASE_URL}/api/status")
    print(f"Status Code: {res.status_code}")
    data = res.json()
    print(f"Total Documents: {data.get('total_documents')}, Total Chunks: {data.get('total_chunks')}, Ready: {data.get('vectorstore_ready')}")
    assert res.status_code == 200
    assert data.get("vectorstore_ready") is True
    print("PASS: System status is ready.\n")

def test_simple_question():
    print("Testing Simple Question: 'What is Waterfall Model?' (FAST MODE)...")
    payload = {
        "question": "What is Waterfall Model?",
        "conversation": []
    }
    t0 = time.time()
    res = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=30)
    elapsed = round(time.time() - t0, 2)
    print(f"Response Status Code: {res.status_code} in {elapsed}s")
    assert res.status_code == 200
    data = res.json()
    print(f"Success: {data.get('success')}")
    print(f"Final Verified Answer:\n{data.get('final_verified_answer') or data.get('answer')}")
    print(f"Sources ({len(data.get('sources', []))}):")
    for s in data.get('sources', [])[:3]:
        print(f"  - {s.get('document')} (Page {s.get('page')}) [{s.get('relevance_pct')}% relevant]")
    print(f"Timing Breakdown: {data.get('timing_breakdown')}")
    print(f"Response Time: {data.get('response_time')}s")
    print(f"Evaluation Metrics:")
    eval_d = data.get('evaluation', {})
    print(f"  - Groundedness: {eval_d.get('groundedness_score')}% ({eval_d.get('groundedness_label')})")
    print(f"  - Faithfulness: {eval_d.get('faithfulness')}%")
    print(f"  - Retrieval Relevance: {eval_d.get('retrieval_relevance')}%")
    print(f"  - Claims: Supported={eval_d.get('supported_claims')}, Partial={eval_d.get('partial_claims')}, Unsupported={eval_d.get('unsupported_claims')}")
    print("PASS: Fast response pipeline completed successfully.\n")

def test_table_question():
    print("Testing Table Question: 'What is the advantage of KNN?'...")
    payload = {
        "question": "What is the advantage of KNN?",
        "conversation": []
    }
    t0 = time.time()
    res = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=30)
    elapsed = round(time.time() - t0, 2)
    print(f"Response Status Code: {res.status_code} in {elapsed}s")
    assert res.status_code == 200
    data = res.json()
    print(f"Answer:\n{data.get('answer')}")
    sd = data.get("structured_data") or {}
    print(f"Structured Data: Tables detected={sd.get('tables_detected')}, Tables retrieved={sd.get('tables_retrieved')}")
    print("PASS: Table question completed.\n")

if __name__ == "__main__":
    test_status()
    test_simple_question()
    test_table_question()
    print("ALL BACKEND PIPELINE TESTS PASSED!")
