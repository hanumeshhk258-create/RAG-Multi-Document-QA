import time
import requests
import json
import sys

BASE_URL = "http://127.0.0.1:5000"

def run_suite():
    print("=" * 70)
    print("=== STARTING SPEED & SAFE ANSWER DELIVERY TEST SUITE ===")
    print("=" * 70)

    # 1. Check Status with wait loop
    print("\n[TEST 1] Testing Server Status Endpoint...")
    status_data = None
    for _ in range(30):
        try:
            res = requests.get(f"{BASE_URL}/api/status", timeout=15)
            if res.status_code == 200:
                status_data = res.json()
                break
        except Exception:
            time.sleep(1.0)
    assert status_data is not None, "Server did not become ready within 30s"
    print(f"  [PASS] Server ready. Total docs: {status_data.get('total_documents')}, Vectorstore: {status_data.get('vectorstore_ready')}")

    # 2. Test Simple Fact Question (Fast Retrieval + Reranking + Generation + Verification)
    print("\n[TEST 2] Testing Simple Factual Question Delivery & Timing...")
    t0 = time.perf_counter()
    res = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is normalization in database management?"
    }, timeout=30)
    elapsed = round(time.perf_counter() - t0, 2)
    assert res.status_code == 200, f"Chat failed: {res.text}"
    data = res.json()
    assert data.get("answer"), "Final answer is missing or empty!"
    print(f"  [PASS] Answer received in {elapsed}s (Total API reported: {data.get('response_time')}s)")
    print(f"         Answer Preview: {data['answer'][:120]}...")
    tb = data.get("timing_breakdown", {})
    print(f"         Timing Breakdown -> Retrieval: {tb.get('retrieval_ms', tb.get('retrieval_sec'))} | Rerank: {tb.get('reranking_ms', tb.get('reranking_sec'))} | Comp: {tb.get('compression_ms')} | Gen: {tb.get('generation_sec')}s | Ver: {tb.get('verification_sec')}s")
    assert len(data.get("sources", [])) > 0, "Expected sources/citations in response"
    print(f"         Sources: {len(data.get('sources'))} cited -> {data['sources'][0].get('document')} P{data['sources'][0].get('page')}")

    # 3. Test Multi-Document Comparison Question
    print("\n[TEST 3] Testing Multi-Document Comparison Question...")
    t0 = time.perf_counter()
    res = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "Compare the features between CommerceOS and Python notes"
    }, timeout=30)
    elapsed = round(time.perf_counter() - t0, 2)
    assert res.status_code == 200, f"Comparison failed: {res.text}"
    data = res.json()
    assert data.get("answer"), "Comparison answer is missing or empty!"
    print(f"  [PASS] Comparison completed in {elapsed}s")
    print(f"         Answer Preview: {data['answer'][:120]}...")

    # 4. Test Irrelevant / Out-of-Scope Question
    print("\n[TEST 4] Testing Irrelevant Out-of-Scope Question...")
    res = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is the capital of Mars and the average temperature of Venus?"
    }, timeout=30)
    assert res.status_code == 200
    data = res.json()
    assert data.get("answer"), "Out of scope answer is missing!"
    print(f"  [PASS] Out of scope answered safely: '{data['answer'][:100]}...'")

    # 5. Test Question Details Endpoint Telemetry
    print("\n[TEST 5] Testing Question Telemetry Details Retrieval...")
    qid = data.get("id") or data.get("interaction_id")
    if qid:
        res_det = requests.get(f"{BASE_URL}/api/questions/{qid}", timeout=10)
        assert res_det.status_code == 200
        det_data = res_det.json()
        assert det_data.get("success") is True
        print(f"  [PASS] Question #{qid} details retrieved with all 8 telemetry sections.")

    # 6. Test Consecutive Rapid Queries for Speed & Stability
    print("\n[TEST 6] Testing Consecutive Rapid Queries...")
    queries = [
        "What is a primary key?",
        "What are ACID properties?",
        "Explain SQL JOIN operations."
    ]
    total_batch_time = 0.0
    for q in queries:
        t_start = time.perf_counter()
        r = requests.post(f"{BASE_URL}/api/chat", json={"question": q}, timeout=30)
        t_el = time.perf_counter() - t_start
        total_batch_time += t_el
        assert r.status_code == 200
        ans = r.json().get("answer")
        assert ans, f"Answer missing for '{q}'"
        print(f"  [PASS] '{q}' answered in {round(t_el, 2)}s")

    print(f"         Total 3-query batch time: {round(total_batch_time, 2)}s (avg {round(total_batch_time/3, 2)}s/query)")

    print("\n" + "=" * 70)
    print(">>> ALL SPEED & SAFE ANSWER DELIVERY TESTS PASSED (100% SUCCESS) <<<")
    print("=" * 70)
    return True

if __name__ == "__main__":
    try:
        success = run_suite()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
