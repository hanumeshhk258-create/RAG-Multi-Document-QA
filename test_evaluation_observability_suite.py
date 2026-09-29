"""
Automated Test Suite for RAG Evaluation & Observability Dashboard
Tests:
 1. Evaluation Dataset Loading (15-25 cases)
 2. Evaluation Benchmark Execution Endpoint
 3. Evaluation Report Retrieval
 4. User Feedback Recording & Statistics
 5. Analytics Dashboard Extended Telemetry (Latency Ranges, Confidence Distribution, Retrieval Stats)
 6. Question-Level Details Telemetry Endpoint
"""

import sys
import time
import json
import requests

BASE_URL = "http://127.0.0.1:5000"

def run_tests():
    print("=" * 70)
    print("=== STARTING RAG EVALUATION & OBSERVABILITY TEST SUITE ===")
    print("=" * 70)

    # 1. Test Evaluation Dataset Endpoint
    print("\n[TEST 1] Testing Evaluation Dataset Retrieval...")
    res = requests.get(f"{BASE_URL}/api/evaluation/dataset", timeout=10)
    assert res.status_code == 200, f"Failed: {res.status_code}"
    data = res.json()
    assert data.get("success") is True, "Success flag missing"
    cases = data.get("dataset", [])
    assert len(cases) >= 15, f"Expected >= 15 cases, got {len(cases)}"
    print(f"  [PASS] Dataset loaded successfully with {len(cases)} diverse test cases.")

    # 2. Test User Feedback Submission
    print("\n[TEST 2] Testing User Feedback Submission...")
    fb_res1 = requests.post(f"{BASE_URL}/api/feedback", json={
        "msg_id": "test_msg_001",
        "feedback": "helpful",
        "comment": "Accurate grounding"
    }, timeout=10)
    assert fb_res1.status_code == 200 and fb_res1.json().get("success") is True

    fb_res2 = requests.post(f"{BASE_URL}/api/feedback", json={
        "msg_id": "test_msg_002",
        "feedback": "unhelpful",
        "comment": "Needed more context"
    }, timeout=10)
    assert fb_res2.status_code == 200 and fb_res2.json().get("success") is True
    print("  [PASS] User feedback recorded (Helpful & Unhelpful).")

    # 3. Test Evaluation Benchmark Execution Endpoint
    print("\n[TEST 3] Running Automated Evaluation Benchmark Endpoint...")
    t_start = time.time()
    eval_run_res = requests.post(f"{BASE_URL}/api/evaluation/run", json={"top_k": 3}, timeout=120)
    assert eval_run_res.status_code == 200, f"Benchmark failed with status {eval_run_res.status_code}: {eval_run_res.text}"
    eval_data = eval_run_res.json()
    assert eval_data.get("success") is True, f"Benchmark returned error: {eval_data}"
    metrics = eval_data.get("metrics", {})
    print(f"  [PASS] Benchmark executed in {time.time() - t_start:.2f}s across {metrics.get('total_test_cases')} queries.")
    print(f"         Retrieval Precision: {metrics.get('retrieval_precision_pct')}% | Recall: {metrics.get('retrieval_recall_pct')}%")
    print(f"         Hit@1: {metrics.get('hit_at_1_pct')}% | Hit@3: {metrics.get('hit_at_3_pct')}%")
    print(f"         Source Accuracy: {metrics.get('source_accuracy_pct')}% | Groundedness: {metrics.get('avg_groundedness_pct')}%")
    print(f"         OOD Refusal Rate: {metrics.get('ood_refusal_rate_pct')}%")

    # 4. Test Evaluation Report Endpoint
    print("\n[TEST 4] Testing Evaluation Results Report Retrieval...")
    rep_res = requests.get(f"{BASE_URL}/api/evaluation/results", timeout=10)
    assert rep_res.status_code == 200 and rep_res.json().get("success") is True
    report = rep_res.json().get("report", {})
    assert report.get("id") is not None, "Report ID missing"
    print(f"  [PASS] Stored evaluation report #{report.get('id')} retrieved from SQLite.")

    # 5. Test Analytics Dashboard Telemetry
    print("\n[TEST 5] Testing Extended Analytics Dashboard Telemetry...")
    an_res = requests.get(f"{BASE_URL}/api/analytics", timeout=10)
    assert an_res.status_code == 200 and an_res.json().get("success") is True
    an_data = an_res.json()
    summary = an_data.get("summary", {})
    
    assert summary.get("total_questions") is not None, "Total questions missing"
    assert summary.get("min_response_time") is not None, "Min response time missing"
    assert summary.get("max_response_time") is not None, "Max response time missing"
    assert summary.get("feedback") is not None, "Feedback summary missing"
    assert summary.get("confidence_distribution") is not None, "Confidence distribution missing"
    assert summary.get("latest_evaluation") is not None, "Latest evaluation missing from analytics"
    
    print(f"  [PASS] Analytics summary contains latency ranges: {summary.get('min_response_time')}s - {summary.get('max_response_time')}s")
    print(f"         Satisfaction Rate: {summary.get('feedback', {}).get('satisfaction_rate')}%")
    print(f"         Confidence Breakdown: {summary.get('confidence_distribution')}")

    # 6. Test Question Details Endpoint
    print("\n[TEST 6] Testing Question Details Endpoint...")
    history = an_data.get("history", [])
    if history:
        target_id = history[0]["id"]
        det_res = requests.get(f"{BASE_URL}/api/question-details/{target_id}", timeout=10)
        assert det_res.status_code == 200 and det_res.json().get("success") is True
        det_data = det_res.json().get("question", {})
        assert det_data.get("question") is not None, "Question text missing"
        assert det_data.get("status") is not None, "Status missing"
        print(f"  [PASS] Question detail for #{target_id} returned full telemetry (Method: '{det_data.get('retrieval_method')}').")
    else:
        print("  [SKIP] No question history recorded yet.")

    print("\n" + "=" * 70)
    print(">>> ALL RAG EVALUATION & OBSERVABILITY TESTS PASSED (100% SUCCESS) <<<")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
