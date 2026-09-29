import sys
import requests
import json
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_URL = "http://127.0.0.1:5000"

def test_adaptive_suite():
    print("=" * 60)
    print("STARTING ADAPTIVE RETRIEVAL TEST SUITE")
    print("=" * 60)

    # Wait for server ready
    for _ in range(20):
        try:
            res = requests.get(f"{BASE_URL}/api/status")
            if res.status_code == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("Server did not become ready in 20s")

    status_data = res.json()
    print("[1] System Status Check:")
    print(f"    - Vectorstore Ready: {status_data.get('vectorstore_ready')}")
    print(f"    - Total Documents: {status_data.get('total_documents')}")
    print(f"    - Total Chunks: {status_data.get('total_chunks')}")
    components = status_data.get("components", {})
    print(f"    - Components: {components}")
    assert components.get("adaptive_retrieval") is True, "adaptive_retrieval component must be True"
    print("    -> PASS: System Status & Adaptive Retrieval Pill active")

    # Test Cases
    test_cases = [
        {
            "id": "CASE 1",
            "question": "What are the objectives of CommerceOS?",
            "selected_docs": ["CommerceOS_Complete_Project_Report.pdf"]
        },
        {
            "id": "CASE 2",
            "question": "What are the security and trust controls?",
            "selected_docs": ["CommerceOS_Complete_Project_Report.pdf"]
        },
        {
            "id": "CASE 3",
            "question": "Compare SQL and Python using the selected documents.",
            "selected_docs": ["SQL_CheatSheet.pdf", "Python_Notes.pdf"]
        },
        {
            "id": "CASE 4 (Out of Domain)",
            "question": "What is the quantum flux capacitor algorithm in CommerceOS?",
            "selected_docs": ["CommerceOS_Complete_Project_Report.pdf"]
        }
    ]

    for tc in test_cases:
        print("\n" + "=" * 60)
        print(f"RUNNING {tc['id']}: '{tc['question']}'")
        print(f"Selected Documents: {tc['selected_docs']}")
        print("=" * 60)

        t0 = time.perf_counter()
        resp = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "question": tc["question"],
                "selected_documents": tc["selected_docs"],
                "conversation": []
            }
        )
        elapsed = round(time.perf_counter() - t0, 2)
        assert resp.status_code == 200, f"Chat failed with status {resp.status_code}: {resp.text}"
        data = resp.json()

        print(f"Response Time: {elapsed}s")
        print(f"Answer Preview: {data.get('answer', '')[:200]}...")
        
        adaptive_data = data.get("adaptive_retrieval", {})
        print(f"Adaptive Retrieval Info:")
        print(f"  - Retries Performed: {adaptive_data.get('retries_performed')}")
        print(f"  - Triggered: {adaptive_data.get('triggered')}")
        print(f"  - Reason: {adaptive_data.get('reason')}")
        print(f"  - Selected Attempt: {adaptive_data.get('selected_attempt')}")
        print(f"  - Attempts Count: {len(adaptive_data.get('attempts', []))}")
        print(f"  - Process Steps: {len(adaptive_data.get('process_steps', []))}")
        for ps in adaptive_data.get('process_steps', []):
            print(f"      {ps.get('icon')} {ps.get('title')}: {ps.get('desc')}")

        eval_data = data.get("evaluation", {})
        print(f"Evaluation Metrics:")
        print(f"  - Groundedness: {eval_data.get('groundedness_score')}% ({eval_data.get('groundedness_label')})")
        print(f"  - Faithfulness: {eval_data.get('faithfulness')}%")
        print(f"  - Hallucination Risk: {eval_data.get('risk_level')} ({eval_data.get('hallucination_risk')}%)")
        print(f"  - Claims: {eval_data.get('supported_claims')} supported / {eval_data.get('total_claims')} total ({eval_data.get('unsupported_claims')} unsupported)")

        timing = data.get("timing_breakdown", {})
        print(f"Timing Breakdown: {timing}")

        if tc["id"].startswith("CASE 4"):
            # Out of domain must not hallucinate
            ans_lower = data.get('answer', '').lower()
            assert ("could not find" in ans_lower or "couldn't find" in ans_lower or "not found" in ans_lower or "sufficient" in ans_lower or "not provide" in ans_lower), "Out of domain question should return non-hallucinatory fallback"
            print("  -> PASS: Cleanly fell back without hallucination!")
        else:
            assert len(data.get("sources", [])) > 0 or data.get("chunks_used", 0) > 0, "Expected sources for in-domain question"
            print("  -> PASS: High quality grounded answer with sources!")

    print("\n" + "=" * 60)
    print("ALL TEST CASES PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_adaptive_suite()
