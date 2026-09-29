import time
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_evidence_suite():
    print("=" * 80)
    print("TESTING EVIDENCE EXPLORER SUITE")
    print("=" * 80)

    # 1. Check Status
    status_res = requests.get(f"{BASE_URL}/api/status", timeout=10).json()
    print(f"System status: {status_res.get('status')}")
    assert status_res.get("status") == "ready", "System is not ready"

    # 2. Test Single Citation Query
    print("\n--- TEST 1: Question with Citations ('What is the Waterfall Model?') ---")
    t0 = time.perf_counter()
    res1 = requests.post(
        f"{BASE_URL}/api/chat",
        json={"question": "What is the Waterfall Model?"},
        timeout=25
    ).json()
    elapsed1 = round(time.perf_counter() - t0, 3)
    
    ans1 = res1.get("final_verified_answer") or res1.get("answer") or ""
    sources1 = res1.get("sources", [])
    print(f"Response ({elapsed1}s): {ans1[:180]}...")
    print(f"Sources count: {len(sources1)}")
    assert len(sources1) > 0, "Expected at least one retrieved source chunk"
    
    top_doc = sources1[0].get("document")
    top_page = sources1[0].get("page", 1)
    print(f"Top source: {top_doc}, Page: {top_page}")

    # 3. Test Evidence API lookup for top source
    print(f"\n--- TEST 2: Direct Evidence API Lookup ({top_doc}, Page {top_page}) ---")
    t_ev_0 = time.perf_counter()
    ev_res = requests.get(
        f"{BASE_URL}/api/evidence",
        params={"doc": top_doc, "page": top_page},
        timeout=5
    ).json()
    t_ev_elapsed = round((time.perf_counter() - t_ev_0) * 1000, 2)
    
    print(f"Evidence API lookup took: {t_ev_elapsed} ms")
    assert ev_res.get("success") is True, f"Evidence lookup failed: {ev_res}"
    assert ev_res.get("document") == top_doc, f"Expected {top_doc}, got {ev_res.get('document')}"
    assert ev_res.get("page") == top_page, f"Expected page {top_page}, got {ev_res.get('page')}"
    assert len(ev_res.get("text", "")) > 20, "Evidence chunk text is empty"
    print(f"Evidence Chunk ID: {ev_res.get('chunk_id')}")
    print(f"Retrieval Method: {ev_res.get('retrieval_method')}")
    print(f"Scores: Retrieval={ev_res.get('retrieval_score')}, Rerank={ev_res.get('rerank_score')}")
    print(f"Evidence Text Preview: {ev_res.get('text')[:150]}...")

    # 4. Test Multiple Citations Query
    print("\n--- TEST 3: Multi-Document Comparison ('Compare SQL and DBMS.') ---")
    t0 = time.perf_counter()
    res2 = requests.post(
        f"{BASE_URL}/api/chat",
        json={"question": "Compare SQL and DBMS."},
        timeout=25
    ).json()
    elapsed2 = round(time.perf_counter() - t0, 3)
    
    sources2 = res2.get("sources", [])
    print(f"Comparison ({elapsed2}s) returned {len(sources2)} sources across documents.")
    assert len(sources2) >= 2, "Expected multiple sources for comparison"
    
    # Verify each source chunk can be inspected via Evidence API
    for idx, s in enumerate(sources2[:3]):
        d_name = s.get("document")
        p_num = s.get("page", 1)
        ev_item = requests.get(
            f"{BASE_URL}/api/evidence",
            params={"doc": d_name, "page": p_num},
            timeout=5
        ).json()
        assert ev_item.get("success") is True, f"Evidence lookup failed for {d_name} p.{p_num}"
        print(f"  [OK] Source {idx+1}: {d_name} (Page {p_num}) - Evidence available ({len(ev_item.get('text', ''))} chars)")

    # 5. Test Non-Existent Document Evidence Error Handling
    print("\n--- TEST 4: Error Handling for Non-Existent Evidence ---")
    err_res = requests.get(
        f"{BASE_URL}/api/evidence",
        params={"doc": "NonExistentDocument_999.pdf", "page": 99},
        timeout=5
    )
    print(f"Non-existent evidence response status: {err_res.status_code}")
    err_data = err_res.json()
    assert err_data.get("success") is False, "Expected success=False for missing document"
    assert "no longer available" in err_data.get("error", "").lower(), "Expected clean error message"
    print(f"Error Message: {err_data.get('error')}")

    print("\n" + "=" * 80)
    print("ALL EVIDENCE EXPLORER BACKEND TESTS PASSED SUCCESSFULLY! (100%)")
    print("=" * 80)

if __name__ == "__main__":
    test_evidence_suite()
