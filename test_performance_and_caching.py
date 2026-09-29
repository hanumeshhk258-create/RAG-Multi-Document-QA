import time
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"

def post_chat(question, selected_docs=None):
    payload = {
        "question": question,
        "selected_documents": selected_docs,
        "conversation": []
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            t1 = time.perf_counter()
            body = json.loads(resp.read().decode("utf-8"))
            return body, round(t1 - t0, 3)
    except urllib.error.HTTPError as e:
        t1 = time.perf_counter()
        body = json.loads(e.read().decode("utf-8"))
        return body, round(t1 - t0, 3)

def run_tests():
    print("=" * 70)
    print("PERFORMANCE & CACHING BENCHMARK TEST SUITE")
    print("=" * 70)
    
    # 1. Health check
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/status") as resp:
            status = json.loads(resp.read().decode("utf-8"))
            print(f"System status: {status.get('status')} | Indexed docs: {status.get('total_documents')}")
    except Exception as e:
        print(f"Could not connect to backend: {e}")
        return

    test_questions = [
        ("Test 1: What is Waterfall Model?", "What is Waterfall Model?", None),
        ("Test 2: What is the problem statement?", "What is the problem statement?", None),
        ("Test 3: What is the conclusion?", "What is the conclusion?", None),
        ("Test 4: What are the security and trust controls?", "What are the security and trust controls?", None),
        ("Test 5: Compare SQL and Python.", "Compare SQL and Python.", ["SQL_CheatSheet.pdf", "Python_Notes.pdf", "DBMS_Notes.pdf"]),
        ("Test 6 (Repeated Cache Test): What is Waterfall Model?", "What is Waterfall Model?", None),
    ]

    results = []

    for label, query, docs in test_questions:
        print(f"\nRunning: {label}...")
        res, wall_time = post_chat(query, docs)
        
        cache_status = res.get("cache_status", "HIT" if res.get("cache_hit") else "MISS")
        timing = res.get("timing_breakdown", {})
        eval_data = res.get("evaluation", {})
        ans_corr = res.get("answer_correction", {}) or eval_data.get("answer_correction", {})
        
        tot_time = res.get("response_time", wall_time)
        ret_time = timing.get("retrieval_sec", timing.get("retrieval_ms", 0))
        rerank_time = timing.get("reranking_sec", timing.get("reranking_ms", 0))
        comp_time = timing.get("compression_sec", timing.get("compression_ms", 0))
        gen_time = timing.get("generation_sec", 0)
        verif_time = timing.get("verification_sec", 0)
        
        confidence = eval_data.get("confidence", "N/A")
        grounding = eval_data.get("groundedness_score", "N/A")
        
        tot_claims = ans_corr.get("total_claims", eval_data.get("total_claims", 0))
        sup_claims = ans_corr.get("supported", eval_data.get("supported_claims", 0))
        part_claims = ans_corr.get("partially_supported", eval_data.get("partial_claims", 0))
        unsup_claims = ans_corr.get("unsupported", eval_data.get("unsupported_claims", 0))
        
        sources = res.get("sources", [])
        src_pages = [(s.get("document"), s.get("page")) for s in sources[:4]]
        
        record = {
            "test": label,
            "query": query,
            "wall_time": wall_time,
            "reported_time": tot_time,
            "cache_status": cache_status,
            "retrieval": ret_time,
            "reranking": rerank_time,
            "compression": comp_time,
            "generation": gen_time,
            "verification": verif_time,
            "confidence": confidence,
            "grounding": grounding,
            "claims": f"{sup_claims} sup, {part_claims} part, {unsup_claims} unsup (total: {tot_claims})",
            "sources": src_pages,
            "answer_snippet": (res.get("final_verified_answer") or res.get("answer") or "")[:150]
        }
        results.append(record)
        
        print(f"  -> Cache: {cache_status}")
        print(f"  -> Total time: {wall_time}s (Reported: {tot_time}s)")
        print(f"  -> Retrieval: {ret_time}s | Rerank: {rerank_time}s | Gen: {gen_time}s | Verif: {verif_time}s")
        print(f"  -> Grounding: {grounding}% | Confidence: {confidence}")
        print(f"  -> Claims: {record['claims']}")
        print(f"  -> Sources: {src_pages}")
        print(f"  -> Answer snippet: {record['answer_snippet']}...")

    print("\n" + "=" * 70)
    print("SUMMARY OF BENCHMARK RESULTS")
    print("=" * 70)
    for r in results:
        print(f"{r['test']}")
        print(f"   Time: {r['wall_time']}s | Cache: {r['cache_status']} | Confidence: {r['confidence']} | Grounding: {r['grounding']}%")
        print(f"   Breakdown: Ret: {r['retrieval']}s | Rerank: {r['reranking']}s | Gen: {r['generation']}s | Verif: {r['verification']}s")
        print(f"   Claims: {r['claims']}")
        print(f"   Sources: {r['sources']}")
        print("-" * 50)

if __name__ == "__main__":
    run_tests()
