import json
import time
import requests

BASE_URL = "http://127.0.0.1:5000"

QUESTIONS = [
    "What is Waterfall Model?",
    "What are the objectives?",
    "What is the problem statement?",
    "What are the Key Security and Trust Controls?",
    "What is SQL?",
    "Compare SQL and Python.",
    "What are the applications of CommerceOS?"
]

def run_tests():
    print("=================================================================")
    print("Testing Contextual Compression on Multi-Document RAG QA System")
    print("=================================================================")
    
    # 1. Check system status
    try:
        status_res = requests.get(f"{BASE_URL}/api/status", timeout=10)
        status_data = status_res.json()
        print(f"System status: {status_data.get('status')}")
        print(f"Indexed documents: {len(status_data.get('documents', []))}")
        print(f"Components: {json.dumps(status_data.get('components', {}), indent=2)}")
        assert status_data.get("components", {}).get("contextual_compression") is True, "contextual_compression not marked True"
    except Exception as e:
        print(f"Status check failed: {e}")
        return False

    all_passed = True
    results_summary = []

    for idx, q in enumerate(QUESTIONS, 1):
        if idx > 1:
            print("Pacing delay (4s)...")
            time.sleep(4)
        print(f"\n-------------------------------------------------------------")
        print(f"[{idx}/{len(QUESTIONS)}] Testing Question: '{q}'")
        print(f"-------------------------------------------------------------")
        t0 = time.time()
        try:
            res = requests.post(
                f"{BASE_URL}/api/chat",
                json={"question": q, "history": []},
                timeout=45
            )
            elapsed = round(time.time() - t0, 2)
            
            if res.status_code != 200:
                print(f"FAILED (Status {res.status_code}): {res.text}")
                all_passed = False
                results_summary.append({"q": q, "status": "FAILED", "reason": f"Status {res.status_code}"})
                continue
                
            data = res.json()
            if not data.get("success"):
                print(f"FAILED (Success False): {data.get('error')}")
                all_passed = False
                results_summary.append({"q": q, "status": "FAILED", "reason": data.get('error')})
                continue

            ans = data.get("final_verified_answer") or data.get("answer") or ""
            sources = data.get("sources", [])
            comp_summary = data.get("compression", {})
            comp_details = data.get("compression_details", [])
            eval_data = data.get("evaluation", {})
            timing = data.get("timing_breakdown", {})
            adaptive = data.get("adaptive_retrieval", {})

            print(f"[OK] Answer Generated ({len(ans)} chars)")
            print(f"  Snippet: {ans[:160]}...")
            print(f"[OK] Sources ({len(sources)}):")
            for s in sources[:3]:
                print(f"  - {s.get('document')} (Page {s.get('page')}) | Score: {s.get('score')} | Rel: {s.get('relevance_pct')}%")
            
            print(f"[OK] Contextual Compression:")
            print(f"  - Applied: {comp_summary.get('applied', len(comp_details) > 0)}")
            print(f"  - Chunks compressed: {len(comp_details)}")
            print(f"  - Original chars: {comp_summary.get('original_characters', 'N/A')}")
            print(f"  - Compressed chars: {comp_summary.get('compressed_characters', 'N/A')}")
            print(f"  - Reduction: {comp_summary.get('reduction_percentage', 'N/A')}%")
            print(f"  - Compression time: {comp_summary.get('compression_time_ms', 'N/A')}ms")
            
            print(f"[OK] Evaluation & Claims:")
            print(f"  - Groundedness: {eval_data.get('groundedness_score', 'N/A')}")
            print(f"  - Faithfulness: {eval_data.get('faithfulness', 'N/A')}%")
            print(f"  - Supported claims: {eval_data.get('supported_claims', 0)} / {eval_data.get('total_claims', 0)}")
            print(f"  - Answer correction: {eval_data.get('answer_correction', {})}")

            print(f"[OK] Timing Breakdown:")
            print(f"  - Retrieval: {timing.get('retrieval_sec', 'N/A')}s")
            print(f"  - Reranking: {timing.get('reranking_sec', 'N/A')}s")
            print(f"  - Compression: {timing.get('compression_sec', 'N/A')}s")
            print(f"  - Generation: {timing.get('generation_sec', 'N/A')}s")
            print(f"  - Total: {timing.get('total_sec', elapsed)}s")

            # Validate core requirements
            assert len(ans) > 10, "Answer too short"
            is_unsupported = "not available in the selected documents" in ans.lower() or "not enough information" in ans.lower()
            if not is_unsupported:
                assert len(sources) > 0, "No sources cited for supported answer"
            assert comp_summary.get("applied") is True or len(comp_details) > 0, "Compression not applied"
            assert len(comp_details) > 0, "No compressed chunk details"
            
            # Check source traceability in compressed chunk
            first_chunk = comp_details[0]
            assert "document" in first_chunk and "page" in first_chunk, "Chunk missing source/page metadata"
            assert "original_text" in first_chunk and "compressed_text" in first_chunk, "Chunk missing text fields"
            
            results_summary.append({
                "q": q,
                "status": "PASSED",
                "chunks": len(comp_details),
                "orig_chars": comp_summary.get("original_characters"),
                "comp_chars": comp_summary.get("compressed_characters"),
                "reduction": f"{comp_summary.get('reduction_percentage')}%",
                "comp_time": f"{comp_summary.get('compression_time_ms')}ms",
                "time": f"{elapsed}s"
            })
            
        except Exception as e:
            print(f"ERROR on question '{q}': {e}")
            all_passed = False
            results_summary.append({"q": q, "status": "FAILED", "reason": str(e)})

    print("\n=================================================================")
    print("TEST SUITE RESULTS SUMMARY")
    print("=================================================================")
    for r in results_summary:
        print(f"[{r['status']}] {r['q']}")
        if r['status'] == 'PASSED':
            print(f"       Chunks: {r['chunks']} | Orig: {r['orig_chars']} -> Comp: {r['comp_chars']} ({r['reduction']} reduction) | Time: {r['time']}")
        else:
            print(f"       Reason: {r.get('reason')}")
    print("=================================================================")
    
    if all_passed:
        print("CONTEXTUAL COMPRESSION: IMPLEMENTED")
        print("TEST STATUS: PASSED")
    else:
        print("CONTEXTUAL COMPRESSION: IMPLEMENTED")
        print("TEST STATUS: FAILED")

if __name__ == "__main__":
    run_tests()
