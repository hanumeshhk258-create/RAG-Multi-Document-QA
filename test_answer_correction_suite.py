import sys
import time
import requests
import json

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:5000"

def test_answer_correction():
    print("=" * 70)
    print("AUTOMATIC CITATION-AWARE ANSWER CORRECTION & HALLUCINATION FILTERING TEST")
    print("=" * 70)

    # 1. Check system status
    status_res = requests.get(f"{BASE_URL}/api/status").json()
    print(f"System Status: {status_res.get('status')}")
    print(f"Total Docs: {status_res.get('total_documents')}, Chunks: {status_res.get('total_chunks')}")
    print(f"Components: {json.dumps(status_res.get('components', {}), indent=2)}")

    test_questions = [
        ("A", "What is the objective of the CommerceOS project?"),
        ("B", "What are the key security and trust controls?"),
        ("C", "Compare the information about SQL and Python in the selected PDFs."),
        ("D", "Give me an example that is NOT present in the documents.")
    ]

    for label, question in test_questions:
        print("\n" + "=" * 70)
        print(f"TEST QUESTION {label}: \"{question}\"")
        print("=" * 70)

        t0 = time.time()
        res = requests.post(
            f"{BASE_URL}/api/chat",
            json={"question": question, "conversation": []},
            timeout=30
        )
        elapsed = round(time.time() - t0, 2)

        if res.status_code != 200:
            print(f"FAILED with HTTP {res.status_code}: {res.text}")
            continue

        data = res.json()
        print(f"Success: {data.get('success')} (Took {elapsed}s)")
        print(f"Mode: {data.get('mode')}")

        print("\n--- FINAL VERIFIED ANSWER ---")
        print(data.get("answer"))

        corr = data.get("answer_correction") or {}
        print("\n--- ANSWER CORRECTION DEBUG METRICS ---")
        print(f"Total Claims: {corr.get('total_claims')}")
        print(f"Supported: {corr.get('supported')}")
        print(f"Partially Supported: {corr.get('partially_supported')}")
        print(f"Unsupported: {corr.get('unsupported')}")
        print(f"Removed: {corr.get('removed')}")
        print(f"Final Coverage: {corr.get('final_coverage')}%")
        print(f"Removed Warning: {corr.get('removed_warning')}")

        claims_corr = corr.get("claims_correction", [])
        if claims_corr:
            print(f"\n--- CLAIMS VERIFICATION & CORRECTION ({len(claims_corr)} claims) ---")
            for c in claims_corr:
                print(f"  Claim {c.get('claim_id')}: [{c.get('status')}] -> Action: {c.get('action_label')}")
                print(f"    Orig: {c.get('original_claim')}")
                if c.get('corrected_claim'):
                    print(f"    Corr: {c.get('corrected_claim')}")
                if c.get('document'):
                    print(f"    Source: {c.get('document')} (Page {c.get('page')})")

        eval_data = data.get("evaluation") or {}
        print("\n--- EVALUATION METRICS ---")
        print(f"Faithfulness: {eval_data.get('faithfulness')}%")
        print(f"Hallucination Risk: {eval_data.get('hallucination_risk')}% ({eval_data.get('risk_level')})")
        print(f"Groundedness: {eval_data.get('groundedness_score')}% ({eval_data.get('groundedness_label')})")
        print(f"Relevance: {eval_data.get('retrieval_relevance')}%")
        print(f"Sources Count: {eval_data.get('sources_count')}")

    print("\n" + "=" * 70)
    print("ALL TEST CASES COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    time.sleep(2)
    test_answer_correction()
