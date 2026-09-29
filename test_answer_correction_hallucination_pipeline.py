import time
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

QUESTIONS = [
    ("A. Waterfall Model", "What is Waterfall Model?"),
    ("B. Problem Statement", "What is the problem statement?"),
    ("C. Conclusion", "What is the conclusion?"),
    ("D. Security and Trust Controls", "What are the security and trust controls?"),
    ("E. Compare SQL and Python", "Compare SQL and Python."),
    ("F. Out-of-Domain Nonexistent", "What is the capital of Mars?")
]

def run_tests():
    # 1. Wait for server readiness
    for _ in range(10):
        try:
            r = requests.get(f"{BASE_URL}/api/status", timeout=5)
            if r.status_code == 200:
                print("Server is ready.")
                break
        except Exception:
            time.sleep(1)

    print("\n" + "="*70)
    print("STARTING ANSWER CORRECTION & HALLUCINATION CONTROL TEST SUITE")
    print("="*70 + "\n")

    for label, q in QUESTIONS:
        print(f"\n--- TESTING [{label}]: \"{q}\" ---")
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{BASE_URL}/api/chat",
                json={"question": q, "selected_documents": None},
                timeout=30
            )
            elapsed = round(time.perf_counter() - t0, 2)
            if resp.status_code != 200:
                print(f"FAILED: Status {resp.status_code}, Response: {resp.text[:200]}")
                continue

            data = resp.json()
            answer = data.get("answer") or ""
            eval_data = data.get("evaluation") or {}
            sources = data.get("sources") or []
            
            conf = eval_data.get("confidence") or "N/A"
            grounding = eval_data.get("groundedness_label") or "N/A"
            faith = eval_data.get("faithfulness", 0)
            risk = eval_data.get("risk_level") or "N/A"
            tot_claims = eval_data.get("total_claims", 0)
            sup_claims = eval_data.get("supported_claims", 0)
            part_claims = eval_data.get("partial_claims", 0)
            unsup_claims = eval_data.get("unsupported_claims", 0)

            print(f"Status: SUCCESS ({elapsed}s)")
            print(f"Confidence: {conf} | Grounding: {grounding} | Risk: {risk}")
            print(f"Faithfulness: {faith}% | Claims: {sup_claims} Sup / {part_claims} Part / {unsup_claims} Unsup (Total: {tot_claims})")
            print("Answer Preview:")
            lines = answer.strip().split("\n")
            preview = "\n".join(lines[:4])
            if len(lines) > 4:
                preview += f"\n... ({len(lines)-4} more lines)"
            print(preview)
            print("Sources:")
            for s in sources[:3]:
                doc = s.get("document") or s.get("filename")
                page = s.get("page")
                rel = s.get("relevance_pct")
                print(f"  • {doc} — Page {page} ({rel}% relevant)")

        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    run_tests()
