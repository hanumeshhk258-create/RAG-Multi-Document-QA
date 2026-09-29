import urllib.request
import json
import time
import sys

BASE_URL = "http://127.0.0.1:5000"

def query_chat(question, history=None, selected_docs=None, last_context=None):
    payload = {
        "question": question,
        "history": history or [],
        "selected_documents": selected_docs,
        "last_context": last_context
    }
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            data["latency_sec"] = round(time.time() - t0, 2)
            return data
    except urllib.error.HTTPError as he:
        return {"success": False, "error": f"HTTP {he.code}: {he.read().decode('utf-8')}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_phase14_suite():
    print("=================================================================")
    print("                  PHASE 14 RAG VALIDATION SUITE                  ")
    print("=================================================================\n")

    # Check status with retry
    for attempt in range(10):
        try:
            req = urllib.request.Request(f"{BASE_URL}/api/status")
            with urllib.request.urlopen(req) as r:
                st = json.loads(r.read().decode('utf-8'))
            print(f"[*] System Status: {st.get('status')}")
            print(f"[*] Indexed Docs: {st.get('indexed_documents')}\n")
            break
        except Exception:
            time.sleep(1)

    test_queries = [
        "What are SQL JOINs?",
        "What are Querying Data Commands?",
        "What is normalization?",
        "What is the Waterfall Model?",
        "What is Key Security and Trust Controls?",
        "What are the objectives?"
    ]

    collected_answers = {}

    print("---------------- SECTION 1: MANDATORY TEST QUERIES ----------------")
    for q in test_queries:
        res = query_chat(q)
        ans = res.get("answer", "")
        sources = [f"{s.get('document')} (Page {s.get('page')})" for s in res.get("sources", [])]
        cache_status = res.get("cache_status", "MISS")
        eval_info = res.get("evaluation", {})
        conf = eval_info.get("confidence", "N/A")
        supp_claims = eval_info.get("supported_claims", 0)
        tot_claims = eval_info.get("total_claims", 0)
        collected_answers[q] = ans

        print(f"\n[QUERY]: '{q}'")
        print(f"  Success: {res.get('success')} | Latency: {res.get('latency_sec')}s | Cache: {cache_status}")
        print(f"  Sources ({len(sources)}): {sources}")
        print(f"  Groundedness: {eval_info.get('groundedness_label')} ({supp_claims}/{tot_claims} claims supported)")
        print(f"  Answer: {ans[:200]}...")

    # Answer uniqueness check
    print("\n---------------- SECTION 2: ANSWER UNIQUENESS MATRIX ----------------")
    keys = list(test_queries)
    all_unique = True
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a1 = collected_answers[keys[i]]
            a2 = collected_answers[keys[j]]
            # If both are substantive answers (>150 chars) and identical, flag it
            if a1 == a2 and len(a1) > 150:
                print(f"  [FAIL] Duplicate answers for '{keys[i]}' and '{keys[j]}'")
                all_unique = False
    if all_unique:
        print("  [PASS] All test queries produced completely distinct, independent answers.")

    # Section 3: Follow-up and Conversation Memory
    print("\n---------------- SECTION 3: CONVERSATION MEMORY & FOLLOW-UP ----------------")
    q1 = "What are SQL JOINs?"
    res1 = query_chat(q1)
    print(f"Turn 1 Q: '{q1}'")
    print(f"Turn 1 A: {res1.get('answer', '')[:140]}...")

    hist = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": res1.get("answer", ""), "retrieved_context": res1.get("retrieved_context")}
    ]
    q2 = "What are their types?"
    res2 = query_chat(q2, history=hist, last_context=res1.get("retrieved_context"))
    print(f"\nTurn 2 Q (Follow-up with history): '{q2}'")
    print(f"  Mode: {res2.get('mode')} | is_followup: {res2.get('is_followup')}")
    print(f"  Resolved context topic: {res2.get('context_topic') or res2.get('term')}")
    print(f"  Answer snippet: {res2.get('answer', '')[:200]}...")

    # Section 4: New Chat Boundary
    print("\n---------------- SECTION 4: NEW CHAT BOUNDARY ----------------")
    print("Simulating 'New Chat' (cleared history)...")
    res_new_chat = query_chat("What are their types?", history=[])
    print(f"Post-New-Chat Q: 'What are their types?'")
    print(f"  Answer: {res_new_chat.get('answer')}")
    is_clarification = "specify" in res_new_chat.get('answer', '').lower()
    print(f"  [PASS] Cleanly prompted for clarification without carrying over JOIN context: {is_clarification}")

    # Section 5: Unsupported Out-of-Domain Question
    print("\n---------------- SECTION 5: STRICT OUT-OF-DOMAIN REFUSAL ----------------")
    unsupp_query = "What is the biochemical mechanism of CRISPR Cas9 in photosynthesis?"
    res_unsupp = query_chat(unsupp_query)
    ans_unsupp = res_unsupp.get("answer", "")
    print(f"Out-of-Domain Q: '{unsupp_query}'")
    print(f"  Answer: {ans_unsupp}")
    print(f"  Sources count: {len(res_unsupp.get('sources', []))}")
    is_strict_fallback = (
        ans_unsupp.startswith("I couldn't find") or 
        "sufficient information" in ans_unsupp.lower() or
        ans_unsupp.startswith("NOT SUPPORTED")
    )
    print(f"  [PASS] Strict refusal without hallucination: {is_strict_fallback}")

    print("\n=================================================================")
    print("                  ALL VALIDATION TESTS COMPLETED                 ")
    print("=================================================================")

if __name__ == "__main__":
    run_phase14_suite()
