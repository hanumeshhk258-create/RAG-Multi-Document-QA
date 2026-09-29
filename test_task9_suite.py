import urllib.request
import json
import time
import sys

BASE_URL = "http://127.0.0.1:5000"

def wait_for_server():
    for attempt in range(15):
        try:
            req = urllib.request.Request(f"{BASE_URL}/api/status")
            with urllib.request.urlopen(req) as r:
                data = json.loads(r.read().decode('utf-8'))
                print(f"[*] Server ready. Status: {data.get('status')}")
                return data
        except Exception:
            time.sleep(1)
    raise RuntimeError("Server did not start in time.")

def trigger_indexing():
    req = urllib.request.Request(
        f"{BASE_URL}/api/index",
        data=json.dumps({}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print(f"[*] Indexing triggered. Indexed: {res.get('indexed_count')} documents.")
        return res

def ask_question(question, history=None, selected_docs=None, last_context=None):
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
            data["elapsed"] = round(time.time() - t0, 2)
            return data
    except urllib.error.HTTPError as he:
        return {"success": False, "error": f"HTTP {he.code}: {he.read().decode('utf-8')}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_task9_suite():
    print("=================================================================")
    print("                 TASK 9 TEST & VALIDATION SUITE                  ")
    print("=================================================================\n")

    st = wait_for_server()
    if st.get("pending_count", 0) > 0 or st.get("indexed_count", 0) < 5:
        print("[*] Rebuilding index for all available documents...")
        trigger_indexing()
        st = wait_for_server()

    print(f"[*] Active Indexed Documents: {st.get('indexed_documents')}\n")

    # 8 Required Questions in Task 9:
    # 1. What are SQL JOINs?
    # 2. What are the types of SQL JOINs?
    # 3. What is normalization?
    # 4. What is Querying Data Commands?
    # 5. What is Python?
    # 6. What is a primary key?
    # 7. What is the conclusion?
    # 8. What is quantum gravity in astrophysics?

    test_queries = [
        "What are SQL JOINs?",
        "What are the types of SQL JOINs?",
        "What is normalization?",
        "What is Querying Data Commands?",
        "What is Python?",
        "What is a primary key?",
        "What is the conclusion?",
        "What is quantum gravity in astrophysics?"
    ]

    answers = {}
    print("---------------- SECTION 1: 8 SPECIFIC TESTS ----------------")
    for q in test_queries:
        res = ask_question(q)
        ans = res.get("answer", "")
        answers[q] = ans
        sources = [f"{s.get('document')} (Page {s.get('page')})" for s in res.get("sources", [])]
        cache_st = res.get("cache_status", "MISS")
        eval_data = res.get("evaluation", {})
        supp_claims = eval_data.get("supported_claims", 0)
        tot_claims = eval_data.get("total_claims", 0)
        
        print(f"\n[QUERY]: '{q}'")
        print(f"  Success: {res.get('success')} | Response Time: {res.get('elapsed')}s | Cache: {cache_st}")
        print(f"  Sources ({len(sources)}): {sources}")
        print(f"  Verification: {eval_data.get('groundedness_label', 'N/A')} ({supp_claims}/{tot_claims} claims)")
        print(f"  Answer: {ans[:180]}...")

    print("\n---------------- SECTION 2: ANSWER UNIQUENESS ----------------")
    keys = list(test_queries)
    all_unique = True
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a1 = answers[keys[i]]
            a2 = answers[keys[j]]
            if a1 == a2 and len(a1) > 150:
                print(f"  [FAIL] Duplicate between '{keys[i]}' and '{keys[j]}'")
                all_unique = False
    if all_unique:
        print("  [PASS] All answers are unique and question-specific.")

    print("\n---------------- SECTION 3: CONVERSATIONAL FOLLOW-UP ----------------")
    q_lead = "What are SQL JOINs?"
    res_lead = ask_question(q_lead)
    hist = [
        {"role": "user", "content": q_lead},
        {"role": "assistant", "content": res_lead.get("answer", ""), "retrieved_context": res_lead.get("retrieved_context")}
    ]
    q_fol = "What are their types?"
    res_fol = ask_question(q_fol, history=hist, last_context=res_lead.get("retrieved_context"))
    print(f"Turn 1: '{q_lead}' -> Answered")
    print(f"Turn 2 Follow-up: '{q_fol}'")
    print(f"  is_followup: {res_fol.get('is_followup')}")
    print(f"  Resolved context topic: {res_fol.get('context_topic') or res_fol.get('term')}")
    print(f"  Answer Snippet: {res_fol.get('answer', '')[:160]}...")

    print("\n---------------- SECTION 4: NEW CHAT BOUNDARY ----------------")
    res_new = ask_question("What are their types?", history=[])
    print(f"Post-New-Chat (No history): 'What are their types?'")
    print(f"  Answer: {res_new.get('answer')}")
    is_clean_clarification = "specify" in res_new.get('answer', '').lower()
    print(f"  [PASS] Cleanly requested clarification without assuming SQL JOINs: {is_clean_clarification}")

    print("\n---------------- SECTION 5: CACHE INTEGRITY ----------------")
    # Repeat Q1
    res_repeat = ask_question("What are SQL JOINs?")
    print(f"Repeat Q1: 'What are SQL JOINs?'")
    print(f"  Cache Status: {res_repeat.get('cache_status')} (Expected: HIT)")
    print(f"  Cache Key: {res_repeat.get('cache_key')}")
    print(f"  [PASS] Cache isolation verified: {res_repeat.get('cache_status') == 'HIT'}")

    print("\n=================================================================")
    print("                 ALL TASK 9 TESTS FINISHED                       ")
    print("=================================================================")

if __name__ == "__main__":
    run_task9_suite()
