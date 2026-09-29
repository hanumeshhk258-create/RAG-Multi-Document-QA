import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def ask_rag(question, history=None, selected_docs=None, last_context=None):
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
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        res["elapsed"] = round(time.time() - t0, 2)
        return res

def run_tests():
    print("=================================================================")
    print("          STARTING COMPREHENSIVE RAG PIPELINE TEST SUITE         ")
    print("=================================================================\n")

    # Step 0: Check System Status
    req = urllib.request.Request(f"{BASE_URL}/api/status")
    with urllib.request.urlopen(req) as r:
        status_data = json.loads(r.read().decode('utf-8'))
    print(f"[*] System Status: {status_data.get('status')}")
    print(f"[*] Total Indexed Documents: {status_data.get('indexed_count')}")
    print(f"[*] Indexed Docs: {status_data.get('indexed_documents')}\n")

    # 1. Five specific required questions
    questions = [
        "What are SQL JOINs?",
        "What is Querying Data Commands?",
        "What is the Waterfall Model?",
        "What is the problem statement?",
        "What are the objectives?"
    ]

    answers = {}
    print("---------------- TEST 1: 5 SPECIFIC QUESTIONS ----------------")
    for q in questions:
        res = ask_rag(q)
        ans = res.get("answer", "")
        srcs = [f"{s['document']} p.{s['page']}" for s in res.get("sources", [])]
        cache_st = res.get("cache_status", "MISS")
        claims_supp = res.get("evaluation", {}).get("supported_claims", 0)
        tot_claims = res.get("evaluation", {}).get("total_claims", 0)
        answers[q] = ans
        print(f"\n[Q]: {q}")
        print(f"  Status: HTTP 200 ({res.get('elapsed')}s) | Cache: {cache_st}")
        print(f"  Sources: {srcs}")
        print(f"  Evidence: {claims_supp}/{tot_claims} supported claims | Confidence: {res.get('evaluation', {}).get('confidence')}")
        print(f"  Answer Snippet: {ans[:160]}...")

    # Check that unrelated questions didn't get same answer
    print("\n---------------- VERIFYING ANSWER UNIQUENESS ----------------")
    q_keys = list(questions)
    for i in range(len(q_keys)):
        for j in range(i+1, len(q_keys)):
            a1 = answers[q_keys[i]]
            a2 = answers[q_keys[j]]
            is_same = (a1 == a2 and len(a1) > 200)
            print(f"  Comparison '{q_keys[i]}' vs '{q_keys[j]}': {'DUPLICATE DETECTED!' if is_same else 'DIFFERENT & UNIQUE (PASS)'}")

    # 2. Test Conversation Memory: Turn 1 -> Turn 2 (Follow-up)
    print("\n---------------- TEST 2: CONVERSATION MEMORY & FOLLOW-UP ----------------")
    q_t1 = "What are SQL JOINs?"
    res_t1 = ask_rag(q_t1)
    print(f"Turn 1 Q: {q_t1}")
    print(f"Turn 1 A Snippet: {res_t1.get('answer', '')[:120]}...")

    hist = [
        {"role": "user", "content": q_t1},
        {"role": "assistant", "content": res_t1.get("answer", ""), "retrieved_context": res_t1.get("retrieved_context")}
    ]
    q_t2 = "What are its types?"
    res_t2 = ask_rag(q_t2, history=hist, last_context=res_t1.get("retrieved_context"))
    print(f"Turn 2 Q (Follow-up): {q_t2}")
    print(f"Turn 2 Mode: {res_t2.get('mode')} | is_followup: {res_t2.get('is_followup')}")
    print(f"Turn 2 Term/Context Topic: {res_t2.get('context_topic') or res_t2.get('term')}")
    print(f"Turn 2 Answer Snippet: {res_t2.get('answer', '')[:160]}...")

    # 3. Test New Chat Boundary
    print("\n---------------- TEST 3: NEW CHAT BOUNDARY ----------------")
    print("Simulating 'New Chat' click (cleared history)...")
    res_new_chat = ask_rag("What are its types?", history=[])
    print(f"Post-New-Chat Q: 'What are its types?'")
    print(f"Answer: {res_new_chat.get('answer')}")
    print(f"Correctly required clarification without assuming SQL JOINs: {res_new_chat.get('answer') == 'Please specify what you would like me to explain.' or 'specify' in res_new_chat.get('answer').lower()}")

    # 4. Test Cache Isolation: Q1 -> Q2 -> Q3 (repeat Q1)
    print("\n---------------- TEST 4: CACHE ISOLATION ----------------")
    print("[Step 1] Sending Q1: 'What are SQL JOINs?'")
    c_res1 = ask_rag("What are SQL JOINs?")
    print(f"  Q1 Cache: {c_res1.get('cache_status')} | Key: {c_res1.get('cache_key')}")

    print("[Step 2] Sending Q2: 'What is the Waterfall Model?'")
    c_res2 = ask_rag("What is the Waterfall Model?")
    print(f"  Q2 Cache: {c_res2.get('cache_status')} | Key: {c_res2.get('cache_key')}")
    print(f"  Q2 Answer has Waterfall keywords: {'waterfall' in c_res2.get('answer', '').lower()}")
    print(f"  Q2 Did NOT receive Q1 answer: {c_res1.get('answer') != c_res2.get('answer')}")

    print("[Step 3] Repeating Q3 (Same as Q1): 'What are SQL JOINs?'")
    c_res3 = ask_rag("What are SQL JOINs?")
    print(f"  Q3 Cache: {c_res3.get('cache_status')} | Key: {c_res3.get('cache_key')}")
    print(f"  Q3 Cache HIT: {c_res3.get('cache_status') == 'HIT'}")

    # 5. Test Unsupported Question
    print("\n---------------- TEST 5: UNSUPPORTED QUESTION GROUNDING ----------------")
    unsupp_q = "What is quantum gravity in cellular biology?"
    res_unsupp = ask_rag(unsupp_q)
    ans_text = res_unsupp.get('answer', '')
    is_strict_fallback = ans_text.startswith("I couldn't find") or ('sufficient information' in ans_text.lower()) or ans_text.startswith("NOT SUPPORTED")
    print(f"Strict fallback returned without hallucination: {is_strict_fallback}")

    print("\n=================================================================")
    print("                    ALL TESTS COMPLETED                          ")
    print("=================================================================")

if __name__ == "__main__":
    run_tests()
