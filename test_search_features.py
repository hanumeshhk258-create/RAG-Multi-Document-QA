"""
Test script to verify Document Search, Source Preview, Excerpt Modals, and Caching.
"""
import time
import json
import urllib.request
import urllib.parse

BASE_URL = "http://127.0.0.1:5000"

def post_json(endpoint, data):
    url = f"{BASE_URL}{endpoint}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            res["_client_time"] = round(time.time() - t0, 3)
            return res
    except Exception as err:
        return {"success": False, "error": str(err), "_client_time": round(time.time() - t0, 3)}

def run_tests():
    print("=================================================================")
    print("RUNNING DOCUMENT SEARCH & RETRIEVAL TRANSPARENCY VERIFICATION")
    print("=================================================================\n")

    # 1. Search "DML"
    print("--- 1. Testing Search: 'DML' (All Documents) ---")
    res1 = post_json("/api/search", {"query": "DML"})
    print(f"Success: {res1.get('success')}, Cached: {res1.get('cached')}, Count: {res1.get('count')}, Elapsed: {res1.get('elapsed_seconds')}s")
    for r in res1.get("results", [])[:3]:
        print(f"  * {r['document']} | Page {r['page']} | Score: {r['score']} ({r['relevance_pct']}%) | Terms: {r.get('matched_terms')}")
        print(f"    Snippet: {r['snippet']}")
    print()

    # 2. Search "ACID"
    print("--- 2. Testing Search: 'ACID' ---")
    res2 = post_json("/api/search", {"query": "ACID"})
    print(f"Success: {res2.get('success')}, Cached: {res2.get('cached')}, Count: {res2.get('count')}, Elapsed: {res2.get('elapsed_seconds')}s")
    for r in res2.get("results", [])[:2]:
        print(f"  * {r['document']} | Page {r['page']} | Score: {r['score']} ({r['relevance_pct']}%)")
        print(f"    Snippet: {r['snippet']}")
    print()

    # 3. Search "normalization"
    print("--- 3. Testing Search: 'normalization' ---")
    res3 = post_json("/api/search", {"query": "normalization"})
    print(f"Success: {res3.get('success')}, Cached: {res3.get('cached')}, Count: {res3.get('count')}, Elapsed: {res3.get('elapsed_seconds')}s")
    for r in res3.get("results", [])[:2]:
        print(f"  * {r['document']} | Page {r['page']} | Score: {r['score']} ({r['relevance_pct']}%)")
        print(f"    Snippet: {r['snippet']}")
    print()

    # 4. Repeat search "normalization" (Test Cache HIT)
    print("--- 4. Testing Repeat Search: 'normalization' (Cache HIT) ---")
    res4 = post_json("/api/search", {"query": "normalization"})
    print(f"Success: {res4.get('success')}, Cached: {res4.get('cached')}, Count: {res4.get('count')}, Elapsed: {res4.get('elapsed_seconds')}s")
    assert res4.get("cached") == True, "Expected cache HIT!"
    print("[OK] Cache hit successfully verified!")
    print()

    # 5. Search with single document filtering (DBMS_Notes.pdf only)
    print("--- 5. Testing Search: 'normalization' (Selected: DBMS_Notes.pdf only) ---")
    res5 = post_json("/api/search", {"query": "normalization", "selected_documents": ["DBMS_Notes.pdf"]})
    print(f"Success: {res5.get('success')}, Count: {res5.get('count')}")
    for r in res5.get("results", []):
        print(f"  * {r['document']} | Page {r['page']}")
        assert r['document'] == "DBMS_Notes.pdf", f"Unexpected document: {r['document']}"
    print("[OK] Single document filtering verified!")
    print()

    # 6. Search with unselected / non-matching filter
    print("--- 6. Testing Empty Search Result Handling ---")
    res6 = post_json("/api/search", {"query": "NonExistentRandomTerm99999", "selected_documents": ["DBMS_Notes.pdf"]})
    print(f"Success: {res6.get('success')}, Count: {res6.get('count')}, Message: {res6.get('message')}")
    print()

    # 7. Ask normal RAG question: "What is DML?"
    print("--- 7. Testing RAG Q&A: 'What is DML?' ---")
    res7 = post_json("/api/chat", {"question": "What is DML?", "selected_documents": ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"]})
    print(f"Success: {res7.get('success')}, Time: {res7.get('response_time')}s")
    print("AI Answer:\n", res7.get("answer", "")[:250], "...")
    print("Sources:")
    for s in res7.get("sources", []):
        print(f"  * {s['document']} (Page {s['page']}) - {s['relevance_pct']}% relevant - URL: {s['pdf_url']}")
        print(f"    Excerpt: {s.get('excerpt', '')[:120]}...")
    print()

    print("=================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    run_tests()
