import json
import time
import urllib.request

BASE_URL = "http://127.0.0.1:5000"

def ask_question(question):
    payload = json.dumps({"question": question, "history": []}).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            data["client_time_sec"] = round(time.time() - t0, 2)
            return data
    except Exception as e:
        return {"success": False, "error": str(e), "client_time_sec": round(time.time() - t0, 2)}

if __name__ == "__main__":
    test_queries = [
        ("TEST 1: SQL", "SQL"),
        ("TEST 2: JOIN", "JOIN"),
        ("TEST 3: Primary Key", "Primary Key"),
        ("TEST 4: What are the types of JOIN?", "What are the types of JOIN?"),
        ("TEST 5: What is normalization?", "What is normalization?"),
        ("TEST 6: RandomXYZ123", "RandomXYZ123"),
        ("TEST 7: Term in PDF #2 (ACID)", "ACID"),
        ("TEST 8: Term in PDF #3 (Decorators)", "Decorators")
    ]

    print("\n=======================================================")
    print("RUNNING SMART RE-RANKING & SOURCE CITATIONS TESTS")
    print("=======================================================\n")

    for label, query in test_queries:
        print(f"--- {label} ---")
        res = ask_question(query)
        print(f"Query Type: {res.get('query_type')}")
        print(f"Response Time: {res.get('response_time')}s")
        print("Answer:\n" + str(res.get('answer', '')))
        print("Sources:")
        for s in res.get('sources', []):
            print(f"  * [Doc: {s.get('document')}] Page {s.get('page')} | Relevance: {s.get('relevance_pct')}% (Score: {s.get('score')}) | URL: {s.get('pdf_url')}")
        print("-" * 50)
