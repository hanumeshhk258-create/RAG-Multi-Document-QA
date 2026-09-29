import json
import time
import urllib.request
import urllib.parse

BASE_URL = "http://127.0.0.1:5000"

def check_status():
    req = urllib.request.Request(f"{BASE_URL}/api/status")
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print("SYSTEM STATUS:", json.dumps(data, indent=2))
        return data

def ask_question(question, history=None):
    payload = json.dumps({"question": question, "history": history or []}).encode('utf-8')
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
    print("Checking status...")
    status = check_status()

    test_queries = [
        ("TEST 1: SQL", "SQL"),
        ("TEST 2: What is SQL?", "What is SQL?"),
        ("TEST 3: JOIN", "JOIN"),
        ("TEST 4: What are the types of JOIN?", "What are the types of JOIN?"),
        ("TEST 5: Primary Key", "Primary Key"),
        ("TEST 6: What is Python?", "What is Python?"),
        ("TEST 7: RandomXYZ123", "RandomXYZ123")
    ]

    print("\n=======================================================")
    print("RUNNING SMART SEARCH & VERIFIED CITATIONS TESTS")
    print("=======================================================\n")

    for label, query in test_queries:
        print(f"--- {label} ---")
        res = ask_question(query)
        print(f"Query Type: {res.get('query_type')}")
        print(f"Response Time: {res.get('response_time')}s (Client: {res.get('client_time_sec')}s)")
        print("Answer:\n" + str(res.get('answer', '')))
        print("Sources:")
        for s in res.get('sources', []):
            print(f"  * [Doc: {s.get('document')}] Page {s.get('page')} | Relevance: {s.get('relevance_pct')}% | URL: {s.get('pdf_url')}")
        print()
