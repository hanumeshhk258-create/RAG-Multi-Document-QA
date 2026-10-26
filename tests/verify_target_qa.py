import urllib.request
import json
import time

BASE = "http://127.0.0.1:5000"

test_cases = [
    {
        "q": "What is SQL?",
        "docs": ["SQL_CheatSheet.pdf"],
        "expected": ["relational", "database", "language", "query"]
    },
    {
        "q": "What is a primary key?",
        "docs": ["DBMS_Notes.pdf"],
        "expected": ["uniquely", "identif", "null", "key"]
    },
    {
        "q": "What are the types of database keys?",
        "docs": ["DBMS_Notes.pdf"],
        "expected": ["primary", "candidate", "foreign"]
    },
    {
        "q": "What are the four ACID properties?",
        "docs": ["DBMS_Notes.pdf"],
        "expected": ["atomicity", "consistency", "isolation", "durability"]
    },
    {
        "q": "What are the types of SQL JOINs?",
        "docs": ["SQL_CheatSheet.pdf"],
        "expected": ["inner", "left", "right", "full"]
    },
    {
        "q": "What is Durability?",
        "docs": ["DBMS_Notes.pdf"],
        "expected": ["committed", "permanent", "crash", "persist"]
    },
    {
        "q": "What is Python?",
        "docs": ["Python_Notes.pdf"],
        "expected": ["high-level", "interpreted", "programming"]
    },
    {
        "q": "What is quantum gravity in astrophysics?",
        "docs": ["DBMS_Notes.pdf"],
        "expected": ["find", "sufficient", "selected documents"],
        "is_unsupported": True
    }
]

print("=== TESTING TARGET QUESTIONS AGAINST LIVE OPTIMIZED SERVER ===")
for i, tc in enumerate(test_cases, 1):
    q = tc["q"]
    docs = tc["docs"]
    exp = tc["expected"]
    payload = json.dumps({"question": q, "selected_documents": docs}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(f"{BASE}/api/chat", data=payload, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=45) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    ans = res.get("answer", "")
    ans_lower = ans.lower()
    
    matches = [e for e in exp if e in ans_lower]
    passed = len(matches) >= (1 if tc.get("is_unsupported") else 2)
    status = "PASS" if passed else "FAIL"
    print(f"[{i}/8] {q} -> {status} ({dt:.2f}s)")
    print(f"     Answer excerpt: {ans[:90]}...")
    if tc.get("is_unsupported"):
        print(f"     Refusal verified: {passed}")
    else:
        print(f"     Matched keywords: {matches}")

# Follow-up test: ACID -> Explain the second property
print("\n=== TESTING CONVERSATION FOLLOW-UP SEQUENCE ===")
p1 = json.dumps({"question": "What are the four ACID properties?", "selected_documents": ["DBMS_Notes.pdf"]}).encode("utf-8")
req1 = urllib.request.Request(f"{BASE}/api/chat", data=p1, headers=headers)
with urllib.request.urlopen(req1) as r1:
    res1 = json.loads(r1.read().decode("utf-8"))
ans1 = res1.get("answer", "")

conv = [
    {"role": "user", "content": "What are the four ACID properties?"},
    {"role": "assistant", "content": ans1}
]
p2 = json.dumps({"question": "Explain the second property.", "conversation": conv, "selected_documents": ["DBMS_Notes.pdf"]}).encode("utf-8")
t0 = time.perf_counter()
req2 = urllib.request.Request(f"{BASE}/api/chat", data=p2, headers=headers)
with urllib.request.urlopen(req2, timeout=45) as r2:
    res2 = json.loads(r2.read().decode("utf-8"))
dt2 = time.perf_counter() - t0
ans2 = res2.get("answer", "")
fu_passed = "consistency" in ans2.lower()
fu_status = "PASS" if fu_passed else "FAIL"
print(f"Follow-up: Explain the second property -> {fu_status} ({dt2:.2f}s)")
print(f"Answer excerpt: {ans2[:100]}...")

# New Chat Isolation test
print("\n=== TESTING NEW CHAT ISOLATION ===")
req_nc = urllib.request.Request(f"{BASE}/api/new_chat", data=b"{}", headers=headers)
with urllib.request.urlopen(req_nc) as r_nc:
    res_nc = json.loads(r_nc.read().decode("utf-8"))
print(f"New Chat cleared: {res_nc.get('success', False)}")

# Test repeated query cache speed
print("\n=== TESTING QUERY CACHE SPEEDUP ===")
t0 = time.perf_counter()
req_cache = urllib.request.Request(f"{BASE}/api/chat", data=payload, headers=headers)
with urllib.request.urlopen(req_cache) as r_cache:
    res_cache = json.loads(r_cache.read().decode("utf-8"))
dt_cache = time.perf_counter() - t0
print(f"Cached repeated query latency: {dt_cache:.4f}s (Cache Status: {res_cache.get('cache_status', 'UNKNOWN')})")
