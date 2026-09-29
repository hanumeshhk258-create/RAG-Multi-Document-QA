import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def post_chat(payload):
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return resp.status, data, round(time.time() - t0, 2)
    except urllib.error.HTTPError as he:
        try:
            data = json.loads(he.read().decode('utf-8'))
        except Exception:
            data = {"error": "HTTP Error"}
        return he.code, data, round(time.time() - t0, 2)
    except Exception as e:
        return 500, {"error": str(e)}, round(time.time() - t0, 2)

def run_audit_tests():
    print("=================================================================")
    print("                 LIVE READ-ONLY AUDIT DIAGNOSTIC                 ")
    print("=================================================================\n")

    # Fetch status
    req = urllib.request.Request(f"{BASE_URL}/api/status")
    with urllib.request.urlopen(req) as r:
        st = json.loads(r.read().decode('utf-8'))
    
    indexed_docs = st.get("indexed_documents", [])
    print(f"System Status: {st.get('status')}")
    print(f"Indexed Documents Count: {len(indexed_docs)}")
    print(f"Indexed Documents: {indexed_docs}\n")

    # Test 1 & 2: Select all documents, ask "Tell me about DBMS"
    print("---------------- TEST 1 & 2: ALL DOCUMENTS SELECTED ----------------")
    p1 = {
        "question": "Tell me about DBMS",
        "selected_documents": indexed_docs,
        "conversation": []
    }
    status1, res1, time1 = post_chat(p1)
    print(f"Frontend Selected Count: {len(indexed_docs)} of {len(indexed_docs)}")
    print(f"Selected Document List: {indexed_docs}")
    print(f"Request Payload: {json.dumps(p1)}")
    print(f"HTTP Status: {status1} ({time1}s)")
    print(f"Backend Documents Used: {res1.get('documents_used', [])}")
    print(f"Chunks Used / Sources: {len(res1.get('sources', []))} chunks from {[s.get('document') for s in res1.get('sources', [])]}")
    print(f"LLM Result / Answer: {res1.get('answer', '')[:160]}...")
    print(f"Final Success: {res1.get('success')}\n")

    # Test 3 & 4: Select only DBMS_Notes.pdf, ask "What is normalization?"
    print("---------------- TEST 3 & 4: SINGLE DOCUMENT (DBMS_Notes.pdf) ----------------")
    p2 = {
        "question": "What is normalization?",
        "selected_documents": ["DBMS_Notes.pdf"],
        "conversation": []
    }
    status2, res2, time2 = post_chat(p2)
    print(f"Frontend Selected Count: 1 of {len(indexed_docs)}")
    print(f"Selected Document List: ['DBMS_Notes.pdf']")
    print(f"Request Payload: {json.dumps(p2)}")
    print(f"HTTP Status: {status2} ({time2}s)")
    print(f"Backend Documents Used: {res2.get('documents_used', [])}")
    print(f"Chunks Used / Sources: {len(res2.get('sources', []))} chunks from {[s.get('document') for s in res2.get('sources', [])]}")
    print(f"LLM Result / Answer: {res2.get('answer', '')[:160]}...")
    print(f"Final Success: {res2.get('success')}\n")

    # Test 5 & 6: Clear selection (empty array), ask "What is normalization?"
    print("---------------- TEST 5 & 6: EMPTY SELECTION (CLEAR SELECTION) ----------------")
    p3 = {
        "question": "What is normalization?",
        "selected_documents": [],
        "conversation": []
    }
    status3, res3, time3 = post_chat(p3)
    print(f"Frontend Selected Count: 0 of {len(indexed_docs)}")
    print(f"Selected Document List: []")
    print(f"Request Payload: {json.dumps(p3)}")
    print(f"HTTP Status: {status3} ({time3}s)")
    print(f"Backend Returned Error: {res3.get('error')}")
    print(f"Final Success: {res3.get('success')}\n")

    # Test 7 & 8: Select DBMS_Notes.pdf again, ask "What is a primary key?"
    print("---------------- TEST 7 & 8: RE-SELECT DBMS_Notes.pdf ----------------")
    p4 = {
        "question": "What is a primary key?",
        "selected_documents": ["DBMS_Notes.pdf"],
        "conversation": []
    }
    status4, res4, time4 = post_chat(p4)
    print(f"Frontend Selected Count: 1 of {len(indexed_docs)}")
    print(f"Selected Document List: ['DBMS_Notes.pdf']")
    print(f"Request Payload: {json.dumps(p4)}")
    print(f"HTTP Status: {status4} ({time4}s)")
    print(f"Backend Documents Used: {res4.get('documents_used', [])}")
    print(f"Chunks Used / Sources: {len(res4.get('sources', []))} chunks from {[s.get('document') for s in res4.get('sources', [])]}")
    print(f"LLM Result / Answer: {res4.get('answer', '')[:160]}...")
    print(f"Final Success: {res4.get('success')}\n")

if __name__ == "__main__":
    run_audit_tests()
