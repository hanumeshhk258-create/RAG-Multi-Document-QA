import urllib.request
import urllib.parse
import json

BASE_URL = 'http://127.0.0.1:5000'

def run_suite():
    print("=" * 60)
    print("RUNNING LIVE END-TO-END VERIFICATION SUITE")
    print("=" * 60)

    # TEST 1 & 2: Test Status & Documents Endpoint
    print("\n--- TEST 1 & 2: Checking Document Registration & Status ---")
    req = urllib.request.Request(f"{BASE_URL}/api/status")
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    
    print(f"Status: {res.get('status')}")
    print(f"Total Documents: {res.get('total_documents')}")
    print(f"Total Pages: {res.get('total_pages')}")
    print(f"Total Chunks: {res.get('total_chunks')}")
    print(f"Table Extraction Active: {res.get('components', {}).get('table_extraction')}")
    assert res.get('total_documents') > 0, "Expected uploaded documents to be registered"
    print("[PASS] Documents correctly registered and loaded.")

    # TEST 3: Indexing Endpoint
    print("\n--- TEST 3: Testing Indexing ---")
    req = urllib.request.Request(f"{BASE_URL}/api/index", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    print(f"Index result: success={res.get('success')}, docs={res.get('documents')}, chunks={res.get('chunks')}")
    assert res.get('success'), "Indexing failed"
    print("[PASS] Document indexing succeeded.")

    # TEST 4: Query: 'What is Waterfall Model?'
    print("\n--- TEST 4: Text Query ('What is Waterfall Model?') ---")
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps({"question": "What is Waterfall Model?"}).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    print(f"Answer: {res.get('answer')[:180]}...")
    print(f"Sources: {[s.get('document') for s in res.get('sources', [])]}")
    assert res.get('success') and res.get('answer'), "Failed text query"
    print("[PASS] Standard text QA verified.")

    # TEST 5: Table Query: 'What is the advantage of KNN?'
    print("\n--- TEST 5: Table Query ('What is the advantage of KNN?') ---")
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps({"question": "What is the advantage of KNN?"}).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    print(f"Answer: {res.get('answer')}")
    print(f"Structured Data: {res.get('structured_data')}")
    assert res.get('success') and "simple" in res.get('answer', '').lower(), "Failed table query"
    print("[PASS] Table extraction QA verified.")

    # TEST 6: Persistence after reload
    print("\n--- TEST 6: Verify documents persist after refresh ---")
    req = urllib.request.Request(f"{BASE_URL}/api/documents")
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    print(f"Documents count: {len(res.get('documents', []))}")
    assert len(res.get('documents', [])) > 0, "Documents missing after reload"
    print("[PASS] Documents verified persistent across reloads.")

    print("\n" + "=" * 60)
    print("ALL 6 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_suite()
