import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"

def get_status():
    req = urllib.request.Request(f"{BASE_URL}/api/status")
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def get_documents():
    req = urllib.request.Request(f"{BASE_URL}/api/documents")
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def ask_question(question, selected_documents=None, history=None):
    payload_dict = {"question": question, "history": history or []}
    if selected_documents is not None:
        payload_dict["selected_documents"] = selected_documents
    
    payload = json.dumps(payload_dict).encode('utf-8')
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
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            data = json.loads(body)
            data["http_code"] = e.code
            return data
        except Exception:
            return {"success": False, "error": body, "http_code": e.code}
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_all_tests():
    print("=" * 70)
    print("RUNNING DOCUMENT SELECTION & SMART FILTERING TESTS")
    print("=" * 70)

    status = get_status()
    print(f"System status: {status.get('status')}")
    print(f"Total documents indexed: {status.get('total_documents')}")
    print(f"Total chunks: {status.get('total_chunks')}")
    print(f"Documents: {[d['filename'] for d in status.get('documents', [])]}")
    print("-" * 70)

    # TEST 1: Select only SQL_CheatSheet.pdf -> Ask "What is SQL?"
    print("\n[TEST 1] Select only 'SQL_CheatSheet.pdf' -> Ask 'What is SQL?'")
    res1 = ask_question("What is SQL?", selected_documents=["SQL_CheatSheet.pdf"])
    print(f"Response ({res1.get('response_time', 0)}s):")
    print(f"Answer: {res1.get('answer')[:200]}...")
    print(f"Sources: {[s['document'] for s in res1.get('sources', [])]}")
    assert res1.get("success"), "Test 1 failed: not successful"
    assert any("SQL" in s["document"] for s in res1.get("sources", [])), "Test 1 failed: source missing SQL doc"
    assert all(s["document"] == "SQL_CheatSheet.pdf" for s in res1.get("sources", [])), "Test 1 failed: unselected source included"
    print(">>> TEST 1 PASSED [OK]")

    # TEST 2: Select only SQL_CheatSheet.pdf -> Ask "What is Python?"
    print("\n[TEST 2] Select only 'SQL_CheatSheet.pdf' -> Ask 'What is Python?'")
    res2 = ask_question("What is Python?", selected_documents=["SQL_CheatSheet.pdf"])
    print(f"Response ({res2.get('response_time', 0)}s):")
    print(f"Answer: {res2.get('answer')}")
    print(f"Sources: {res2.get('sources', [])}")
    assert res2.get("success"), "Test 2 failed: not successful"
    assert "couldn't find this information in the selected documents" in res2.get("answer").lower() or "could not find" in res2.get("answer").lower(), "Test 2 failed: hallucinated Python when Python was not selected"
    assert len(res2.get("sources", [])) == 0, "Test 2 failed: sources should be empty"
    print(">>> TEST 2 PASSED [OK]")

    # TEST 3: Select SQL_CheatSheet.pdf + Python_Notes.pdf -> Ask "Compare SQL and Python."
    print("\n[TEST 3] Select SQL + Python -> Ask 'Compare SQL and Python.'")
    res3 = ask_question("Compare SQL and Python.", selected_documents=["SQL_CheatSheet.pdf", "Python_Notes.pdf"])
    print(f"Response ({res3.get('response_time', 0)}s):")
    print(f"Answer: {res3.get('answer')[:250]}...")
    src_docs3 = set(s['document'] for s in res3.get('sources', []))
    print(f"Sources: {list(src_docs3)}")
    assert res3.get("success"), "Test 3 failed: not successful"
    print(">>> TEST 3 PASSED [OK]")

    # TEST 4: Ask one-word "JOIN" with SQL_CheatSheet.pdf
    print("\n[TEST 4] Ask one-word query: 'JOIN' (SQL selected)")
    res4 = ask_question("JOIN", selected_documents=["SQL_CheatSheet.pdf"])
    print(f"Query Type: {res4.get('query_type')}")
    print(f"Answer:\n{res4.get('answer')}")
    print(f"Sources: {[s['document'] + ' Page ' + str(s['page']) for s in res4.get('sources', [])]}")
    assert res4.get("success"), "Test 4 failed"
    assert res4.get("query_type") == "ONE_WORD", f"Test 4 expected ONE_WORD, got {res4.get('query_type')}"
    assert len(res4.get("sources", [])) > 0, "Test 4 sources empty"
    print(">>> TEST 4 PASSED [OK]")

    # TEST 5: Ask "Primary Key" with DBMS_Notes.pdf selected
    print("\n[TEST 5] Ask 'Primary Key' (DBMS_Notes.pdf selected)")
    res5 = ask_question("Primary Key", selected_documents=["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"])
    print(f"Query Type: {res5.get('query_type')}")
    print(f"Answer:\n{res5.get('answer')}")
    print(f"Sources: {[s['document'] + ' Page ' + str(s['page']) for s in res5.get('sources', [])]}")
    assert res5.get("success"), "Test 5 failed"
    print(">>> TEST 5 PASSED [OK]")

    # TEST 6: Upload 4 PDFs -> Select 2 (DBMS & Python) -> Verify other 2 (SQL & INS) are NEVER returned
    print("\n[TEST 6] Select DBMS_Notes.pdf & Python_Notes.pdf -> Ask 'What is SQL?'")
    res6 = ask_question("What is SQL?", selected_documents=["DBMS_Notes.pdf", "Python_Notes.pdf"])
    print(f"Answer: {res6.get('answer')}")
    print(f"Sources: {[s['document'] for s in res6.get('sources', [])]}")
    # Verify no chunks from SQL_CheatSheet or INS are returned
    for s in res6.get("sources", []):
        assert s["document"] in ["DBMS_Notes.pdf", "Python_Notes.pdf"], f"Test 6 failed: unselected document {s['document']} returned"
    print(">>> TEST 6 PASSED [OK]")

    # TEST 7: Deselect all (empty selected_documents list) -> Ask a question
    print("\n[TEST 7] Deselect all -> Ask 'What is SQL?'")
    res7 = ask_question("What is SQL?", selected_documents=[])
    print(f"HTTP response: {res7}")
    assert res7.get("success") == False, "Test 7 failed: should return success=False when no document selected"
    assert "select at least one document" in res7.get("error", "").lower(), f"Test 7 error message incorrect: {res7.get('error')}"
    print(">>> TEST 7 PASSED [OK]")

    print("\n" + "=" * 70)
    print("ALL 7 TEST CASES PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()
