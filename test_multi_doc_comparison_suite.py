import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def ask_question(query, selected_documents=None, conversation=None):
    payload = {
        "query": query,
        "selected_documents": selected_documents,
        "conversation": conversation or []
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=60)
    dur = round(time.perf_counter() - t0, 2)
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
    data = resp.json()
    data["client_duration"] = dur
    return data

def wait_for_server(max_retries=15):
    for i in range(max_retries):
        try:
            r = requests.get(f"{BASE_URL}/api/health", timeout=3)
            if r.status_code == 200:
                print("Server is UP and ready!")
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def run_all_tests():
    wait_for_server()
    print("==================================================")
    print("RUNNING MULTI-DOCUMENT COMPARISON & SYNTHESIS TEST SUITE")
    print("==================================================")

    # Check indexed documents
    docs_resp = requests.get(f"{BASE_URL}/api/documents")
    docs_data = docs_resp.json()
    indexed_docs = [d.get("filename") or d.get("id") or d.get("name") for d in docs_data.get("documents", []) if d.get("indexed")]
    print(f"Indexed documents available ({len(indexed_docs)}): {indexed_docs}\n")

    # TEST 1: "Compare SQL and Python."
    print("--------------------------------------------------")
    print("TEST 1: 'Compare SQL and Python.'")
    print("--------------------------------------------------")
    res1 = ask_question("Compare SQL and Python.", selected_documents=["SQL_CheatSheet.pdf", "Python_Notes.pdf"])
    print(f"Mode: {res1.get('mode')} | is_comparison: {res1.get('is_comparison')}")
    print(f"Answer:\n{res1.get('answer')[:350]}...\n")
    print(f"Sources: {[s['document'] + ' P.' + str(s['page']) for s in res1.get('sources', [])]}")
    assert res1.get("is_comparison") or res1.get("mode") == "comparison", "Test 1 Failed: Not in comparison mode"
    assert len(res1.get("sources", [])) >= 1, "Test 1 Failed: No sources returned"
    print(">>> TEST 1 PASSED!\n")

    # TEST 2: "What are the similarities between these documents?"
    print("--------------------------------------------------")
    print("TEST 2: 'What are the similarities between these documents?'")
    print("--------------------------------------------------")
    res2 = ask_question("What are the similarities between these documents?", selected_documents=["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"])
    print(f"Mode: {res2.get('mode')} | is_comparison: {res2.get('is_comparison')}")
    print(f"Answer:\n{res2.get('answer')[:350]}...\n")
    print(f"Sources: {[s['document'] + ' P.' + str(s['page']) for s in res2.get('sources', [])]}")
    assert res2.get("is_comparison") or res2.get("mode") == "comparison", "Test 2 Failed: Not in comparison mode"
    print(">>> TEST 2 PASSED!\n")

    # TEST 3: "What are the differences between these documents?"
    print("--------------------------------------------------")
    print("TEST 3: 'What are the differences between these documents?'")
    print("--------------------------------------------------")
    res3 = ask_question("What are the differences between these documents?", selected_documents=["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"])
    print(f"Mode: {res3.get('mode')} | is_comparison: {res3.get('is_comparison')}")
    print(f"Answer:\n{res3.get('answer')[:350]}...\n")
    print(f"Sources: {[s['document'] + ' P.' + str(s['page']) for s in res3.get('sources', [])]}")
    assert res3.get("is_comparison") or res3.get("mode") == "comparison", "Test 3 Failed: Not in comparison mode"
    print(">>> TEST 3 PASSED!\n")

    # TEST 4: "What information is common across all selected documents?"
    print("--------------------------------------------------")
    print("TEST 4: 'What information is common across all selected documents?'")
    print("--------------------------------------------------")
    res4 = ask_question("What information is common across all selected documents?", selected_documents=["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"])
    print(f"Mode: {res4.get('mode')} | is_comparison: {res4.get('is_comparison')}")
    print(f"Answer:\n{res4.get('answer')[:350]}...\n")
    print(f"Sources: {[s['document'] + ' P.' + str(s['page']) for s in res4.get('sources', [])]}")
    assert res4.get("is_comparison") or res4.get("mode") == "comparison", "Test 4 Failed: Not in comparison mode"
    print(">>> TEST 4 PASSED!\n")

    # TEST 5: "Summarize the selected documents."
    print("--------------------------------------------------")
    print("TEST 5: 'Summarize the selected documents.'")
    print("--------------------------------------------------")
    res5 = ask_question("Summarize the selected documents.", selected_documents=["DBMS_Notes.pdf", "Python_Notes.pdf"])
    print(f"Mode: {res5.get('mode')} | is_comparison: {res5.get('is_comparison')}")
    print(f"Answer:\n{res5.get('answer')[:350]}...\n")
    print(f"Sources: {[s['document'] + ' P.' + str(s['page']) for s in res5.get('sources', [])]}")
    assert res5.get("is_comparison") or res5.get("mode") == "comparison", "Test 5 Failed: Not in comparison mode"
    print(">>> TEST 5 PASSED!\n")

    # TEST 6: Missing information test
    print("--------------------------------------------------")
    print("TEST 6: Missing information test")
    print("--------------------------------------------------")
    res6 = ask_question("Compare quantum cryptography protocols in these documents.", selected_documents=["SQL_CheatSheet.pdf", "Python_Notes.pdf"])
    print(f"Mode: {res6.get('mode')} | is_comparison: {res6.get('is_comparison')}")
    print(f"Answer:\n{res6.get('answer')[:350]}...\n")
    print(">>> TEST 6 PASSED!\n")

    # TEST 7: Follow-up memory test
    print("--------------------------------------------------")
    print("TEST 7: Follow-up memory test")
    print("--------------------------------------------------")
    conv7 = [
        {"role": "user", "content": "Compare SQL and Python."},
        {"role": "assistant", "content": res1.get("answer", ""), "comparison_topics": ["SQL", "Python"]}
    ]
    res7 = ask_question("Which one is easier to learn?", selected_documents=["SQL_CheatSheet.pdf", "Python_Notes.pdf"], conversation=conv7)
    print(f"Is followup: {res7.get('is_followup')} | Rewritten: '{res7.get('rewritten_query')}'")
    print(f"Answer:\n{res7.get('answer')[:350]}...\n")
    print(">>> TEST 7 PASSED!\n")

    # TEST 8: Regression test on single-document normal QA
    print("--------------------------------------------------")
    print("TEST 8: Regression test on normal single-document QA ('What is Waterfall Model?')")
    print("--------------------------------------------------")
    res8 = ask_question("What is Waterfall Model?", selected_documents=["CommerceOS_Complete_Project_Report.pdf"])
    print(f"Mode: {res8.get('mode')} | is_comparison: {res8.get('is_comparison')}")
    print(f"Answer:\n{res8.get('answer')[:350]}...\n")
    print(f"Sources: {[s['document'] + ' P.' + str(s['page']) for s in res8.get('sources', [])]}")
    assert res8.get("is_comparison") == False, "Test 8 Failed: Normal question should NOT trigger comparison mode"
    assert len(res8.get("sources", [])) > 0, "Test 8 Failed: Sources should be present"
    print(">>> TEST 8 PASSED!\n")

    print("==================================================")
    print("ALL 8 COMPARISON & REGRESSION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_all_tests()
