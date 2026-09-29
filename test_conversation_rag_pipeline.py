import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"

def get_status():
    req = urllib.request.Request(f"{BASE_URL}/api/status")
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode())

def ask_question(question, selected_documents=None, conversation=None, last_context=None):
    payload_dict = {
        "question": question,
        "conversation": conversation or []
    }
    if selected_documents is not None:
        payload_dict["selected_documents"] = selected_documents
    if last_context is not None:
        payload_dict["last_context"] = last_context
    
    payload = json.dumps(payload_dict).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
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
    print("=" * 80)
    print("CONVERSATION-AWARE RAG PIPELINE VERIFICATION TEST SUITE (TESTS 1 - 10)")
    print("=" * 80)

    status = get_status()
    print(f"Status: {status.get('status')}")
    print(f"Total Documents: {status.get('total_documents')}")
    print(f"Total Chunks: {status.get('total_chunks')}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: "What are ACID properties?"
    # -------------------------------------------------------------------------
    print("\n[TEST 1] 'What are ACID properties?'")
    conv = []
    r1 = ask_question("What are ACID properties?", selected_documents=["DBMS_Notes.pdf"], conversation=conv)
    ans1 = r1.get("answer", "")
    print(f"Turn 1 Answer ({r1.get('response_time', 0)}s):\n{ans1[:160]}...")
    assert "Atomicity" in ans1 or "Consistency" in ans1 or "Isolation" in ans1, f"Test 1 failed: {ans1}"
    print(">>> TEST 1 PASSED [OK]")

    conv.append({"role": "user", "content": "What are ACID properties?"})
    conv.append({"role": "assistant", "content": ans1, "retrieved_context": r1.get("retrieved_context", [])})

    # -------------------------------------------------------------------------
    # TEST 2: "What is the third property?" -> "Isolation"
    # -------------------------------------------------------------------------
    print("\n[TEST 2] 'What is the third property?'")
    r2 = ask_question("What is the third property?", selected_documents=["DBMS_Notes.pdf"], conversation=conv)
    ans2 = r2.get("answer", "")
    print(f"Rewritten Query: '{r2.get('rewritten_query')}'")
    print(f"Answer ({r2.get('response_time', 0)}s):\n{ans2[:160]}...")
    assert "isolation" in ans2.lower(), f"Test 2 failed to identify Isolation: {ans2}"
    print(">>> TEST 2 PASSED (Identified Isolation) [OK]")

    conv.append({"role": "user", "content": "What is the third property?"})
    conv.append({"role": "assistant", "content": ans2, "retrieved_context": r2.get("retrieved_context", [])})

    # -------------------------------------------------------------------------
    # TEST 3: "What is the last property?" -> "Durability"
    # -------------------------------------------------------------------------
    print("\n[TEST 3] 'What is the last property?'")
    r3 = ask_question("What is the last property?", selected_documents=["DBMS_Notes.pdf"], conversation=conv)
    ans3 = r3.get("answer", "")
    print(f"Rewritten Query: '{r3.get('rewritten_query')}'")
    print(f"Answer ({r3.get('response_time', 0)}s):\n{ans3[:160]}...")
    assert "durability" in ans3.lower(), f"Test 3 failed to identify Durability: {ans3}"
    print(">>> TEST 3 PASSED (Identified Durability) [OK]")

    conv.append({"role": "user", "content": "What is the last property?"})
    conv.append({"role": "assistant", "content": ans3, "retrieved_context": r3.get("retrieved_context", [])})

    # -------------------------------------------------------------------------
    # TEST 4: "Explain it." -> Explanation of the property from previous context
    # -------------------------------------------------------------------------
    print("\n[TEST 4] 'Explain it.'")
    r4 = ask_question("Explain it.", selected_documents=["DBMS_Notes.pdf"], conversation=conv)
    ans4 = r4.get("answer", "")
    print(f"Rewritten Query: '{r4.get('rewritten_query')}'")
    print(f"Answer ({r4.get('response_time', 0)}s):\n{ans4[:160]}...")
    assert ans4 != "I couldn't find this information in the selected documents.", f"Test 4 failed: {ans4}"
    print(">>> TEST 4 PASSED [OK]")

    conv.append({"role": "user", "content": "Explain it."})
    conv.append({"role": "assistant", "content": ans4, "retrieved_context": r4.get("retrieved_context", [])})

    # -------------------------------------------------------------------------
    # TEST 5: "Give an example." -> Example of the previous topic
    # -------------------------------------------------------------------------
    print("\n[TEST 5] 'Give an example.'")
    r5 = ask_question("Give an example.", selected_documents=["DBMS_Notes.pdf"], conversation=conv)
    ans5 = r5.get("answer", "")
    print(f"Rewritten Query: '{r5.get('rewritten_query')}'")
    print(f"Answer ({r5.get('response_time', 0)}s):\n{ans5[:160]}...")
    assert ans5 != "I couldn't find this information in the selected documents.", f"Test 5 failed: {ans5}"
    print(">>> TEST 5 PASSED [OK]")

    # -------------------------------------------------------------------------
    # TEST 6: "What is SQL?" -> SQL definition from uploaded PDF
    # -------------------------------------------------------------------------
    print("\n[TEST 6] 'What is SQL?'")
    r6 = ask_question("What is SQL?", selected_documents=["SQL_CheatSheet.pdf"], conversation=[])
    ans6 = r6.get("answer", "")
    print(f"Answer ({r6.get('response_time', 0)}s):\n{ans6[:160]}...")
    assert ans6 != "I couldn't find this information in the selected documents.", f"Test 6 failed: {ans6}"
    print(">>> TEST 6 PASSED [OK]")

    # -------------------------------------------------------------------------
    # TEST 7: "What is JOIN?" -> JOIN definition from uploaded PDF
    # -------------------------------------------------------------------------
    print("\n[TEST 7] 'What is JOIN?'")
    conv_join = []
    r7 = ask_question("What is JOIN?", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv_join)
    ans7 = r7.get("answer", "")
    print(f"Answer ({r7.get('response_time', 0)}s):\n{ans7[:160]}...")
    assert "join" in ans7.lower() or "table" in ans7.lower(), f"Test 7 failed: {ans7}"
    print(">>> TEST 7 PASSED [OK]")

    conv_join.append({"role": "user", "content": "What is JOIN?"})
    conv_join.append({"role": "assistant", "content": ans7, "retrieved_context": r7.get("retrieved_context", [])})

    # -------------------------------------------------------------------------
    # TEST 8: "What are the types of JOIN?" -> Types from SQL document
    # -------------------------------------------------------------------------
    print("\n[TEST 8] 'What are the types of JOIN?'")
    r8 = ask_question("What are the types of JOIN?", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv_join)
    ans8 = r8.get("answer", "")
    print(f"Rewritten Query: '{r8.get('rewritten_query')}'")
    print(f"Answer ({r8.get('response_time', 0)}s):\n{ans8[:160]}...")
    assert "inner join" in ans8.lower() or "left join" in ans8.lower(), f"Test 8 failed: {ans8}"
    print(">>> TEST 8 PASSED [OK]")

    conv_join.append({"role": "user", "content": "What are the types of JOIN?"})
    conv_join.append({"role": "assistant", "content": ans8, "retrieved_context": r8.get("retrieved_context", [])})

    # -------------------------------------------------------------------------
    # TEST 9: "Explain the second type." -> Explanation of the second JOIN type (LEFT JOIN)
    # -------------------------------------------------------------------------
    print("\n[TEST 9] 'Explain the second type.'")
    r9 = ask_question("Explain the second type.", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv_join)
    ans9 = r9.get("answer", "")
    print(f"Rewritten Query: '{r9.get('rewritten_query')}'")
    print(f"Answer ({r9.get('response_time', 0)}s):\n{ans9[:160]}...")
    assert "left" in ans9.lower() or "join" in ans9.lower(), f"Test 9 failed: {ans9}"
    print(">>> TEST 9 PASSED (Explained Second Type) [OK]")

    # -------------------------------------------------------------------------
    # TEST 10: Multi-PDF Filtering
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Multi-PDF selection: select only Python_Notes.pdf and ask 'What is a generator in Python?'")
    r10 = ask_question("What is a generator in Python?", selected_documents=["Python_Notes.pdf"], conversation=[])
    ans10 = r10.get("answer", "")
    sources10 = r10.get("sources", [])
    print(f"Answer ({r10.get('response_time', 0)}s):\n{ans10[:160]}...")
    print(f"Sources: {[s['document'] for s in sources10]}")
    assert sources10, f"Test 10 failed: No sources returned for Python_Notes.pdf"
    for s in sources10:
        assert s["document"] == "Python_Notes.pdf", f"Test 10 failed: Retrieved unselected document {s['document']}"
    assert "generator" in ans10.lower() or "iterator" in ans10.lower() or "yield" in ans10.lower(), f"Test 10 failed: {ans10}"
    print(">>> TEST 10 PASSED (Strict Multi-PDF Filtering) [OK]")

    print("\n" + "=" * 80)
    print("ALL 10 TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()
