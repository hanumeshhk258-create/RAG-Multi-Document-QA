import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"

def get_status():
    req = urllib.request.Request(f"{BASE_URL}/api/status")
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode())

def ask_question(question, selected_documents=None, conversation=None):
    payload_dict = {
        "question": question,
        "conversation": conversation or []
    }
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

def trigger_new_chat():
    req = urllib.request.Request(f"{BASE_URL}/api/new_chat", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode())

def run_tests():
    print("=" * 75)
    print("RAG RETRIEVAL & CONVERSATIONAL MEMORY VERIFICATION TEST SUITE")
    print("=" * 75)
    
    status = get_status()
    print(f"Status: {status.get('status')}")
    print(f"Total Documents: {status.get('total_documents')}")
    print(f"Total Chunks: {status.get('total_chunks')}")
    print("-" * 75)

    # ----------------------------------------------------
    # TEST 1: "What are ACID properties?" -> "Give an example of it."
    # ----------------------------------------------------
    print("\n[TEST 1] 'What are ACID properties?' -> Follow-up: 'Give an example of it.'")
    conv1 = []
    r1 = ask_question("What are ACID properties?", selected_documents=["DBMS_Notes.pdf"], conversation=conv1)
    ans1 = r1.get("answer", "")
    print(f"Turn 1 Answer ({r1.get('response_time', 0)}s):\n{ans1[:140]}...")
    assert "Atomicity" in ans1 or "Consistency" in ans1 or "ACID" in ans1, f"Turn 1 failed: {ans1}"
    
    conv1.append({"role": "user", "content": "What are ACID properties?"})
    conv1.append({"role": "assistant", "content": ans1})
    
    r1_followup = ask_question("Give an example of it.", selected_documents=["DBMS_Notes.pdf"], conversation=conv1)
    ans1_f = r1_followup.get("answer", "")
    print(f"Follow-up Rewritten Query: '{r1_followup.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r1_followup.get('response_time', 0)}s):\n{ans1_f[:180]}...")
    print(f"Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r1_followup.get('sources', [])]}")
    assert ans1_f != "I couldn't find this information in the selected documents.", f"Test 1 follow-up returned not found: {ans1_f}"
    print(">>> TEST 1 PASSED [OK]")

    # ----------------------------------------------------
    # TEST 2: "What are ACID properties?" -> "What is the last property?"
    # ----------------------------------------------------
    print("\n[TEST 2] 'What are ACID properties?' -> Follow-up: 'What is the last property?'")
    conv2 = [
        {"role": "user", "content": "What are ACID properties?"},
        {"role": "assistant", "content": ans1}
    ]
    r2 = ask_question("What is the last property?", selected_documents=["DBMS_Notes.pdf"], conversation=conv2)
    ans2 = r2.get("answer", "")
    print(f"Follow-up Rewritten Query: '{r2.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r2.get('response_time', 0)}s):\n{ans2[:180]}...")
    print(f"Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r2.get('sources', [])]}")
    assert "Durability" in ans2 or "durability" in ans2.lower(), f"Test 2 failed to mention Durability: {ans2}"
    print(">>> TEST 2 PASSED [OK]")

    # ----------------------------------------------------
    # TEST 3: "What is JOIN?" -> "Give an example."
    # ----------------------------------------------------
    print("\n[TEST 3] 'What is JOIN?' -> Follow-up: 'Give an example.'")
    conv3 = []
    r3_1 = ask_question("What is JOIN?", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv3)
    ans3_1 = r3_1.get("answer", "")
    print(f"Turn 1 Answer ({r3_1.get('response_time', 0)}s):\n{ans3_1[:140]}...")
    
    conv3.append({"role": "user", "content": "What is JOIN?"})
    conv3.append({"role": "assistant", "content": ans3_1})
    
    r3_2 = ask_question("Give an example.", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv3)
    ans3_2 = r3_2.get("answer", "")
    print(f"Follow-up Rewritten Query: '{r3_2.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r3_2.get('response_time', 0)}s):\n{ans3_2[:180]}...")
    print(f"Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r3_2.get('sources', [])]}")
    assert "SELECT" in ans3_2 or "JOIN" in ans3_2 or "employees" in ans3_2, f"Test 3 failed: {ans3_2}"
    print(">>> TEST 3 PASSED [OK]")

    # ----------------------------------------------------
    # TEST 4: "What is normalization?" -> "What are its types?"
    # ----------------------------------------------------
    print("\n[TEST 4] 'What is normalization?' -> Follow-up: 'What are its types?'")
    conv4 = []
    r4_1 = ask_question("What is normalization?", selected_documents=["DBMS_Notes.pdf"], conversation=conv4)
    ans4_1 = r4_1.get("answer", "")
    print(f"Turn 1 Answer ({r4_1.get('response_time', 0)}s):\n{ans4_1[:140]}...")
    
    conv4.append({"role": "user", "content": "What is normalization?"})
    conv4.append({"role": "assistant", "content": ans4_1})
    
    r4_2 = ask_question("What are its types?", selected_documents=["DBMS_Notes.pdf"], conversation=conv4)
    ans4_2 = r4_2.get("answer", "")
    print(f"Follow-up Rewritten Query: '{r4_2.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r4_2.get('response_time', 0)}s):\n{ans4_2[:180]}...")
    print(f"Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r4_2.get('sources', [])]}")
    assert "1NF" in ans4_2 or "First Normal Form" in ans4_2 or "2NF" in ans4_2, f"Test 4 failed: {ans4_2}"
    print(">>> TEST 4 PASSED [OK]")

    # ----------------------------------------------------
    # TEST 5: "What is SQL?" -> "Explain it in simple words."
    # ----------------------------------------------------
    print("\n[TEST 5] 'What is SQL?' -> Follow-up: 'Explain it in simple words.'")
    conv5 = []
    r5_1 = ask_question("What is SQL?", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv5)
    ans5_1 = r5_1.get("answer", "")
    print(f"Turn 1 Answer ({r5_1.get('response_time', 0)}s):\n{ans5_1[:140]}...")
    
    conv5.append({"role": "user", "content": "What is SQL?"})
    conv5.append({"role": "assistant", "content": ans5_1})
    
    r5_2 = ask_question("Explain it in simple words.", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv5)
    ans5_2 = r5_2.get("answer", "")
    print(f"Follow-up Rewritten Query: '{r5_2.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r5_2.get('response_time', 0)}s):\n{ans5_2[:180]}...")
    print(f"Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r5_2.get('sources', [])]}")
    assert "database" in ans5_2.lower() or "sql" in ans5_2.lower(), f"Test 5 failed: {ans5_2}"
    print(">>> TEST 5 PASSED [OK]")

    # ----------------------------------------------------
    # TEST 6: "What is Atomicity?" -> "Give an example."
    # ----------------------------------------------------
    print("\n[TEST 6] 'What is Atomicity?' -> Follow-up: 'Give an example.'")
    conv6 = []
    r6_1 = ask_question("What is Atomicity?", selected_documents=["DBMS_Notes.pdf"], conversation=conv6)
    ans6_1 = r6_1.get("answer", "")
    print(f"Turn 1 Answer ({r6_1.get('response_time', 0)}s):\n{ans6_1[:140]}...")
    
    conv6.append({"role": "user", "content": "What is Atomicity?"})
    conv6.append({"role": "assistant", "content": ans6_1})
    
    r6_2 = ask_question("Give an example.", selected_documents=["DBMS_Notes.pdf"], conversation=conv6)
    ans6_2 = r6_2.get("answer", "")
    print(f"Follow-up Rewritten Query: '{r6_2.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r6_2.get('response_time', 0)}s):\n{ans6_2[:180]}...")
    print(f"Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r6_2.get('sources', [])]}")
    assert ans6_2 != "I couldn't find this information in the selected documents.", f"Test 6 failed: {ans6_2}"
    print(">>> TEST 6 PASSED [OK]")

    # ----------------------------------------------------
    # TEST 7: Unrelated Question Not in Document
    # ----------------------------------------------------
    print("\n[TEST 7] Unrelated Question Not in Document ('What is Quantum Superposition?' in SQL_CheatSheet.pdf)")
    r7 = ask_question("What is Quantum Superposition?", selected_documents=["SQL_CheatSheet.pdf"], conversation=[])
    ans7 = r7.get("answer", "")
    print(f"Answer ({r7.get('response_time', 0)}s):\n{ans7}")
    assert "couldn't find" in ans7.lower() or "could not find" in ans7.lower(), f"Test 7 failed: {ans7}"
    print(">>> TEST 7 PASSED [OK]")

    # ----------------------------------------------------
    # STEP 20: 4-QUESTION CONVERSATION SEQUENCE ON DBMS_Notes.pdf
    # ----------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 20: 4-QUESTION CONVERSATION SEQUENCE ON DBMS_Notes.pdf")
    print("=" * 75)
    seq_conv = []
    
    # 1. What are ACID properties?
    print("\n1. 'What are ACID properties?'")
    s1 = ask_question("What are ACID properties?", selected_documents=["DBMS_Notes.pdf"], conversation=seq_conv)
    s1_ans = s1.get("answer", "")
    print(f"Answer ({s1.get('response_time', 0)}s):\n{s1_ans[:140]}...")
    seq_conv.append({"role": "user", "content": "What are ACID properties?"})
    seq_conv.append({"role": "assistant", "content": s1_ans})
    assert "Atomicity" in s1_ans or "ACID" in s1_ans

    # 2. Give an example of it.
    print("\n2. 'Give an example of it.'")
    s2 = ask_question("Give an example of it.", selected_documents=["DBMS_Notes.pdf"], conversation=seq_conv)
    s2_ans = s2.get("answer", "")
    print(f"Rewritten: '{s2.get('rewritten_query')}' | Answer ({s2.get('response_time', 0)}s):\n{s2_ans[:140]}...")
    seq_conv.append({"role": "user", "content": "Give an example of it."})
    seq_conv.append({"role": "assistant", "content": s2_ans})
    assert s2_ans != "I couldn't find this information in the selected documents."

    # 3. What about the last property?
    print("\n3. 'What about the last property?'")
    s3 = ask_question("What about the last property?", selected_documents=["DBMS_Notes.pdf"], conversation=seq_conv)
    s3_ans = s3.get("answer", "")
    print(f"Rewritten: '{s3.get('rewritten_query')}' | Answer ({s3.get('response_time', 0)}s):\n{s3_ans[:140]}...")
    seq_conv.append({"role": "user", "content": "What about the last property?"})
    seq_conv.append({"role": "assistant", "content": s3_ans})
    assert "Durability" in s3_ans or "durability" in s3_ans.lower()

    # 4. Explain it simply.
    print("\n4. 'Explain it simply.'")
    s4 = ask_question("Explain it simply.", selected_documents=["DBMS_Notes.pdf"], conversation=seq_conv)
    s4_ans = s4.get("answer", "")
    print(f"Rewritten: '{s4.get('rewritten_query')}' | Answer ({s4.get('response_time', 0)}s):\n{s4_ans[:140]}...")
    assert s4_ans != "I couldn't find this information in the selected documents."
    print(">>> STEP 20 ALL 4 QUESTIONS MAINTAINED CONTEXT [OK]")

    print("\n" + "=" * 75)
    print("ALL TESTS AND STEP 20 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
