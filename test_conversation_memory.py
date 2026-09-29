import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"

def get_status():
    req = urllib.request.Request(f"{BASE_URL}/api/status")
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def ask_question(question, selected_documents=None, conversation=None):
    payload_dict = {"question": question, "conversation": conversation or []}
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

def clear_chat():
    req = urllib.request.Request(f"{BASE_URL}/api/clear_chat", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def run_tests():
    print("=" * 70)
    print("RUNNING CONVERSATION MEMORY & FOLLOW-UP QUESTION TESTS")
    print("=" * 70)

    # TEST 1: What is JOIN? -> What are its types?
    print("\n[TEST 1] 'What is JOIN?' -> Follow-up: 'What are its types?'")
    r1 = ask_question("What is JOIN?", selected_documents=["SQL_CheatSheet.pdf"])
    print(f"Turn 1 Answer ({r1.get('response_time', 0)}s): {r1.get('answer')[:120]}...")
    assert r1.get("success"), "Turn 1 failed"
    
    conv1 = [
        {"role": "user", "content": "What is JOIN?"},
        {"role": "assistant", "content": r1.get("answer")}
    ]
    r1_followup = ask_question("What are its types?", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv1)
    print(f"Follow-up Rewritten Query: '{r1_followup.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r1_followup.get('response_time', 0)}s):\n{r1_followup.get('answer')}")
    print(f"Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r1_followup.get('sources', [])]}")
    assert r1_followup.get("success"), "Follow-up failed"
    assert len(r1_followup.get("sources", [])) > 0, "Follow-up should have sources"
    print(">>> TEST 1 PASSED [OK]")

    # TEST 2: What is normalization? -> Why is it used?
    print("\n[TEST 2] 'What is normalization?' -> Follow-up: 'Why is it used?'")
    r2 = ask_question("What is normalization?", selected_documents=["DBMS_Notes.pdf"])
    print(f"Turn 1 Answer ({r2.get('response_time', 0)}s): {r2.get('answer')[:120]}...")
    assert r2.get("success"), "Turn 1 failed"

    conv2 = [
        {"role": "user", "content": "What is normalization?"},
        {"role": "assistant", "content": r2.get("answer")}
    ]
    r2_followup = ask_question("Why is it used?", selected_documents=["DBMS_Notes.pdf"], conversation=conv2)
    print(f"Follow-up Rewritten Query: '{r2_followup.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r2_followup.get('response_time', 0)}s):\n{r2_followup.get('answer')}")
    print(f"Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r2_followup.get('sources', [])]}")
    assert r2_followup.get("success"), "Follow-up failed"
    assert any("DBMS" in s["document"] for s in r2_followup.get("sources", [])), "Sources missing DBMS doc"
    print(">>> TEST 2 PASSED [OK]")

    # TEST 3: What is Python? -> What are its advantages?
    print("\n[TEST 3] 'What is Python?' -> Follow-up: 'What are its advantages?'")
    r3 = ask_question("What is Python?", selected_documents=["Python_Notes.pdf"])
    print(f"Turn 1 Answer ({r3.get('response_time', 0)}s): {r3.get('answer')[:120]}...")
    assert r3.get("success"), "Turn 1 failed"

    conv3 = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": r3.get("answer")}
    ]
    r3_followup = ask_question("What are its advantages?", selected_documents=["Python_Notes.pdf"], conversation=conv3)
    print(f"Follow-up Rewritten Query: '{r3_followup.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r3_followup.get('response_time', 0)}s):\n{r3_followup.get('answer')}")
    assert r3_followup.get("success"), "Follow-up failed"
    print(">>> TEST 3 PASSED [OK]")

    # TEST 4: What is SQL? -> Give me an example.
    print("\n[TEST 4] 'What is SQL?' -> Follow-up: 'Give me an example.'")
    r4 = ask_question("What is SQL?", selected_documents=["SQL_CheatSheet.pdf"])
    assert r4.get("success"), "Turn 1 failed"

    conv4 = [
        {"role": "user", "content": "What is SQL?"},
        {"role": "assistant", "content": r4.get("answer")}
    ]
    r4_followup = ask_question("Give me an example.", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv4)
    print(f"Follow-up Rewritten Query: '{r4_followup.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r4_followup.get('response_time', 0)}s):\n{r4_followup.get('answer')}")
    assert r4_followup.get("success"), "Follow-up failed"
    print(">>> TEST 4 PASSED [OK]")

    # TEST 5: What is JOIN? -> Explain the first type.
    print("\n[TEST 5] 'What is JOIN?' -> Follow-up: 'Explain the first type.'")
    r5 = ask_question("What is JOIN?", selected_documents=["SQL_CheatSheet.pdf"])
    assert r5.get("success"), "Turn 1 failed"

    conv5 = [
        {"role": "user", "content": "What is JOIN?"},
        {"role": "assistant", "content": "JOIN types include INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN."}
    ]
    r5_followup = ask_question("Explain the first type.", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv5)
    print(f"Follow-up Rewritten Query: '{r5_followup.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r5_followup.get('response_time', 0)}s):\n{r5_followup.get('answer')}")
    assert r5_followup.get("success"), "Follow-up failed"
    print(">>> TEST 5 PASSED [OK]")

    # TEST 6: Document Filtering with Follow-Up
    print("\n[TEST 6] Selected: SQL only -> Ask 'What is SQL?' -> Follow-up: 'What is Python?'")
    conv6 = [
        {"role": "user", "content": "What is SQL?"},
        {"role": "assistant", "content": "SQL is a database query language."}
    ]
    r6 = ask_question("What is Python?", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv6)
    print(f"Answer: {r6.get('answer')}")
    print(f"Sources: {r6.get('sources')}")
    assert "couldn't find this information in the selected documents" in r6.get("answer").lower() or "could not find" in r6.get("answer").lower()
    assert len(r6.get("sources", [])) == 0
    print(">>> TEST 6 PASSED [OK]")

    # TEST 7: New Topic Detection
    print("\n[TEST 7] Previous: 'What is SQL?' -> New: 'What is a primary key?'")
    conv7 = [
        {"role": "user", "content": "What is SQL?"},
        {"role": "assistant", "content": "SQL is structured query language."}
    ]
    r7 = ask_question("What is a primary key?", selected_documents=["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"], conversation=conv7)
    print(f"Rewritten Query: '{r7.get('rewritten_query')}'")
    print(f"Answer: {r7.get('answer')[:150]}...")
    assert "primary key" in r7.get('rewritten_query').lower(), "New topic was improperly overwritten"
    print(">>> TEST 7 PASSED [OK]")

    # TEST 8: Clear Chat
    print("\n[TEST 8] Clear Chat Endpoint")
    r8 = clear_chat()
    print(f"Clear Chat Response: {r8}")
    assert r8.get("success") == True
    status = get_status()
    assert status.get("total_documents") == 4, "Clear chat should not delete documents"
    assert status.get("vectorstore_ready") == True, "Clear chat should not delete vectorstore"
    print(">>> TEST 8 PASSED [OK]")

    print("\n" + "=" * 70)
    print("ALL 8 CONVERSATION MEMORY TESTS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
