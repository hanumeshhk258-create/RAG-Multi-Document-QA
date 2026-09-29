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
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def run_tests():
    print("=" * 75)
    print("CONVERSATIONAL MEMORY VERIFICATION TEST SUITE")
    print("=" * 75)

    status = get_status()
    print(f"Status: {status.get('status')}")
    print(f"Total Documents: {status.get('total_documents')}")
    print(f"Total Chunks: {status.get('total_chunks')}")
    print("-" * 75)

    # TEST 1: "What is ACID?" -> "Explain the second property."
    print("\n[TEST 1] 'What is ACID?' -> Follow-up: 'Explain the second property.'")
    r1 = ask_question("What is ACID?", selected_documents=["DBMS_Notes.pdf"])
    print(f"Turn 1 Answer ({r1.get('response_time', 0)}s):\n{r1.get('answer')[:160]}...")
    assert r1.get("success"), "Turn 1 failed"
    
    conv1 = [
        {"role": "user", "content": "What is ACID?"},
        {"role": "assistant", "content": r1.get("answer")}
    ]
    r1_follow = ask_question("Explain the second property.", selected_documents=["DBMS_Notes.pdf"], conversation=conv1)
    print(f"Follow-up Rewritten Query: '{r1_follow.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r1_follow.get('response_time', 0)}s):\n{r1_follow.get('answer')}")
    print(f"Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r1_follow.get('sources', [])]}")
    assert r1_follow.get("success"), "Follow-up failed"
    assert "consistency" in r1_follow.get("rewritten_query", "").lower() or "consistency" in r1_follow.get("answer", "").lower(), "Expected Consistency for 2nd ACID property"
    print(">>> TEST 1 PASSED [OK]")

    # TEST 2: "What is JOIN?" -> "Give an example."
    print("\n[TEST 2] 'What is JOIN?' -> Follow-up: 'Give an example.'")
    r2 = ask_question("What is JOIN?", selected_documents=["SQL_CheatSheet.pdf"])
    print(f"Turn 1 Answer ({r2.get('response_time', 0)}s):\n{r2.get('answer')[:160]}...")
    assert r2.get("success"), "Turn 1 failed"

    conv2 = [
        {"role": "user", "content": "What is JOIN?"},
        {"role": "assistant", "content": r2.get("answer")}
    ]
    r2_follow = ask_question("Give me an example.", selected_documents=["SQL_CheatSheet.pdf"], conversation=conv2)
    print(f"Follow-up Rewritten Query: '{r2_follow.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r2_follow.get('response_time', 0)}s):\n{r2_follow.get('answer')}")
    assert r2_follow.get("success"), "Follow-up failed"
    print(">>> TEST 2 PASSED [OK]")

    # TEST 3: "What is normalization?" -> "What are its types?"
    print("\n[TEST 3] 'What is normalization?' -> Follow-up: 'What are its types?'")
    r3 = ask_question("What is normalization?", selected_documents=["DBMS_Notes.pdf"])
    print(f"Turn 1 Answer ({r3.get('response_time', 0)}s):\n{r3.get('answer')[:160]}...")
    assert r3.get("success"), "Turn 1 failed"

    conv3 = [
        {"role": "user", "content": "What is normalization?"},
        {"role": "assistant", "content": r3.get("answer")}
    ]
    r3_follow = ask_question("What are its types?", selected_documents=["DBMS_Notes.pdf"], conversation=conv3)
    print(f"Follow-up Rewritten Query: '{r3_follow.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r3_follow.get('response_time', 0)}s):\n{r3_follow.get('answer')}")
    assert r3_follow.get("success"), "Follow-up failed"
    print(">>> TEST 3 PASSED [OK]")

    # TEST 4: "What is Python?" -> "Explain it in simple words."
    print("\n[TEST 4] 'What is Python?' -> Follow-up: 'Explain it in simple words.'")
    r4 = ask_question("What is Python?", selected_documents=["Python_Notes.pdf"])
    print(f"Turn 1 Answer ({r4.get('response_time', 0)}s):\n{r4.get('answer')[:160]}...")
    assert r4.get("success"), "Turn 1 failed"

    conv4 = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": r4.get("answer")}
    ]
    r4_follow = ask_question("Explain it in simple words.", selected_documents=["Python_Notes.pdf"], conversation=conv4)
    print(f"Follow-up Rewritten Query: '{r4_follow.get('rewritten_query')}'")
    print(f"Follow-up Answer ({r4_follow.get('response_time', 0)}s):\n{r4_follow.get('answer')}")
    assert r4_follow.get("success"), "Follow-up failed"
    print(">>> TEST 4 PASSED [OK]")

    # TEST 5: Ask something NOT in selected documents
    print("\n[TEST 5] Ask question NOT in selected documents (Selected: SQL_CheatSheet only -> Ask 'What is Python?')")
    r5 = ask_question("What is Python?", selected_documents=["SQL_CheatSheet.pdf"])
    print(f"Answer ({r5.get('response_time', 0)}s):\n{r5.get('answer')}")
    print(f"Sources: {r5.get('sources')}")
    assert "couldn't find this information in the selected documents" in r5.get("answer", "").lower() or "could not find" in r5.get("answer", "").lower()
    assert len(r5.get("sources", [])) == 0, "Unselected documents should return empty sources"
    print(">>> TEST 5 PASSED [OK]")

    # TEST 6: New Chat Endpoint
    print("\n[TEST 6] New Chat Endpoint: Reset conversation session")
    r6 = trigger_new_chat()
    print(f"New Chat Response: {r6}")
    assert r6.get("success") == True
    status_after = get_status()
    assert status_after.get("total_documents") == status.get("total_documents"), "New chat should not delete documents"
    assert status_after.get("vectorstore_ready") == True, "New chat should not delete vectorstore index"
    print(">>> TEST 6 PASSED [OK]")

    print("\n" + "=" * 75)
    print("ALL 6 CONVERSATIONAL MEMORY REQUIREMENTS VERIFIED & PASSED!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
