import urllib.request
import urllib.error
import json
import time

base_url = "http://127.0.0.1:5000"

def wait_for_server():
    print("Waiting for server to be ready...")
    for _ in range(15):
        try:
            with urllib.request.urlopen(f"{base_url}/api/status") as resp:
                if resp.status == 200:
                    print("Server is ready!")
                    return True
        except Exception:
            time.sleep(1)
    return False

def post_chat(question, conversation=None, last_context=None, selected_docs=None):
    payload = {
        "question": question,
        "conversation": conversation or [],
        "last_context": last_context,
        "selected_documents": selected_docs or ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf", "Python_Notes.pdf"]
    }
    req_data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=req_data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

if not wait_for_server():
    print("Server failed to respond.")
    exit(1)

print("\n==================================================")
print("TEST 1: ACID -> What is the third property?")
print("==================================================")
# Turn 1
r1 = post_chat("What is ACID?")
print(f"Turn 1 Answer snippet: {r1.get('answer', '')[:120]}...")
print(f"Sources: {[s['document'] for s in r1.get('sources', [])]}")

history_1 = [
    {"role": "user", "content": "What is ACID?"},
    {"role": "assistant", "content": r1.get('answer', '')}
]
last_ctx_1 = r1.get("retrieved_context")

# Turn 2
r2 = post_chat("What is the third property?", conversation=history_1, last_context=last_ctx_1)
print(f"Turn 2 Rewritten Query: {r2.get('rewritten_query')}")
print(f"Turn 2 is_followup: {r2.get('is_followup')}")
print(f"Turn 2 Answer: {r2.get('answer')[:150]}...")
print(f"Turn 2 Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r2.get('sources', [])]}")
assert "isolation" in r2.get('answer', '').lower(), "Expected Isolation in answer"
assert r2.get('is_followup') is True, "Expected is_followup=True"
print("[PASS] Test 1 Passed: Correctly identified Isolation as third property with document citations!")

print("\n==================================================")
print("TEST 2: JOIN -> Give an example.")
print("==================================================")
# Turn 1
r2_1 = post_chat("What is JOIN?")
print(f"Turn 1 Answer snippet: {r2_1.get('answer', '')[:120]}...")

history_2 = [
    {"role": "user", "content": "What is JOIN?"},
    {"role": "assistant", "content": r2_1.get('answer', '')}
]
last_ctx_2 = r2_1.get("retrieved_context")

# Turn 2
r2_2 = post_chat("Give an example.", conversation=history_2, last_context=last_ctx_2)
print(f"Turn 2 Rewritten Query: {r2_2.get('rewritten_query')}")
print(f"Turn 2 is_followup: {r2_2.get('is_followup')}")
print(f"Turn 2 Answer: {r2_2.get('answer')[:180]}...")
print(f"Turn 2 Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r2_2.get('sources', [])]}")
assert len(r2_2.get('sources', [])) > 0, "Expected document sources for example"
assert r2_2.get('is_followup') is True, "Expected is_followup=True"
print("[PASS] Test 2 Passed: Correctly provided document-grounded JOIN example!")

print("\n==================================================")
print("TEST 3: DML commands -> What does the first one do?")
print("==================================================")
# Turn 1
r3_1 = post_chat("What are DML commands?")
print(f"Turn 1 Answer snippet: {r3_1.get('answer', '')[:120]}...")

history_3 = [
    {"role": "user", "content": "What are DML commands?"},
    {"role": "assistant", "content": r3_1.get('answer', '')}
]
last_ctx_3 = r3_1.get("retrieved_context")

# Turn 2
r3_2 = post_chat("What does the first one do?", conversation=history_3, last_context=last_ctx_3)
print(f"Turn 2 Rewritten Query: {r3_2.get('rewritten_query')}")
print(f"Turn 2 is_followup: {r3_2.get('is_followup')}")
print(f"Turn 2 Answer: {r3_2.get('answer')[:180]}...")
print(f"Turn 2 Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r3_2.get('sources', [])]}")
assert "select" in r3_2.get('answer', '').lower() or "insert" in r3_2.get('answer', '').lower(), "Expected SELECT/INSERT in answer"
assert r3_2.get('is_followup') is True, "Expected is_followup=True"
print("[PASS] Test 3 Passed: Correctly explained the first DML command!")

print("\n==================================================")
print("TEST 4: Python -> Explain it simply.")
print("==================================================")
# Turn 1
r4_1 = post_chat("What is Python?")
print(f"Turn 1 Answer snippet: {r4_1.get('answer', '')[:120]}...")

history_4 = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": r4_1.get('answer', '')}
]
last_ctx_4 = r4_1.get("retrieved_context")

# Turn 2
r4_2 = post_chat("Explain it simply.", conversation=history_4, last_context=last_ctx_4)
print(f"Turn 2 Rewritten Query: {r4_2.get('rewritten_query')}")
print(f"Turn 2 is_followup: {r4_2.get('is_followup')}")
print(f"Turn 2 Answer: {r4_2.get('answer')[:180]}...")
print(f"Turn 2 Sources: {[s['document'] + ' Page ' + str(s['page']) for s in r4_2.get('sources', [])]}")
assert len(r4_2.get('sources', [])) > 0, "Expected document sources for Python"
assert r4_2.get('is_followup') is True, "Expected is_followup=True"
print("[PASS] Test 4 Passed: Correctly explained Python simply using document grounding!")

print("\n==================================================")
print("TEST 5: Reset Conversation (New Chat) -> What is the third property?")
print("==================================================")
# Simulate New Chat by clearing conversation history and context
req_new_chat = urllib.request.Request(f"{base_url}/api/new_chat", data=b"{}", headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req_new_chat) as resp:
    nc_data = json.loads(resp.read().decode('utf-8'))
    print(f"New Chat API status: {nc_data.get('message')}")

r5 = post_chat("What is the third property?", conversation=[], last_context=None)
print(f"No-History Query Answer: {r5.get('answer')}")
print(f"is_followup: {r5.get('is_followup')}")
print(f"Sources: {r5.get('sources')}")
assert "specify" in r5.get('answer', '').lower() or "couldn't find" in r5.get('answer', '').lower(), "Expected clarification without prior history"
print("[PASS] Test 5 Passed: Conversation history reset handled gracefully!")

print("\n==================================================")
print("ALL 5 CONVERSATION-AWARE FOLLOW-UP TESTS PASSED!")
print("==================================================")
