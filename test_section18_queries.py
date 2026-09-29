import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def post_chat(question, conversation=None, selected_docs=None):
    payload = {
        "question": question,
        "conversation": conversation or [],
        "selected_documents": selected_docs
    }
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("==================================================")
    print("RUNNING SECTION 18 TEST QUERIES")
    print("==================================================")

    # 1. "What is DML?"
    r1 = post_chat("What is DML?")
    print(f"\n1. 'What is DML?'")
    print(f"   Success: {r1.get('success')}, Method: {r1.get('retrieval_method')}")
    print(f"   Sources: {[s['document'] for s in r1.get('sources', [])]}")
    assert r1.get('success') and len(r1.get('sources', [])) > 0, "Failed Q1"

    # 2. "What is ACID?"
    r2 = post_chat("What is ACID?")
    print(f"\n2. 'What is ACID?'")
    print(f"   Success: {r2.get('success')}")
    print(f"   Sources: {[s['document'] for s in r2.get('sources', [])]}")
    assert r2.get('success') and any('DBMS' in s['document'] for s in r2.get('sources', [])), "Failed Q2"

    # 3. "What is normalization?"
    r3 = post_chat("What is normalization?")
    print(f"\n3. 'What is normalization?'")
    print(f"   Success: {r3.get('success')}")
    print(f"   Sources: {[s['document'] for s in r3.get('sources', [])]}")
    assert r3.get('success'), "Failed Q3"

    # 4. "What are the types of normalization?"
    r4 = post_chat("What are the types of normalization?")
    print(f"\n4. 'What are the types of normalization?'")
    print(f"   Success: {r4.get('success')}")
    print(f"   Sources: {[s['document'] for s in r4.get('sources', [])]}")
    assert r4.get('success'), "Failed Q4"

    # 5. "What is the objective of CommerceOS?"
    r5 = post_chat("What is the objective of CommerceOS?")
    print(f"\n5. 'What is the objective of CommerceOS?'")
    print(f"   Success: {r5.get('success')}")
    print(f"   Sources: {[s['document'] for s in r5.get('sources', [])]}")
    assert r5.get('success') and any('CommerceOS' in s['document'] for s in r5.get('sources', [])), "Failed Q5"

    # 6. "Compare the objectives in DBMS Notes and CommerceOS."
    r6 = post_chat("Compare the objectives in DBMS Notes and CommerceOS.")
    print(f"\n6. 'Compare the objectives in DBMS Notes and CommerceOS.'")
    print(f"   Success: {r6.get('success')}, Mode: {r6.get('mode')}")
    print(f"   Documents used: {r6.get('documents_used', [])}")
    assert r6.get('success'), "Failed Q6"

    # 7. Follow-up: "What is normalization?" then "What are its types?"
    history = [
        {"role": "user", "content": "What is normalization?"},
        {"role": "assistant", "content": r3.get("answer", ""), "retrieved_context": r3.get("retrieved_context", [])}
    ]
    r7 = post_chat("What are its types?", conversation=history)
    print(f"\n7. Follow-up: 'What are its types?'")
    print(f"   Rewritten: {r7.get('rewritten_query')}")
    print(f"   Is Followup: {r7.get('is_followup')}")
    print(f"   Sources: {[s['document'] for s in r7.get('sources', [])]}")
    assert r7.get('is_followup') or 'normalization' in r7.get('rewritten_query', '').lower(), "Failed Q7"

    # 8. Select only Python_Notes.pdf and ask about ACID (which is in DBMS_Notes.pdf)
    r8 = post_chat("What is ACID?", selected_docs=["Python_Notes.pdf"])
    print(f"\n8. Document Filtering: Select 'Python_Notes.pdf' only and ask 'What is ACID?'")
    print(f"   Answer: {r8.get('answer')[:80]}...")
    print(f"   Sources: {[s['document'] for s in r8.get('sources', [])]}")
    assert not any('DBMS' in s['document'] for s in r8.get('sources', [])), "Failed Q8: Unselected doc used!"
    print("   [PASS] Strict filtering verified: unselected DBMS_Notes.pdf was NOT used.")

    print("\n==================================================")
    print("ALL 8 SECTION 18 TEST CASES PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
