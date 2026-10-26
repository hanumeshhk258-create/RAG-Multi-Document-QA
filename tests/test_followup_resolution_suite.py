"""
Automated Test Suite for Follow-up Question Resolution, Context Propagation, and Isolation.

Tests:
1. "What are the four ACID properties?" -> Returns all 4 properties.
2. "Explain the first property." -> Grounded explanation of Atomicity, context: Atomicity (ACID).
3. "Explain the second property." -> Grounded explanation of Consistency, context: Consistency (ACID).
4. "What is the third property?" -> Grounded explanation of Isolation, context: Isolation (ACID).
5. "Explain the last property." -> Grounded explanation of Durability, context: Durability (ACID).
6. "Give me an example of the second property." -> Grounded explanation/example of Consistency.
7. Start New Chat and ask: "Explain the second property." -> Refuses without prior context / asks for clarification / strict fallback.
8. Switch to another session with different history -> Strict session isolation without contamination.
"""

import sys
import os
import json
import time
import unittest
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"


def send_chat_request(query, chat_history=None, selected_documents=None, session_id=None):
    url = f"{BASE_URL}/api/chat"
    payload = {
        "query": query,
        "question": query,
        "chat_history": chat_history or [],
        "conversation": chat_history or [],
        "history": chat_history or [],
        "selected_documents": selected_documents or ["DBMS_Notes.pdf"],
        "session_id": session_id or "test_session_acid"
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))
    except Exception as e:
        return 500, {"error": str(e)}


class TestFollowUpResolution(unittest.TestCase):

    def test_01_full_acid_conversation_lifecycle(self):
        print("\n" + "=" * 80)
        print("RUNNING TESTS 1 THROUGH 6: ACID Follow-up Conversation Chain")
        print("=" * 80)

        history = []
        session_id = f"acid_eval_session_{int(time.time())}"

        # -----------------------------------------------------------------
        # TEST 1: "What are the four ACID properties?"
        # -----------------------------------------------------------------
        q1 = "What are the four ACID properties?"
        print(f"\n[TEST 1] User Question: '{q1}'")
        status, res1 = send_chat_request(q1, history, ["DBMS_Notes.pdf"], session_id)
        self.assertEqual(status, 200, f"Test 1 failed with status {status}: {res1}")
        self.assertTrue(res1.get("success"), f"Test 1 failed: {res1}")
        ans1 = res1.get("final_verified_answer") or res1.get("answer", "")
        print(f"[TEST 1] Answer Preview:\n{ans1[:250]}...")

        # Assert all 4 properties appear in the answer
        ans1_lower = ans1.lower()
        self.assertIn("atomicity", ans1_lower, "Test 1 answer must contain Atomicity")
        self.assertIn("consistency", ans1_lower, "Test 1 answer must contain Consistency")
        self.assertIn("isolation", ans1_lower, "Test 1 answer must contain Isolation")
        self.assertIn("durability", ans1_lower, "Test 1 answer must contain Durability")

        # Update history
        history.append({"role": "user", "content": q1})
        history.append({"role": "assistant", "content": ans1})

        # -----------------------------------------------------------------
        # TEST 2: "Explain the first property."
        # -----------------------------------------------------------------
        q2 = "Explain the first property."
        print(f"\n[TEST 2] User Question: '{q2}'")
        status, res2 = send_chat_request(q2, history, ["DBMS_Notes.pdf"], session_id)
        self.assertEqual(status, 200, f"Test 2 failed with status {status}: {res2}")
        ans2 = res2.get("final_verified_answer") or res2.get("answer", "")
        ctx2 = res2.get("context_topic", "")
        rw2 = res2.get("rewritten_query", "")
        print(f"[TEST 2] Context Topic: '{ctx2}' | Rewritten: '{rw2}'")
        print(f"[TEST 2] Answer Preview:\n{ans2[:250]}...")

        self.assertIn("Atomicity", ctx2, f"Test 2 context label should identify Atomicity, got: '{ctx2}'")
        self.assertIn("atomicity", ans2.lower(), "Test 2 answer should explain Atomicity")
        self.assertTrue(
            res2.get("is_followup"),
            "Test 2 should be recognized as a follow-up"
        )

        history.append({"role": "user", "content": q2})
        history.append({"role": "assistant", "content": ans2})

        # -----------------------------------------------------------------
        # TEST 3: "Explain the second property." (and "Explain second property of that?")
        # -----------------------------------------------------------------
        q3 = "Explain second property of that?"
        print(f"\n[TEST 3] User Question: '{q3}'")
        status, res3 = send_chat_request(q3, history, ["DBMS_Notes.pdf"], session_id)
        self.assertEqual(status, 200, f"Test 3 failed with status {status}: {res3}")
        ans3 = res3.get("final_verified_answer") or res3.get("answer", "")
        ctx3 = res3.get("context_topic", "")
        rw3 = res3.get("rewritten_query", "")
        print(f"[TEST 3] Context Topic: '{ctx3}' | Rewritten: '{rw3}'")
        print(f"[TEST 3] Answer Preview:\n{ans3[:250]}...")

        self.assertIn("Consistency", ctx3, f"Test 3 context label should identify Consistency, got: '{ctx3}'")
        self.assertNotIn("last property of that", ctx3.lower(), "Test 3 context label must NOT be 'last property of that'")
        self.assertIn("consistency", ans3.lower(), "Test 3 answer should explain Consistency")
        self.assertNotIn("I couldn't find sufficient information", ans3, "Test 3 should NOT return unsupported fallback")

        history.append({"role": "user", "content": q3})
        history.append({"role": "assistant", "content": ans3})

        # -----------------------------------------------------------------
        # TEST 4: "What is the third property?"
        # -----------------------------------------------------------------
        q4 = "What is the third property?"
        print(f"\n[TEST 4] User Question: '{q4}'")
        status, res4 = send_chat_request(q4, history, ["DBMS_Notes.pdf"], session_id)
        self.assertEqual(status, 200, f"Test 4 failed with status {status}: {res4}")
        ans4 = res4.get("final_verified_answer") or res4.get("answer", "")
        ctx4 = res4.get("context_topic", "")
        rw4 = res4.get("rewritten_query", "")
        print(f"[TEST 4] Context Topic: '{ctx4}' | Rewritten: '{rw4}'")
        print(f"[TEST 4] Answer Preview:\n{ans4[:250]}...")

        self.assertIn("Isolation", ctx4, f"Test 4 context label should identify Isolation, got: '{ctx4}'")
        self.assertIn("isolation", ans4.lower(), "Test 4 answer should explain Isolation")

        history.append({"role": "user", "content": q4})
        history.append({"role": "assistant", "content": ans4})

        # -----------------------------------------------------------------
        # TEST 5: "Explain the last property."
        # -----------------------------------------------------------------
        q5 = "Explain the last property."
        print(f"\n[TEST 5] User Question: '{q5}'")
        status, res5 = send_chat_request(q5, history, ["DBMS_Notes.pdf"], session_id)
        self.assertEqual(status, 200, f"Test 5 failed with status {status}: {res5}")
        ans5 = res5.get("final_verified_answer") or res5.get("answer", "")
        ctx5 = res5.get("context_topic", "")
        rw5 = res5.get("rewritten_query", "")
        print(f"[TEST 5] Context Topic: '{ctx5}' | Rewritten: '{rw5}'")
        print(f"[TEST 5] Answer Preview:\n{ans5[:250]}...")

        self.assertIn("Durability", ctx5, f"Test 5 context label should identify Durability, got: '{ctx5}'")
        self.assertIn("durability", ans5.lower(), "Test 5 answer should explain Durability")

        history.append({"role": "user", "content": q5})
        history.append({"role": "assistant", "content": ans5})

        # -----------------------------------------------------------------
        # TEST 6: "Give me an example of the second property."
        # Expected: A document-grounded response if supported, or strict fallback behavior.
        # -----------------------------------------------------------------
        q6 = "Give me an example of the second property."
        print(f"\n[TEST 6] User Question: '{q6}'")
        status, res6 = send_chat_request(q6, history, ["DBMS_Notes.pdf"], session_id)
        self.assertEqual(status, 200, f"Test 6 failed with status {status}: {res6}")
        ans6 = res6.get("final_verified_answer") or res6.get("answer", "")
        ctx6 = res6.get("context_topic", "")
        rw6 = res6.get("rewritten_query", "")
        print(f"[TEST 6] Context Topic: '{ctx6}' | Rewritten: '{rw6}'")
        print(f"[TEST 6] Answer Preview:\n{ans6[:250]}...")

        self.assertIn("Consistency", ctx6, f"Test 6 context label should identify Consistency, got: '{ctx6}'")
        self.assertIn("Consistency", rw6, f"Test 6 rewritten query should specify Consistency, got: '{rw6}'")
        # DBMS_Notes.pdf does not define a concrete transaction example, so strict fallback is expected
        is_supported_or_strict_fallback = (
            "consistency" in ans6.lower() or
            "couldn't find" in ans6.lower() or
            "not contain enough information" in ans6.lower()
        )
        self.assertTrue(is_supported_or_strict_fallback, f"Test 6 should return grounded answer or strict fallback, got: '{ans6}'")

    def test_02_new_chat_isolation(self):
        print("\n" + "=" * 80)
        print("RUNNING TEST 7: New Chat Clean Startup Isolation")
        print("=" * 80)

        # -----------------------------------------------------------------
        # TEST 7: Start New Chat and ask: "Explain the second property."
        # History is EMPTY. Must not reuse previous session's ACID list!
        # -----------------------------------------------------------------
        q7 = "Explain the second property."
        print(f"\n[TEST 7] User Question in clean session: '{q7}'")
        status, res7 = send_chat_request(q7, chat_history=[], selected_documents=["DBMS_Notes.pdf"], session_id="new_clean_chat")
        self.assertEqual(status, 200, f"Test 7 failed with status {status}: {res7}")
        ans7 = res7.get("final_verified_answer") or res7.get("answer", "")
        is_fu = res7.get("is_followup", False)
        ctx7 = res7.get("context_topic", "")
        print(f"[TEST 7] Answer: '{ans7}' | is_followup: {is_fu} | context_topic: '{ctx7}'")

        # Must not fabricate an answer about Consistency without context
        self.assertFalse(is_fu, "Test 7 should NOT be flagged as an understood follow-up in empty chat")
        self.assertEqual(ctx7, "", "Test 7 context topic should be empty")
        # Should ask for clarification or state insufficient information
        is_clarif_or_fallback = (
            "Please specify" in ans7 or
            "Please clarify" in ans7 or
            "couldn't find" in ans7.lower() or
            "do not contain enough information" in ans7.lower()
        )
        self.assertTrue(is_clarif_or_fallback, f"Test 7 should request clarification or fallback, got: '{ans7}'")

    def test_03_switch_session_isolation(self):
        print("\n" + "=" * 80)
        print("RUNNING TEST 8: Switch Session Memory Isolation")
        print("=" * 80)

        # -----------------------------------------------------------------
        # TEST 8: Switch to a session that discusses Python, and ask follow-up
        # -----------------------------------------------------------------
        python_history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its readable syntax, dynamic typing, and rich ecosystem of libraries."}
        ]
        q8 = "What are its key data structures?"
        print(f"\n[TEST 8] User Question in Python session: '{q8}'")
        status, res8 = send_chat_request(q8, chat_history=python_history, selected_documents=["Python_Notes.pdf"], session_id="python_session")
        self.assertEqual(status, 200, f"Test 8 failed with status {status}: {res8}")
        ans8 = res8.get("final_verified_answer") or res8.get("answer", "")
        ctx8 = res8.get("context_topic", "")
        rw8 = res8.get("rewritten_query", "")
        print(f"[TEST 8] Context Topic: '{ctx8}' | Rewritten: '{rw8}'")
        print(f"[TEST 8] Answer Preview:\n{ans8[:250]}...")

        # Must use Python context, NOT ACID!
        self.assertNotIn("acid", ctx8.lower(), "Test 8 context must NOT mention ACID")
        self.assertIn("python", ctx8.lower() + rw8.lower(), "Test 8 should resolve to Python")
        self.assertIn("list", ans8.lower(), "Test 8 answer should discuss data structures (Lists, Tuples, Dictionaries)")
        self.assertIn("tuple", ans8.lower(), "Test 8 answer should discuss data structures (Lists, Tuples, Dictionaries)")


if __name__ == "__main__":
    unittest.main()
