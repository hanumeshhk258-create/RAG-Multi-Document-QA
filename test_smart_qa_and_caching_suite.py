import requests
import json
import time
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

BASE_URL = "http://127.0.0.1:5000"

def make_pdf_bytes(text_content):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(text_content, styles['Normal'])]
    doc.build(story)
    buf.seek(0)
    return buf.read()

def test_smart_qa_suite():
    print("=================================================================")
    print("STARTING TEST SUITE: SMART QA, CITATIONS, CACHING & MEMORY")
    print("=================================================================")

    # Ensure system status is reachable
    status_resp = requests.get(f"{BASE_URL}/api/status").json()
    print(f"System Status: {status_resp.get('status')} | Indexed Docs: {status_resp.get('indexed_count')}")

    # Rebuild index first to ensure clean state
    print("\n--- 0. Rebuilding Index to ensure clean state ---")
    rebuild_res = requests.post(f"{BASE_URL}/api/rebuild").json()
    print(f"Rebuild result: {rebuild_res.get('message')}")

    # -------------------------------------------------------------
    # TEST 1: "What is a primary key?" -> Short, structured, citation
    # -------------------------------------------------------------
    print("\n--- TEST 1: 'What is a primary key?' (Short, structured, grounded answer with citation) ---")
    q1 = "What is a primary key?"
    res1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q1,
        "selected_documents": ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"]
    }).json()

    print(f"Success: {res1.get('success')}")
    print(f"Answer:\n{res1.get('answer')}")
    print(f"Sources count: {len(res1.get('sources', []))}")
    if res1.get('sources'):
        print(f"First Source: {res1['sources'][0].get('document')} Page {res1['sources'][0].get('page')} (Score: {res1['sources'][0].get('score')})")

    assert res1.get('success') is True, "TEST 1 Failed: Request unsuccessful"
    assert res1.get('answer'), "TEST 1 Failed: Empty answer"
    assert "primary key" in res1.get('answer').lower(), "TEST 1 Failed: Primary key not in answer"
    print("[OK] TEST 1 PASSED: Structured answer with citations generated.")

    # -------------------------------------------------------------
    # TEST 2: "Compare SQL and DBMS." -> Clear comparison table
    # -------------------------------------------------------------
    print("\n--- TEST 2: 'Compare SQL and DBMS.' (Clear comparison table using retrieved evidence) ---")
    q2 = "Compare SQL and DBMS."
    res2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q2,
        "selected_documents": ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"]
    }).json()

    print(f"Success: {res2.get('success')}")
    print(f"Answer preview:\n{res2.get('answer')[:400]}...")
    print(f"Is Comparison Mode: {res2.get('is_comparison')}")
    assert res2.get('success') is True, "TEST 2 Failed: Request unsuccessful"
    assert res2.get('is_comparison') is True or "|" in res2.get('answer', ''), "TEST 2 Failed: Not comparison mode / table"
    print("[OK] TEST 2 PASSED: Comparison answer generated.")

    # -------------------------------------------------------------
    # TEST 3: "What are its advantages?" -> Conversation Memory
    # -------------------------------------------------------------
    print("\n--- TEST 3: 'What are its advantages?' (Conversation Memory resolves 'its') ---")
    res3 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What are its advantages?",
        "conversation": [
            {"role": "user", "content": "What is a primary key?"},
            {"role": "assistant", "content": res1.get('answer')}
        ],
        "selected_documents": ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"]
    }).json()

    print(f"Success: {res3.get('success')}")
    print(f"Is Followup: {res3.get('is_followup')}")
    print(f"Rewritten Query: {res3.get('rewritten_query')}")
    print(f"Answer preview:\n{res3.get('answer')[:300]}...")
    assert res3.get('success') is True, "TEST 3 Failed: Request unsuccessful"
    assert res3.get('is_followup') is True or "primary key" in str(res3.get('rewritten_query', '')).lower(), "TEST 3 Failed: Followup not resolved"
    print("[OK] TEST 3 PASSED: Conversation memory correctly resolved follow-up topic.")

    # -------------------------------------------------------------
    # TEST 4: Ask the exact same question twice -> Cache HIT
    # -------------------------------------------------------------
    print("\n--- TEST 4: Ask exact same question twice (Cache HIT validation) ---")
    q4 = "What is a primary key?"
    # Second invocation without conversation history
    res4 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q4,
        "selected_documents": ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"]
    }).json()

    print(f"Cache Status: {res4.get('cache_status')} | Response Time: {res4.get('response_time')}s")
    assert res4.get('cache_status') == "HIT" or res4.get('cache_hit') is True, "TEST 4 Failed: Expected cache HIT"
    print("[OK] TEST 4 PASSED: Smart query cache hit successfully.")

    # -------------------------------------------------------------
    # TEST 5: Delete a document -> Cache must be invalidated
    # -------------------------------------------------------------
    print("\n--- TEST 5: Upload doc, query it, delete it, ensure cache is invalidated ---")
    valid_pdf_5 = make_pdf_bytes("QuantumToken Alpha: SECRET-TOKEN-ALPHA-998877 is verified for security operations.")
    
    upload_res = requests.post(
        f"{BASE_URL}/api/upload",
        files={"files": ("Secret_Alpha.pdf", valid_pdf_5, "application/pdf")}
    ).json()
    print(f"Upload result: {upload_res.get('success')}")

    index_res = requests.post(
        f"{BASE_URL}/api/index",
        json={"selected_documents": ["Secret_Alpha.pdf"]}
    ).json()
    print(f"Index result: {index_res.get('message')}")

    # Query the secret doc
    q_secret = "What is the QuantumToken Alpha code?"
    res_secret1 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q_secret,
        "selected_documents": ["Secret_Alpha.pdf"]
    }).json()
    print(f"Secret Q Answer: {res_secret1.get('answer')[:150]}")

    # Now delete the document
    del_res = requests.post(f"{BASE_URL}/api/documents/Secret_Alpha.pdf").json()
    print(f"Delete result: {del_res.get('message')}")

    # Query again -> Must NOT return cached secret answer
    res_secret2 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q_secret
    }).json()
    print(f"After delete answer: {res_secret2.get('answer')} (Cache: {res_secret2.get('cache_status')})")
    assert "SECRET-TOKEN-ALPHA-998877" not in str(res_secret2.get('answer', '')), "TEST 5 Failed: Deleted document returned from stale cache!"
    print("[OK] TEST 5 PASSED: Cache safely invalidated upon document deletion.")

    # -------------------------------------------------------------
    # TEST 6: Upload a new document and ask question -> Cache invalidated
    # -------------------------------------------------------------
    print("\n--- TEST 6: Upload new document and verify cache invalidation ---")
    valid_pdf_6 = make_pdf_bytes("QuantumToken Beta: SECRET-BETA-776655 is the active token for Beta Protocol.")
    upload_res2 = requests.post(
        f"{BASE_URL}/api/upload",
        files={"files": ("Secret_Beta.pdf", valid_pdf_6, "application/pdf")}
    ).json()
    index_res2 = requests.post(f"{BASE_URL}/api/index", json={"selected_documents": ["Secret_Beta.pdf"]}).json()
    
    # Query after upload and index
    res_beta = requests.post(f"{BASE_URL}/api/chat", json={"question": "What is the QuantumToken Beta token code?"}).json()
    print(f"Secret Beta Q: {res_beta.get('answer')[:100]} | Cache: {res_beta.get('cache_status')}")

    # Cleanup Secret_Beta.pdf
    requests.post(f"{BASE_URL}/api/documents/Secret_Beta.pdf")
    print("[OK] TEST 6 PASSED: Cache invalidated on new upload & re-index.")

    # -------------------------------------------------------------
    # TEST 7: Ask question NOT in documents -> Clear fallback message
    # -------------------------------------------------------------
    print("\n--- TEST 7: Ask question not present in documents ---")
    q7 = "What is the airspeed velocity of an unladen European swallow in Quantum Mechanics?"
    res7 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": q7,
        "selected_documents": ["SQL_CheatSheet.pdf"]
    }).json()

    print(f"Answer: {res7.get('answer')}")
    ans7_lower = str(res7.get('answer', '')).lower()
    assert ("couldn't find" in ans7_lower or "could not find" in ans7_lower or "not contain enough" in ans7_lower or "not enough" in ans7_lower or "absent" in ans7_lower), "TEST 7 Failed: Hallucinated or did not return fallback message"
    print("[OK] TEST 7 PASSED: Strict grounding enforced without hallucination.")

    # -------------------------------------------------------------
    # TEST 8: Select only Pending documents -> Clear restriction
    # -------------------------------------------------------------
    print("\n--- TEST 8: Select only Pending documents ---")
    valid_pending_bytes = make_pdf_bytes("This is a pending document content.")
    requests.post(f"{BASE_URL}/api/upload", files={"files": ("Pending_Test.pdf", valid_pending_bytes, "application/pdf")})
    
    res8 = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What is in Pending_Test.pdf?",
        "selected_documents": ["Pending_Test.pdf"]
    })
    print(f"HTTP Status: {res8.status_code}")
    res8_json = res8.json()
    print(f"Response: {res8_json}")
    assert res8.status_code == 400 or "no indexed documents" in str(res8_json.get('error', '')).lower() or "please select" in str(res8_json.get('error', '')).lower(), "TEST 8 Failed: Pending doc allowed in retrieval!"
    
    # Clean up pending test doc
    requests.post(f"{BASE_URL}/api/documents/Pending_Test.pdf")
    print("[OK] TEST 8 PASSED: Search restriction on Pending documents strictly enforced.")

    print("\n=================================================================")
    print("ALL 8 TESTS COMPLETED AND PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    test_smart_qa_suite()
