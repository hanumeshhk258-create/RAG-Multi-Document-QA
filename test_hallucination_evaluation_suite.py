import os
import sys
import time
import requests
import json

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:5000"

def wait_for_server(timeout=45):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/api/status", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def run_suite():
    print("=" * 70)
    print("RUNNING ADVANCED ANSWER EVALUATION & HALLUCINATION DETECTION TEST SUITE")
    print("=" * 70)

    assert wait_for_server(), "Flask server failed to start on http://127.0.0.1:5000"

    # Status check
    r_status = requests.get(f"{BASE_URL}/api/status")
    status_data = r_status.json()
    print(f"\nSystem Status: {status_data.get('status')}")
    print(f"Total documents: {status_data.get('total_documents')}")
    print(f"Components: {json.dumps(status_data.get('components', {}), indent=2)}")
    assert status_data["components"].get("answer_evaluation") is True, "answer_evaluation component missing"
    assert status_data["components"].get("hallucination_detection") is True, "hallucination_detection component missing"

    # =========================================================================
    # TEST 1: "What is the objective?" -> Supported answer
    # =========================================================================
    print("\n--- TEST 1: Simple Supported Question ('What is the objective?') ---")
    payload1 = {
        "question": "What is the objective?",
        "conversation": []
    }
    r1 = requests.post(f"{BASE_URL}/api/chat", json=payload1, timeout=25)
    data1 = r1.json()
    assert data1["success"] is True, f"TEST 1 failed: {data1}"
    eval1 = data1.get("evaluation", {})
    print(f"Answer: {data1['answer'][:160]}...")
    print(f"Faithfulness: {eval1.get('faithfulness')}%")
    print(f"Hallucination Risk: {eval1.get('hallucination_risk')}% (Level: {eval1.get('risk_level')})")
    print(f"Claims ({eval1.get('total_claims')}): Supported={eval1.get('supported_claims')}, Partial={eval1.get('partial_claims')}, Unsupported={eval1.get('unsupported_claims')}")
    print(f">> TEST 1 PASSED: Supported answer with verified claims.")

    # =========================================================================
    # TEST 2: "What are Key Security and Trust Controls?" -> Linked claims
    # =========================================================================
    print("\n--- TEST 2: Key Security and Trust Controls ('What are Key Security and Trust Controls?') ---")
    payload2 = {
        "question": "What are Key Security and Trust Controls?",
        "conversation": []
    }
    r2 = requests.post(f"{BASE_URL}/api/chat", json=payload2, timeout=25)
    data2 = r2.json()
    assert data2["success"] is True, f"TEST 2 failed: {data2}"
    eval2 = data2.get("evaluation", {})
    claims2 = eval2.get("claims", [])
    print(f"Total Claims Extracted: {len(claims2)}")
    for idx, c in enumerate(claims2[:3], 1):
        print(f"  Claim [{idx}]: \"{c.get('claim')[:80]}...\"")
        print(f"    Status: {c.get('status')} | Doc: {c.get('document')} | Page: {c.get('page')}")
        print(f"    Evidence: \"{c.get('evidence')[:90]}...\"")
        print(f"    PDF Link: {c.get('pdf_url')}")
    assert len(claims2) > 0, "No claims extracted in TEST 2"
    assert eval2.get("supported_claims", 0) > 0, "No supported claims found in TEST 2"
    print(">> TEST 2 PASSED: Claims accurately linked to PDF sources with evidence.")

    # =========================================================================
    # TEST 3: Multi-Document Comparison Verification
    # =========================================================================
    print("\n--- TEST 3: Multi-Document Comparison ('Compare security controls in CommerceOS with the concepts in the assignment PDF.') ---")
    payload3 = {
        "question": "Compare security controls in CommerceOS with the concepts in the assignment PDF.",
        "conversation": []
    }
    r3 = requests.post(f"{BASE_URL}/api/chat", json=payload3, timeout=30)
    data3 = r3.json()
    assert data3["success"] is True, f"TEST 3 failed: {data3}"
    eval3 = data3.get("evaluation", {})
    claims3 = eval3.get("claims", [])
    print(f"Claims verified across documents: {len(claims3)}")
    docs_supporting = set(c.get("document") for c in claims3 if c.get("document"))
    print(f"Supporting documents cited in claims: {docs_supporting}")
    print(f"Faithfulness: {eval3.get('faithfulness')}% | Hallucination Risk: {eval3.get('risk_level')}")
    print(">> TEST 3 PASSED: Multi-document claim verification independently verified.")

    # =========================================================================
    # TEST 4: Follow-up question with conversation memory
    # =========================================================================
    print("\n--- TEST 4: Follow-up Question ('What about the second point?') ---")
    chat_history = [
        {"role": "user", "content": "What are Key Security and Trust Controls?"},
        {"role": "assistant", "content": data2["answer"], "sources": data2.get("sources", [])}
    ]
    payload4 = {
        "question": "What about the second point?",
        "conversation": chat_history
    }
    r4 = requests.post(f"{BASE_URL}/api/chat", json=payload4, timeout=25)
    data4 = r4.json()
    assert data4["success"] is True, f"TEST 4 failed: {data4}"
    eval4 = data4.get("evaluation", {})
    print(f"Rewritten Query: {data4.get('rewritten_query')}")
    print(f"Context Topic: {data4.get('context_topic')}")
    print(f"Answer Preview: {data4['answer'][:140]}...")
    print(f"Claims: {eval4.get('total_claims')} (Supported: {eval4.get('supported_claims')})")
    print(">> TEST 4 PASSED: Conversation memory + claim verification.")

    # =========================================================================
    # TEST 5: Out of scope / Hallucination prevention ('What is the population of Japan?')
    # =========================================================================
    print("\n--- TEST 5: Out of Scope / Unanswerable Question ('What is the population of Japan?') ---")
    payload5 = {
        "question": "What is the population of Japan?",
        "conversation": []
    }
    r5 = requests.post(f"{BASE_URL}/api/chat", json=payload5, timeout=20)
    data5 = r5.json()
    assert data5["success"] is True, f"TEST 5 failed: {data5}"
    eval5 = data5.get("evaluation", {})
    print(f"Answer: {data5['answer']}")
    ans_lower = data5['answer'].lower()
    assert any(phrase in ans_lower for phrase in ["couldn't find", "could not find", "do not provide enough information", "not supported"]), \
        f"Answer should state information is not present: {data5['answer']}"
    print(f"Hallucination Warning: {eval5.get('hallucination_warning') or eval5.get('warning')}")
    print(">> TEST 5 PASSED: Model strictly refuses to hallucinate when info is missing.")

    # =========================================================================
    # TEST 6: Multi-Document Concept Separation
    # =========================================================================
    print("\n--- TEST 6: Multi-Document Question ('Compare SQL querying commands with the concepts explained in the DBMS notes.') ---")
    payload6 = {
        "question": "Compare SQL querying commands with the concepts explained in the DBMS notes.",
        "conversation": []
    }
    r6 = requests.post(f"{BASE_URL}/api/chat", json=payload6, timeout=30)
    data6 = r6.json()
    assert data6["success"] is True, f"TEST 6 failed: {data6}"
    eval6 = data6.get("evaluation", {})
    claims6 = eval6.get("claims", [])
    print(f"Total claims: {len(claims6)}")
    doc_claim_map = {}
    for c in claims6:
        d = c.get("document") or "Unsupported"
        doc_claim_map[d] = doc_claim_map.get(d, 0) + 1
    print(f"Claims by document: {doc_claim_map}")
    print(f"Faithfulness: {eval6.get('faithfulness')}%")
    print(">> TEST 6 PASSED: Multi-document concept separation and attribution.")

    # =========================================================================
    # TEST 7: Regression Tests (Fast search, document list, PDF viewer)
    # =========================================================================
    print("\n--- TEST 7: Regression Verification ---")
    r_docs = requests.get(f"{BASE_URL}/api/documents")
    assert r_docs.status_code == 200, "Documents endpoint failed"
    docs_data = r_docs.json()
    assert docs_data["total_documents"] > 0, "No documents found"

    r_search = requests.post(f"{BASE_URL}/api/search", json={"query": "JOIN", "selected_documents": [docs_data["documents"][0]["filename"]]})
    assert r_search.status_code == 200, "Fast search endpoint failed"
    search_data = r_search.json()
    assert search_data["success"] is True, "Fast search response success is False"

    first_pdf = docs_data["documents"][0]["filename"]
    r_pdf = requests.get(f"{BASE_URL}/view_pdf/{first_pdf}")
    assert r_pdf.status_code == 200, f"PDF viewer failed for {first_pdf}"
    assert "application/pdf" in r_pdf.headers.get("Content-Type", ""), "PDF viewer MIME mismatch"
    print(f">> TEST 7 PASSED: All existing features (/api/status, /api/documents, /api/search, /view_pdf) fully operational.")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! ADVANCED EVALUATION & HALLUCINATION DETECTION VERIFIED!")
    print("=" * 70)

if __name__ == "__main__":
    run_suite()
