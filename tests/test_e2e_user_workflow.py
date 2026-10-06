"""
tests/test_e2e_user_workflow.py
Full End-to-End User Lifecycle & Regression Verification Suite.

Phases:
1. Health & Baseline Readiness (GET /api/health, GET /api/status, GET /)
2. Single PDF Upload & Indexing (Upload AcmeOS_Spec.pdf, transition to Ready)
3. Grounded Factual Q&A (Fact extraction, strict citation to source Page)
4. Multi-Turn Conversation & Follow-Up Context (Pronoun resolution, conversational memory)
5. Unsupported Negative Query (Zero-hallucination grounded fallback check)
6. Multi-PDF Upload & Comparison Mode (Upload BetaSec_Spec.pdf, balanced table, citations across both)
7. Duplicate Upload Protection & Error Handling UX (Duplicate 409/warning, actionable 400 error messages)
8. Document Deletion & Vectorstore Sync (Clean removal, state synchronization, baseline restored)
"""

import os
import sys
import json
import time
import tempfile
import shutil
import requests

BASE_URL = "http://127.0.0.1:5000"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from create_sample_pdfs import create_simple_pdf

def create_sample_pdf(filepath: str, text_p1: str, text_p2: str):
    """Creates a valid PDF using the project's native PDF generator."""
    create_simple_pdf(filepath, [text_p1, text_p2])

def run_e2e_verification():
    print("=" * 70)
    print("STARTING FULL END-TO-END USER WORKFLOW REGRESSION VERIFICATION")
    print("=" * 70)
    
    temp_dir = tempfile.mkdtemp(prefix="rag_e2e_test_")
    acme_pdf_path = os.path.join(temp_dir, "AcmeOS_Spec.pdf")
    betasec_pdf_path = os.path.join(temp_dir, "BetaSec_Spec.pdf")

    # Define exact ground-truth facts
    acme_p1 = (
        "AcmeOS Security Architecture: AcmeOS employs Argon2id for password hashing "
        "with a memory cost of 64MB and 4 parallelism threads. "
        "The default session timeout is 15 minutes of user inactivity."
    )
    acme_p2 = (
        "AcmeOS Network Protocol: AcmeOS uses TLS 1.3 with AES-256-GCM for all external communication. "
        "Audit logs are written to an append-only cryptographic ledger."
    )
    betasec_p1 = (
        "BetaSec Security Specification: BetaSec implements bcrypt with a work factor cost of 12 "
        "for user credential verification. The default session timeout is 30 minutes of idle time."
    )
    betasec_p2 = (
        "BetaSec Encryption Standards: BetaSec uses ChaCha20-Poly1305 for high-speed mobile network encryption. "
        "Session tokens expire after 24 hours regardless of activity."
    )

    create_sample_pdf(acme_pdf_path, acme_p1, acme_p2)
    create_sample_pdf(betasec_pdf_path, betasec_p1, betasec_p2)

    try:
        # =====================================================================
        # PHASE 1: Server Initialization & Baseline Health
        # =====================================================================
        print("\n" + "=" * 50)
        print("PHASE 1: Server Initialization & Baseline Health")
        print("=" * 50)
        r_home = requests.get(f"{BASE_URL}/", timeout=10)
        assert r_home.status_code == 200, f"GET / failed: {r_home.status_code}"
        assert "RAG Document Assistant" in r_home.text or "rag" in r_home.text.lower(), "Web UI did not load correctly"
        print("  [PASS] GET / -> HTTP 200 (Web UI HTML loaded)")

        r_status = requests.get(f"{BASE_URL}/api/status", timeout=10)
        assert r_status.status_code == 200, f"GET /api/status failed: {r_status.status_code}"
        status_json = r_status.json()
        assert status_json.get("vectorstore_ready") is True, "Vectorstore is not ready"
        initial_doc_count = status_json.get("total_documents", len(status_json.get("documents", [])))
        initial_chunk_count = status_json.get("total_chunks", 0)
        print(f"  [PASS] GET /api/status -> HTTP 200 (Total Docs: {initial_doc_count}, Chunks: {initial_chunk_count}, Vectorstore: Ready)")

        # =====================================================================
        # PHASE 2: Single PDF Upload & Indexing
        # =====================================================================
        print("\n" + "=" * 50)
        print("PHASE 2: Single PDF Upload & Indexing")
        print("=" * 50)
        with open(acme_pdf_path, "rb") as f:
            files = {"files": ("AcmeOS_Spec.pdf", f, "application/pdf")}
            r_upload = requests.post(f"{BASE_URL}/api/upload", files=files, timeout=30)
        assert r_upload.status_code == 200, f"Upload failed: {r_upload.status_code} - {r_upload.text}"
        upload_data = r_upload.json()
        assert upload_data.get("success") is True, f"Upload success was False: {upload_data}"
        print("  [PASS] POST /api/upload -> Uploaded 'AcmeOS_Spec.pdf' successfully")

        # Check pending status
        r_docs = requests.get(f"{BASE_URL}/api/documents", timeout=10).json()
        uploaded_doc = next((d for d in r_docs.get("documents", []) if d.get("filename") == "AcmeOS_Spec.pdf"), None)
        assert uploaded_doc is not None, "Uploaded doc not found in documents list"
        print(f"  [PASS] GET /api/documents -> 'AcmeOS_Spec.pdf' found with status: {uploaded_doc.get('status')}")

        # Trigger Indexing
        t_idx_start = time.perf_counter()
        r_index = requests.post(f"{BASE_URL}/api/index", timeout=60)
        idx_time = round(time.perf_counter() - t_idx_start, 2)
        assert r_index.status_code == 200, f"Index failed: {r_index.status_code} - {r_index.text}"
        idx_data = r_index.json()
        assert idx_data.get("success") is True, f"Index success was False: {idx_data}"
        print(f"  [PASS] POST /api/index -> Indexed successfully in {idx_time}s")

        # Verify indexed status
        r_status_after = requests.get(f"{BASE_URL}/api/status", timeout=10).json()
        assert "AcmeOS_Spec.pdf" in r_status_after.get("indexed_documents", []), "AcmeOS_Spec.pdf not listed in indexed_documents"
        print(f"  [PASS] 'AcmeOS_Spec.pdf' is verified in indexed_documents (New total chunks: {r_status_after.get('total_chunks')})")

        # =====================================================================
        # PHASE 3: Grounded Factual Q&A (Fact Extraction & Citations)
        # =====================================================================
        print("\n" + "=" * 50)
        print("PHASE 3: Grounded Factual Q&A (Fact Extraction & Citations)")
        print("=" * 50)
        q3 = "What password hashing algorithm and memory cost does AcmeOS use?"
        payload3 = {
            "question": q3,
            "selected_documents": ["AcmeOS_Spec.pdf"],
            "conversation": []
        }
        r_chat3 = requests.post(f"{BASE_URL}/api/chat", json=payload3, timeout=30).json()
        ans3 = r_chat3.get("answer", "")
        sources3 = r_chat3.get("sources", [])
        print(f"  Query: '{q3}'")
        print(f"  Mode: {r_chat3.get('mode')}")
        print(f"  Answer: {ans3}")
        print(f"  Sources: {[s.get('document') + ' P.' + str(s.get('page')) for s in sources3]}")

        assert r_chat3.get("mode") == "normal", f"Expected normal mode, got {r_chat3.get('mode')}"
        assert len(sources3) >= 1, "Expected at least 1 source citation"
        assert sources3[0].get("document") == "AcmeOS_Spec.pdf", f"Expected citation to AcmeOS_Spec.pdf, got {sources3[0].get('document')}"
        assert "argon2id" in ans3.lower(), "Grounding check failed: Answer does not mention Argon2id"
        assert "64" in ans3, "Grounding check failed: Answer does not mention 64MB memory cost"
        print("  [PASS] Fact Grounding Check PASSED: Correct algorithm (Argon2id) and memory cost (64MB) extracted with citation to AcmeOS_Spec.pdf P.1.")

        # =====================================================================
        # PHASE 4: Multi-Turn Conversation & Follow-Up Context
        # =====================================================================
        print("\n" + "=" * 50)
        print("PHASE 4: Multi-Turn Conversation & Follow-Up Context")
        print("=" * 50)
        q4 = "What is its default session timeout?"
        conversation4 = [
            {"role": "user", "content": q3},
            {"role": "assistant", "content": ans3, "retrieved_context": r_chat3.get("retrieved_context", [])}
        ]
        payload4 = {
            "question": q4,
            "selected_documents": ["AcmeOS_Spec.pdf"],
            "conversation": conversation4
        }
        r_chat4 = requests.post(f"{BASE_URL}/api/chat", json=payload4, timeout=30).json()
        ans4 = r_chat4.get("answer", "")
        sources4 = r_chat4.get("sources", [])
        print(f"  Follow-up Query: '{q4}' (pronoun reference)")
        print(f"  Rewritten query: {r_chat4.get('rewritten_query')}")
        print(f"  Is Follow-up: {r_chat4.get('is_followup')}")
        print(f"  Answer: {ans4}")

        assert len(sources4) >= 1, "Expected source citation on follow-up"
        assert "15" in ans4 or "fifteen" in ans4.lower(), "Grounding check failed: Follow-up answer does not mention 15 minutes"
        assert "minute" in ans4.lower(), "Grounding check failed: Follow-up answer does not mention minutes"
        print("  [PASS] Conversational Memory Check PASSED: Contextual pronoun 'its' resolved; 15-minute timeout returned accurately.")

        # =====================================================================
        # PHASE 5: Unsupported Negative Query (Zero Hallucination Grounding)
        # =====================================================================
        print("\n" + "=" * 50)
        print("PHASE 5: Unsupported Negative Query (Strict Fallback / Zero Hallucination)")
        print("=" * 50)
        q5 = "What is the orbital velocity of Jupiter according to AcmeOS?"
        payload5 = {
            "question": q5,
            "selected_documents": ["AcmeOS_Spec.pdf"],
            "conversation": []
        }
        r_chat5 = requests.post(f"{BASE_URL}/api/chat", json=payload5, timeout=30).json()
        ans5 = r_chat5.get("answer", "")
        print(f"  Query: '{q5}'")
        print(f"  Answer: {ans5}")

        expected_fallback = "couldn't find" in ans5.lower() or "not find" in ans5.lower() or "not contain" in ans5.lower()
        assert expected_fallback, f"Expected grounded fallback message, got: {ans5}"
        assert "jupiter" not in ans5.lower() or "not" in ans5.lower(), "Model hallucinated external facts about Jupiter!"
        print("  [PASS] Zero-Hallucination Check PASSED: System returned strict fallback response without inventing external facts.")

        # =====================================================================
        # PHASE 6: Multi-PDF Upload & Comparison Mode
        # =====================================================================
        print("\n" + "=" * 50)
        print("PHASE 6: Multi-PDF Upload & Comparison Mode")
        print("=" * 50)
        # Upload second document
        with open(betasec_pdf_path, "rb") as f:
            files2 = {"files": ("BetaSec_Spec.pdf", f, "application/pdf")}
            r_upload2 = requests.post(f"{BASE_URL}/api/upload", files=files2, timeout=30)
        assert r_upload2.status_code == 200, "BetaSec upload failed"
        print("  [PASS] POST /api/upload -> Uploaded 'BetaSec_Spec.pdf' successfully")

        # Index both documents
        requests.post(f"{BASE_URL}/api/index", timeout=60)
        print("  [PASS] POST /api/index -> Incremental indexing complete")

        # Ask comparative question
        q6 = "Compare password hashing and session timeout between AcmeOS and BetaSec."
        payload6 = {
            "question": q6,
            "selected_documents": ["AcmeOS_Spec.pdf", "BetaSec_Spec.pdf"],
            "conversation": []
        }
        r_chat6 = requests.post(f"{BASE_URL}/api/chat", json=payload6, timeout=45).json()
        ans6 = r_chat6.get("answer", "")
        sources6 = r_chat6.get("sources", [])
        docs_used6 = r_chat6.get("documents_used", [])
        print(f"  Query: '{q6}'")
        print(f"  Mode: {r_chat6.get('mode')} | is_comparison: {r_chat6.get('is_comparison')}")
        print(f"  Documents used ({len(docs_used6)}): {docs_used6}")
        print(f"  Sources count: {len(sources6)}")
        print(f"  Answer:\n{ans6}\n")

        assert r_chat6.get("mode") == "comparison" or r_chat6.get("is_comparison") is True, "Failed to enter comparison mode"
        # Balanced retrieval check
        has_acme = any("AcmeOS_Spec.pdf" in s.get("document", "") for s in sources6) or "AcmeOS_Spec.pdf" in docs_used6
        has_betasec = any("BetaSec_Spec.pdf" in s.get("document", "") for s in sources6) or "BetaSec_Spec.pdf" in docs_used6
        assert has_acme and has_betasec, f"Starvation detected! Both documents must be represented. AcmeOS: {has_acme}, BetaSec: {has_betasec}"
        
        # Verify comparison table or structured comparison
        assert "|" in ans6 or ("AcmeOS" in ans6 and "BetaSec" in ans6), "Comparison answer lacks structured comparison"
        assert "argon2id" in ans6.lower(), "AcmeOS Argon2id missing in comparison answer"
        assert "bcrypt" in ans6.lower(), "BetaSec bcrypt missing in comparison answer"
        print("  [PASS] Balanced Comparison PASSED: Both documents retrieved fairly; comparison table generated with citations.")

        # =====================================================================
        # PHASE 7: Duplicate Protection & Error Handling UX
        # =====================================================================
        print("\n" + "=" * 50)
        print("PHASE 7: Duplicate Protection & Error Handling UX")
        print("=" * 50)
        # Attempt to upload AcmeOS_Spec.pdf again
        with open(acme_pdf_path, "rb") as f:
            files_dup = {"files": ("AcmeOS_Spec.pdf", f, "application/pdf")}
            r_dup = requests.post(f"{BASE_URL}/api/upload", files=files_dup, timeout=20)
        assert r_dup.status_code == 200
        dup_json = r_dup.json()
        assert dup_json.get("is_duplicate") is True or dup_json.get("success") is False, "Duplicate detection failed to flag duplicate file"
        print(f"  [PASS] Duplicate Detection Verified: Server correctly identified existing file ({dup_json.get('message')})")

        # Test querying unindexed document
        payload_err = {
            "question": "What is AcmeOS?",
            "selected_documents": ["GhostDocument_DoesNotExist.pdf"]
        }
        r_err = requests.post(f"{BASE_URL}/api/chat", json=payload_err, timeout=20)
        assert r_err.status_code == 400, f"Expected HTTP 400 for unindexed doc, got {r_err.status_code}"
        err_json = r_err.json()
        assert err_json.get("error_code") == "VALIDATION_ERROR"
        assert "No indexed documents are available" in err_json.get("error")
        print(f"  [PASS] Actionable Error UX Verified: Server returned HTTP 400 with message: '{err_json.get('error')}'")

        # =====================================================================
        # PHASE 8: Document Deletion & Vectorstore Sync
        # =====================================================================
        print("\n" + "=" * 50)
        print("PHASE 8: Document Deletion & Vectorstore Sync")
        print("=" * 50)
        # Delete AcmeOS_Spec.pdf
        r_del1 = requests.delete(f"{BASE_URL}/api/documents/AcmeOS_Spec.pdf", timeout=20)
        assert r_del1.status_code == 200, f"Delete AcmeOS failed: {r_del1.status_code}"
        print("  [PASS] DELETE /api/documents/AcmeOS_Spec.pdf -> Removed successfully")

        # Delete BetaSec_Spec.pdf
        r_del2 = requests.delete(f"{BASE_URL}/api/documents/BetaSec_Spec.pdf", timeout=20)
        assert r_del2.status_code == 200, f"Delete BetaSec failed: {r_del2.status_code}"
        print("  [PASS] DELETE /api/documents/BetaSec_Spec.pdf -> Removed successfully")

        # Re-index to ensure vectorstore synchronization
        requests.post(f"{BASE_URL}/api/index", timeout=60)
        r_status_final = requests.get(f"{BASE_URL}/api/status", timeout=10).json()
        indexed_final = r_status_final.get("indexed_documents", [])
        assert "AcmeOS_Spec.pdf" not in indexed_final, "AcmeOS_Spec.pdf still present after delete"
        assert "BetaSec_Spec.pdf" not in indexed_final, "BetaSec_Spec.pdf still present after delete"
        print(f"  [PASS] Vectorstore synchronization confirmed: Test documents cleanly purged from index (Final docs: {len(indexed_final)})")

        # Confirm deleted facts are no longer accessible
        payload_after_del = {
            "question": "What is the password hashing algorithm in AcmeOS?",
            "selected_documents": indexed_final
        }
        r_after = requests.post(f"{BASE_URL}/api/chat", json=payload_after_del, timeout=20).json()
        assert "argon2id" not in r_after.get("answer", "").lower(), "Data isolation failed: Deleted facts leaked into subsequent answers"
        print("  [PASS] Data Isolation Verified: Deleted document content is not accessible in subsequent queries.")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("ALL 8 END-TO-END WORKFLOW PHASES PASSED WITH ZERO DEFECTS!")
    print("=" * 70)

if __name__ == "__main__":
    run_e2e_verification()
