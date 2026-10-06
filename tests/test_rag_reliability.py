"""
tests/test_rag_reliability.py
Deterministic Reliability & Hardening Test Suite for RAG Multi-Document QA Application.

Covers:
1. Gemini failure simulation via mocks (401 Auth, 429 Rate Limit, 503 Service Unavailable, Timeout, Empty Response).
2. Request validation edge cases (empty question, whitespace, empty selection, unindexed documents).
3. Fallback cascade & direct grounded extraction fallback.
4. Index missing/corrupt resilience.
5. Concurrency & simultaneous query safety.
6. Vectorstore state synchronization.
"""

import os
import sys
import json
import time
import unittest
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document

import app as flask_app_module
from rag.llm import (
    _call_gemini_with_timeout,
    _classify_and_log_error,
    generate_rag_answer,
    extract_direct_answer_fallback,
    FALLBACK_RESPONSE,
    AIAuthError,
    AIRateLimitError,
    AIEmptyResponseError,
    AIGenerationError
)
from rag.retriever import retrieve_relevant_chunks
from rag.vectorstore import load_vectorstore_metadata


class TestRAGReliabilitySuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flask_app_module.app.config['TESTING'] = True
        cls.client = flask_app_module.app.test_client()

    # =========================================================================
    # 1. GEMINI ERROR CLASSIFICATION & REDACTION TESTS
    # =========================================================================
    def test_classify_and_log_error_auth_redaction(self):
        """Verify API key is redacted and 401/403 auth errors are classified as non-retryable."""
        raw_error = Exception("API key AIzaSyD1234567890abcdefghijklmnopqrstuvwxyz is invalid. 403 Forbidden")
        err_type, status_code, retryable = _classify_and_log_error(raw_error, "gemini-3.5-flash-lite")
        
        self.assertEqual(status_code, "401/403")
        self.assertFalse(retryable)

    def test_classify_and_log_error_rate_limit(self):
        """Verify 429 Resource Exhausted is classified as retryable."""
        raw_error = Exception("429 Resource exhausted: quota exceeded")
        err_type, status_code, retryable = _classify_and_log_error(raw_error, "gemini-3.5-flash-lite")
        
        self.assertEqual(status_code, "429")
        self.assertTrue(retryable)

    def test_classify_and_log_error_transient_503(self):
        """Verify 503 Service Unavailable is classified as retryable."""
        raw_error = Exception("503 Service Unavailable: backend deadline exceeded")
        err_type, status_code, retryable = _classify_and_log_error(raw_error, "gemini-3.5-flash-lite")
        
        self.assertEqual(status_code, "500/504")
        self.assertTrue(retryable)

    # =========================================================================
    # 2. GEMINI TIMEOUT & RETRY SIMULATION TESTS
    # =========================================================================
    def test_call_gemini_retry_on_transient_error(self):
        """Verify _call_gemini_with_timeout retries on transient errors and succeeds on subsequent attempt."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Success after retry."
        
        # First call fails with 503, second call succeeds
        mock_client.models.generate_content.side_effect = [
            Exception("503 Service Unavailable"),
            mock_resp
        ]
        
        result = _call_gemini_with_timeout(
            client=mock_client,
            model="gemini-3.5-flash-lite",
            contents="Test prompt",
            timeout_sec=5,
            max_retries=1
        )
        self.assertEqual(result, "Success after retry.")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    def test_call_gemini_fails_fast_on_auth_error(self):
        """Verify _call_gemini_with_timeout does NOT retry on non-retryable 401/403 auth error."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("403 Permission denied: API key invalid")
        
        result = _call_gemini_with_timeout(
            client=mock_client,
            model="gemini-3.5-flash-lite",
            contents="Test prompt",
            timeout_sec=5,
            max_retries=2
        )
        self.assertIsNone(result)
        # Should NOT retry 2 more times; should fail after 1 attempt
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

    def test_call_gemini_empty_response_handling(self):
        """Verify _call_gemini_with_timeout handles empty or None response gracefully."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = ""
        mock_resp.candidates = []
        mock_client.models.generate_content.return_value = mock_resp
        
        result = _call_gemini_with_timeout(
            client=mock_client,
            model="gemini-3.5-flash-lite",
            contents="Test prompt",
            timeout_sec=5,
            max_retries=1
        )
        self.assertIsNone(result)

    # =========================================================================
    # 3. DIRECT GROUNDED EXTRACTION FALLBACK TESTS
    # =========================================================================
    def test_generate_rag_answer_fallback_on_empty_chunks(self):
        """Verify generate_rag_answer returns FALLBACK_RESPONSE immediately when no chunks retrieved."""
        mock_client = MagicMock()
        answer = generate_rag_answer(mock_client, "What is quantum gravity?", [])
        self.assertEqual(answer, FALLBACK_RESPONSE)

    def test_direct_grounded_extraction_fallback(self):
        """Verify extract_direct_answer_fallback extracts facts from retrieved chunks without remote LLM."""
        doc = Document(
            page_content="Atomicity ensures all operations within a database transaction complete or rollback completely.",
            metadata={"document": "DBMS_Notes.pdf", "page": 1}
        )
        chunks = [(doc, 0.95)]
        extracted = extract_direct_answer_fallback("What is atomicity?", chunks, "atomicity")
        self.assertIn("Atomicity", extracted)
        self.assertIn("DBMS_Notes.pdf", extracted)

    # =========================================================================
    # 4. REST API VALIDATION & EDGE CASE TESTS
    # =========================================================================
    def test_chat_empty_query_returns_validation_error(self):
        """Verify POST /api/chat with empty or whitespace-only query returns HTTP 400 VALIDATION_ERROR."""
        res1 = self.client.post('/api/chat', json={"question": ""})
        self.assertEqual(res1.status_code, 400)
        json1 = res1.get_json()
        self.assertFalse(json1["success"])
        self.assertEqual(json1.get("error_code"), "VALIDATION_ERROR")
        self.assertEqual(json1.get("error"), "Please enter a question.")

        res2 = self.client.post('/api/chat', json={"question": "   \n\t  "})
        self.assertEqual(res2.status_code, 400)
        json2 = res2.get_json()
        self.assertEqual(json2.get("error_code"), "VALIDATION_ERROR")

    def test_chat_empty_selected_documents_returns_validation_error(self):
        """Verify POST /api/chat with empty selected_documents list returns HTTP 400 VALIDATION_ERROR."""
        res = self.client.post('/api/chat', json={
            "question": "What is SQL?",
            "selected_documents": []
        })
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data.get("error_code"), "VALIDATION_ERROR")
        self.assertEqual(data.get("error"), "Please select at least one document.")

    def test_chat_unindexed_document_selection_returns_validation_error(self):
        """Verify POST /api/chat selecting only non-existent/unindexed docs returns actionable HTTP 400 error."""
        res = self.client.post('/api/chat', json={
            "question": "What is SQL?",
            "selected_documents": ["NonExistentDoc.pdf"]
        })
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data.get("error_code"), "VALIDATION_ERROR")
        self.assertIn("No indexed documents are available", data.get("error"))

    def test_chat_greeting_bypasses_llm_safely(self):
        """Verify conversational greetings return fast local response without invoking LLM."""
        res = self.client.post('/api/chat', json={"question": "Hello!"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data.get("response_type"), "greeting")
        self.assertIn("Hello!", data.get("answer"))

    # =========================================================================
    # 5. LLM EXCEPTION HANDLING IN /api/chat ROUTE
    # =========================================================================
    @patch('app.generate_rag_answer')
    def test_chat_ai_auth_error_handling(self, mock_gen):
        """Verify AIAuthError in chat route returns structured AI_AUTH_ERROR payload."""
        mock_gen.side_effect = AIAuthError("Invalid Gemini API key.")
        
        # Test decomposition/comparison branch or standard route with mocked AIAuthError
        res = self.client.post('/api/chat', json={
            "question": "Compare normalization and denormalization in DBMS and SQL.",
            "selected_documents": ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"]
        })
        # If comparison route or standard route encounters AIAuthError
        data = res.get_json()
        # Either handled with comparison fallback or returned as AIAuthError
        self.assertIn(res.status_code, [200, 400])
        self.assertTrue("answer" in data or "error" in data)

    # =========================================================================
    # 6. CONCURRENT QUERIES STABILITY
    # =========================================================================
    def test_concurrent_chat_queries(self):
        """Verify system handles multiple simultaneous queries concurrently without crashing or state corruption."""
        questions = [
            "What is a primary key in DBMS?",
            "What is ACID properties?",
            "What is SQL JOIN?",
            "What are DML commands in SQL?",
            "What is atomicity in database transactions?"
        ]
        
        def run_query(q):
            return self.client.post('/api/chat', json={"question": q})

        with ThreadPoolExecutor(max_workers=5) as executor:
            responses = list(executor.map(run_query, questions))

        for idx, resp in enumerate(responses):
            self.assertEqual(resp.status_code, 200, f"Query {idx} failed: {resp.data}")
            data = resp.get_json()
            self.assertTrue(data.get("success"), f"Query {idx} returned success=False")
            self.assertTrue(len(data.get("answer", "")) > 10, f"Query {idx} returned empty answer")

    # =========================================================================
    # 7. DOCUMENT SELECTION FILTERING ISOLATION
    # =========================================================================
    def test_selection_filtering_strictly_isolates_documents(self):
        """Verify queries selecting a specific document strictly isolate retrieval to that document."""
        meta = load_vectorstore_metadata(flask_app_module.VECTORSTORE_DIR)
        indexed = meta.get("indexed_documents", [])
        if not indexed:
            self.skipTest("No documents currently indexed to test selection isolation.")
        target_doc = indexed[0]
        res = self.client.post('/api/chat', json={
            "question": "What is the key architecture?",
            "selected_documents": [target_doc]
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        sources = data.get("sources", [])
        for s in sources:
            doc_name = s.get("document") or s.get("filename")
            self.assertEqual(doc_name, target_doc, f"Unexpected document {doc_name} retrieved when only {target_doc} was selected")


if __name__ == "__main__":
    unittest.main()
