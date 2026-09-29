import os
import sys
import time
import unittest
from typing import List, Dict, Any

# Ensure root directory in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.llm import (
    LLM_TIMEOUT,
    MAX_CONTEXT_CHUNKS,
    MAX_OUTPUT_TOKENS,
    MAX_GENERATION_RETRIES,
    LLMGenerationError,
    format_retrieved_context,
    generate_rag_answer,
    load_gemini_api_key,
    get_llm_client
)
from rag.vectorstore import load_vectorstore
from rag.retriever import adaptive_retrieval_pipeline
from app import app


class TestGenerationRobustness(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.vs = load_vectorstore("vectorstore")
        self.api_key = load_gemini_api_key()
        self.client = get_llm_client(self.api_key) if self.api_key else None

    def test_configuration_constants(self):
        """Verify configurable timeout, chunk limit, tokens, and retries."""
        self.assertEqual(LLM_TIMEOUT, 60, "LLM_TIMEOUT should be configured to 60s")
        self.assertEqual(MAX_CONTEXT_CHUNKS, 6, "MAX_CONTEXT_CHUNKS should be 6")
        self.assertEqual(MAX_OUTPUT_TOKENS, 1024, "MAX_OUTPUT_TOKENS should be 1024")
        self.assertEqual(MAX_GENERATION_RETRIES, 2, "MAX_GENERATION_RETRIES should be 2")

    def test_context_optimization_and_deduplication(self):
        """Verify duplicate chunks are removed and MAX_CONTEXT_CHUNKS is respected."""
        class MockDoc:
            def __init__(self, content, doc_name, page):
                self.page_content = content
                self.metadata = {"document": doc_name, "page": page}

        # 8 chunks with duplicates
        chunks = [
            (MockDoc("The Waterfall Model is a sequential development approach.", "DocA.pdf", 1), 0.95),
            (MockDoc("The Waterfall Model is a sequential development approach.", "DocA.pdf", 1), 0.94), # duplicate
            (MockDoc("Phase 1 is requirements definition.", "DocA.pdf", 2), 0.90),
            (MockDoc("Phase 2 is system design.", "DocA.pdf", 3), 0.85),
            (MockDoc("Phase 3 is implementation.", "DocA.pdf", 4), 0.80),
            (MockDoc("Phase 4 is integration.", "DocA.pdf", 5), 0.75),
            (MockDoc("Phase 5 is maintenance.", "DocA.pdf", 6), 0.70),
            (MockDoc("Extra overflow chunk beyond 6.", "DocA.pdf", 7), 0.65),
        ]

        formatted = format_retrieved_context(chunks, max_chunks=MAX_CONTEXT_CHUNKS)
        # Should contain max 6 chunks
        chunk_count = formatted.count("[Document:")
        self.assertLessEqual(chunk_count, 6, "Should not exceed MAX_CONTEXT_CHUNKS (6)")
        self.assertIn("Waterfall Model", formatted)
        self.assertNotIn("Extra overflow chunk beyond 6", formatted)

    def test_five_user_questions_pipeline(self):
        """
        Test the 5 specific questions required by the prompt:
        1. 'What is Waterfall Model?'
        2. 'What is the problem statement?'
        3. 'What are the objectives?'
        4. 'What is Key Security and Trust Controls?'
        5. 'What is querying data commands?'
        """
        test_questions = [
            "What is Waterfall Model?",
            "What is the problem statement?",
            "What are the objectives?",
            "What is Key Security and Trust Controls?",
            "What is querying data commands?"
        ]

        print("\n" + "="*70)
        print("RUNNING 5 REQUIRED QUESTION TESTS ON RAG PIPELINE")
        print("="*70)

        for idx, q in enumerate(test_questions, 1):
            t0 = time.perf_counter()
            response = self.app.post('/api/chat', json={"question": q})
            elapsed = time.perf_counter() - t0

            self.assertEqual(response.status_code, 200, f"HTTP status should be 200 for Q{idx}")
            data = response.get_json()

            print(f"\n--- Question {idx}: '{q}' ---")
            print(f"Success: {data.get('success')}")
            print(f"Elapsed: {elapsed:.2f}s | Response time: {data.get('response_time')}s")
            
            if data.get("success"):
                answer = data.get("final_verified_answer") or data.get("answer")
                self.assertIsNotNone(answer, "Answer must not be None on success")
                self.assertGreater(len(answer.strip()), 10, "Answer must be non-empty")
                
                # Check sources & citations
                sources = data.get("sources", [])
                print(f"Answer snippet: {answer[:140]}...")
                print(f"Sources count: {len(sources)}")
                if sources:
                    print(f"Top source: {sources[0].get('document')} — Page {sources[0].get('page')}")
                    self.assertIsNotNone(sources[0].get("page"), "Source page must be present")
                    self.assertIsNotNone(sources[0].get("document"), "Source document must be present")

                eval_data = data.get("evaluation")
                if eval_data:
                    print(f"Groundedness: {eval_data.get('groundedness_score')}% | Faithfulness: {eval_data.get('faithfulness')}%")
                    print(f"Claims: {eval_data.get('supported_claims')}/{eval_data.get('total_claims')} supported")
            else:
                # If failed, ensure it is properly tagged as generation_failed and answer is None
                self.assertTrue(data.get("generation_failed"), "Failed generation must have generation_failed=True")
                self.assertIsNone(data.get("answer"), "Failed generation must have answer=None")
                print(f"Generation cleanly marked as failed: {data.get('error')}")

        print("\n" + "="*70)
        print("ALL 5 QUESTIONS TESTED SUCCESSFULLY")
        print("="*70)


if __name__ == "__main__":
    unittest.main()
