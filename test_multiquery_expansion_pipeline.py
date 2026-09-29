import os
import sys
import time
import json

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rag.retriever import (
    search_vector,
    search_bm25,
    fuse_search_results,
    multi_query_hybrid_search,
    rerank_chunks,
    retrieve_relevant_chunks,
    _CROSS_ENCODER,
    VECTOR_WEIGHT,
    BM25_WEIGHT
)
from rag.llm import expand_query_with_gemini, get_llm_client, load_gemini_api_key
from rag.evaluator import evaluate_rag_response
from rag.vectorstore import load_vectorstore, load_vectorstore_metadata
from rag.embeddings import get_embedding_model

def run_tests():
    print("=" * 75)
    print("QUERY EXPANSION + MULTI-QUERY HYBRID RETRIEVAL TEST SUITE")
    print("=" * 75)

    # 1. Load Vectorstore
    print("\n[STEP 1] Loading Embedding Model and Vectorstore...")
    emb = get_embedding_model("all-MiniLM-L6-v2")
    vs = load_vectorstore(emb, folder_path="vectorstore")
    assert vs is not None, "Failed to load FAISS vectorstore!"
    meta = load_vectorstore_metadata("vectorstore")
    print(f"[OK] Vectorstore loaded with {meta.get('total_documents')} documents and {meta.get('total_chunks')} chunks.")

    # 2. Check Gemini Client & Cross-Encoder Status
    print("\n[STEP 2] Verifying Gemini LLM and Cross-Encoder...")
    key = load_gemini_api_key()
    client = None
    try:
        client = get_llm_client(key)
        print("[OK] Gemini client initialized successfully.")
    except Exception as e:
        print(f"[WARN] Gemini client warning: {e}")

    ce_ready = _CROSS_ENCODER.is_available()
    print(f"[OK] Cross-Encoder available: {ce_ready} (Model: {_CROSS_ENCODER.model_name})")

    # 3. Test Direct Query Expansion
    print("\n[STEP 3] Testing Query Expansion with Gemini...")
    sample_q = "What is querying data commands?"
    expanded = expand_query_with_gemini(llm_client=client, query=sample_q, max_expanded=2)
    print(f"  * Original Query: \"{sample_q}\"")
    print(f"  * Generated Queries ({len(expanded)}):")
    for i, eq in enumerate(expanded, 1):
        print(f"    {i}. \"{eq}\"")
    assert len(expanded) >= 1 and len(expanded) <= 3
    assert expanded[0] == sample_q, "First query MUST be original query!"

    # 4. Test Multi-Query Hybrid Search on Required Test Cases
    test_cases = [
        ("TEST 1 (ACID)", "What is ACID?", []),
        ("TEST 2 (Atomicity)", "Explain atomicity in DBMS", []),
        ("TEST 3 (Querying Data)", "What is querying data commands?", []),
        ("TEST 4 (Normalization)", "What are the types of normalization?", []),
        ("TEST 5 (Conversational Follow-up)", "Tell me its advantages", [
            {"role": "user", "content": "What are the types of normalization?"},
            {"role": "assistant", "content": "Normalization in DBMS includes 1NF, 2NF, 3NF, and BCNF to minimize data redundancy."}
        ]),
        ("TEST 6 (Multi-Document Comparison)", "Compare SQL and DBMS", [])
    ]

    for label, query, history in test_cases:
        print("\n" + "=" * 75)
        print(f"{label}: \"{query}\"")
        print("=" * 75)

        t0 = time.perf_counter()

        retrieved_items, q_type, ext_term, rew_query, stats = retrieve_relevant_chunks(
            vectorstore=vs,
            query=query,
            top_k=5,
            chat_history=history,
            llm_client=client,
            api_key=key
        )
        total_time_ms = round((time.perf_counter() - t0) * 1000, 1)

        print(f"  * Strategy: {stats.get('strategy')}")
        print(f"  * Original Query: \"{stats.get('original_query')}\"")
        print(f"  * Rewritten / Resolved Query: \"{stats.get('rewritten_query')}\"")
        print(f"  * Expanded Queries ({len(stats.get('expanded_queries', []))}): {stats.get('expanded_queries')}")
        print(f"  * Dense (FAISS): {stats.get('semantic_candidates')} candidates | Sparse (BM25): {stats.get('keyword_candidates')} candidates")
        print(f"  * Unique Candidates Fused: {stats.get('unique_candidates')} chunks")
        print(f"  * Final Reranked Chunks: {stats.get('final_chunks')} chunks")
        print(f"  * Latency: Expansion={stats.get('query_expansion_ms')}ms | Retrieval={stats.get('retrieval_ms')}ms | Rerank={stats.get('reranking_ms')}ms | Total={total_time_ms}ms")

        assert len(retrieved_items) > 0, f"Retrieval failed for '{query}'"

        # Check metadata on top chunks
        print("\n  * Retrieved Chunks Metadata Table:")
        print(f"    {'Rank':<5} | {'Document':<35} | {'Page':<5} | {'Vector':<8} | {'BM25':<8} | {'Hybrid':<8} | {'Rerank':<8} | {'Source Query'}")
        print("    " + "-" * 105)

        for item in retrieved_items:
            doc = item[0]
            score = item[1]
            rel_pct = item[2]
            meta = doc.metadata
            doc_name = meta.get("document_name") or meta.get("document", "Document")
            page_num = meta.get("page_number") or meta.get("page", 1)
            v_score = meta.get("vector_score", 0.0)
            b_score = meta.get("bm25_score", 0.0)
            h_score = meta.get("hybrid_score", 0.0)
            r_score = meta.get("reranker_score", score)
            f_rank = meta.get("final_rank", 1)
            src_q = meta.get("source_query", query)

            assert "document_name" in meta or "document" in meta
            assert "page_number" in meta or "page" in meta
            assert "vector_score" in meta
            assert "bm25_score" in meta
            assert "hybrid_score" in meta
            assert "reranker_score" in meta
            assert "final_rank" in meta
            assert "source_query" in meta

            print(f"    #{f_rank:<4} | {doc_name:<35} | P{page_num:<4} | {v_score:<8.3f} | {b_score:<8.3f} | {h_score:<8.3f} | {r_score:<8.3f} | \"{src_q[:35]}\"")

        # Evaluate answer quality
        sample_ans = f"Detailed explanation for {query} referencing {doc_name}."
        eval_data = evaluate_rag_response(
            query=query,
            answer=sample_ans,
            retrieved_items=retrieved_items,
            response_time_sec=total_time_ms / 1000.0,
            retrieval_stats=stats
        )
        print(f"\n  * Answer Quality Evaluation:")
        print(f"    - Groundedness: {eval_data.get('groundedness_score')}% ({eval_data.get('groundedness_label')})")
        print(f"    - Retrieval Relevance: {eval_data.get('retrieval_relevance')}%")
        print(f"    - Source Coverage: {eval_data.get('source_coverage')}%")
        print(f"    - Confidence: {eval_data.get('confidence')}")

    # 5. Test Flask REST API Endpoints with Test Client
    print("\n" + "=" * 75)
    print("[STEP 5] Testing Flask REST API Endpoints...")
    print("=" * 75)
    from app import app
    app.config['TESTING'] = True
    client_app = app.test_client()

    # Test GET /api/status
    res = client_app.get('/api/status')
    assert res.status_code == 200
    status_data = res.get_json()
    print(f"[OK] /api/status components: {status_data.get('components')}")
    assert status_data["components"]["hybrid_search"] is True
    assert status_data["components"]["faiss"] is True
    assert status_data["components"]["bm25"] is True
    assert status_data["components"]["cross_encoder"] is True
    assert status_data["components"]["multi_query"] is True

    # Test POST /api/chat with full conversational follow-up
    print("\n[OK] Testing POST /api/chat with follow-up turn...")
    chat_res = client_app.post('/api/chat', json={
        "question": "Tell me its advantages",
        "history": [
            {"role": "user", "content": "What is normalization in DBMS?"},
            {"role": "assistant", "content": "Normalization organizes data to reduce redundancy and improve data integrity."}
        ],
        "selected_documents": ["DBMS_Notes.pdf"]
    })
    assert chat_res.status_code == 200
    chat_data = chat_res.get_json()
    assert chat_data["success"] is True
    print(f"  * Follow-up understood: {chat_data.get('is_followup')}")
    print(f"  * Resolved Query: {chat_data.get('rewritten_query')}")
    print(f"  * Answer Preview: {chat_data.get('answer')[:120]}...")
    print(f"  * Timing Breakdown: {chat_data.get('timing_breakdown')}")

    print("\n" + "=" * 75)
    print("ALL QUERY EXPANSION + MULTI-QUERY RETRIEVAL TESTS PASSED!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
