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
    rerank_chunks,
    retrieve_relevant_chunks,
    _CROSS_ENCODER,
    VECTOR_WEIGHT,
    BM25_WEIGHT
)
from rag.evaluator import evaluate_rag_response
from rag.vectorstore import load_vectorstore, load_vectorstore_metadata
from rag.embeddings import get_embedding_model

def run_tests():
    print("=" * 70)
    print("HYBRID SEARCH + RERANKING COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    # 1. Load Vectorstore
    print("\n[STEP 1] Loading Embedding Model and Vectorstore...")
    emb = get_embedding_model("all-MiniLM-L6-v2")
    vs = load_vectorstore(emb, folder_path="vectorstore")
    assert vs is not None, "Failed to load FAISS vectorstore!"
    meta = load_vectorstore_metadata("vectorstore")
    print(f"[OK] Vectorstore loaded with {meta.get('total_documents')} documents and {meta.get('total_chunks')} chunks.")

    # 2. Verify Cross-Encoder Status
    print("\n[STEP 2] Verifying Cross-Encoder Status...")
    ce_ready = _CROSS_ENCODER.is_available()
    print(f"[OK] Cross-Encoder available: {ce_ready} (Model: {_CROSS_ENCODER.model_name})")

    # 3. Test Retrieval Pipeline on Required Queries
    test_queries = [
        "What is ACID?",
        "What is normalization?",
        "What is SQL JOIN?",
        "Give the exact objectives mentioned in CommerceOS.",
        "Compare SQL and DBMS."
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print(f"\n" + "-" * 70)
        print(f"TEST {q_idx}: '{query}'")
        print("-" * 70)

        t0 = time.perf_counter()
        
        # A. Vector Search
        faiss_results = search_vector(vectorstore=vs, query=query, top_k=15)
        print(f"  [1] FAISS Dense: Retrieved {len(faiss_results)} candidates.")
        assert len(faiss_results) > 0, f"FAISS returned 0 results for '{query}'"

        # B. BM25 Search
        bm25_results, matched_terms = search_bm25(vectorstore=vs, query=query, top_k=15)
        print(f"  [2] BM25 Sparse: Retrieved {len(bm25_results)} candidates (Matched: {matched_terms}).")

        # C. Score Fusion
        fused = fuse_search_results(
            semantic_results=faiss_results,
            keyword_results=bm25_results,
            vector_weight=VECTOR_WEIGHT,
            bm25_weight=BM25_WEIGHT,
            top_k=20
        )
        print(f"  [3] Hybrid Fusion: {len(fused)} candidate chunks fused (Weights: {VECTOR_WEIGHT} FAISS, {BM25_WEIGHT} BM25).")
        assert len(fused) > 0, f"Fusion returned 0 candidates for '{query}'"

        # D. Reranking
        t_rerank_start = time.perf_counter()
        final_top = rerank_chunks(query=query, candidate_chunks=fused, top_k=5)
        t_rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000, 1)
        t_total_ms = round((time.perf_counter() - t0) * 1000, 1)

        print(f"  [4] Cross-Encoder Reranking: Top {len(final_top)} final chunks selected in {t_rerank_ms}ms (Total: {t_total_ms}ms).")
        assert len(final_top) > 0, f"Reranker returned 0 chunks for '{query}'"

        # E. Verify Metadata on all top chunks
        print("\n  [5] Retrieved Chunks Metadata Verification:")
        print(f"      {'Rank':<5} | {'Document':<35} | {'Page':<5} | {'Vector':<8} | {'BM25':<8} | {'Hybrid':<8} | {'Rerank':<8}")
        print("      " + "-" * 85)
        for cand in final_top:
            doc_name = cand["document_name"]
            page_num = cand["page_number"]
            v_score = cand["vector_score"]
            b_score = cand["bm25_score"]
            h_score = cand["hybrid_score"]
            r_score = cand["reranker_score"]
            f_rank = cand["final_rank"]

            # Assert all required metadata fields exist and are valid
            assert "document_name" in cand
            assert "page_number" in cand
            assert "chunk_id" in cand
            assert "chunk_text" in cand or "text" in cand
            assert "vector_score" in cand
            assert "bm25_score" in cand
            assert "hybrid_score" in cand
            assert "reranker_score" in cand
            assert "final_rank" in cand

            print(f"      #{f_rank:<4} | {doc_name:<35} | P{page_num:<4} | {v_score:<8.3f} | {b_score:<8.3f} | {h_score:<8.3f} | {r_score:<8.3f}")

        # F. Test End-to-End retrieve_relevant_chunks
        retrieved_items, query_type, ext_term, rew_query, stats = retrieve_relevant_chunks(
            vectorstore=vs,
            query=query,
            top_k=5
        )
        print(f"\n  [6] End-to-End Pipeline Stats:")
        print(f"      * Strategy: {stats.get('method')}")
        print(f"      * FAISS: {stats.get('dense_method')} ({stats.get('semantic_candidates')} candidates)")
        print(f"      * BM25: {stats.get('sparse_method')} ({stats.get('keyword_candidates')} candidates)")
        print(f"      * Reranker: {stats.get('reranker_method')} ({stats.get('final_chunks')} chunks)")
        print(f"      * Latency: Retrieval={stats.get('retrieval_ms')}ms | Reranking={stats.get('reranking_ms')}ms")

        # G. Test Answer Quality Evaluation based on Reranker Results
        sample_ans = f"This is an answer for {query} referencing {final_top[0]['document_name']}."
        eval_data = evaluate_rag_response(
            query=query,
            answer=sample_ans,
            retrieved_items=retrieved_items,
            response_time_sec=t_total_ms / 1000.0,
            retrieval_stats=stats
        )
        print(f"  [7] Answer Quality Metrics:")
        print(f"      * Groundedness: {eval_data.get('groundedness_score')}% ({eval_data.get('groundedness_label')})")
        print(f"      * Retrieval Relevance: {eval_data.get('retrieval_relevance')}%")
        print(f"      * Source Coverage: {eval_data.get('source_coverage')}%")
        print(f"      * Confidence: {eval_data.get('confidence')}")

    # 4. Test Flask REST API Endpoints with Test Client
    print("\n" + "=" * 70)
    print("[STEP 4] Testing Flask REST API Endpoints...")
    print("=" * 70)
    from app import app
    app.config['TESTING'] = True
    client = app.test_client()

    # Test GET /api/status
    res = client.get('/api/status')
    assert res.status_code == 200
    status_data = res.get_json()
    print(f"[OK] /api/status returned: vectorstore_ready={status_data.get('vectorstore_ready')}, components={status_data.get('components')}")
    assert status_data["components"]["hybrid_search"] is True
    assert status_data["components"]["faiss"] is True
    assert status_data["components"]["bm25"] is True
    assert status_data["components"]["cross_encoder"] is True

    # Test POST /api/chat with document filtering
    print("\n[OK] Testing POST /api/chat with single document filter (DBMS_Notes.pdf)...")
    chat_res = client.post('/api/chat', json={
        "question": "What is ACID?",
        "selected_documents": ["DBMS_Notes.pdf"]
    })
    assert chat_res.status_code == 200
    chat_data = chat_res.get_json()
    assert chat_data["success"] is True
    print(f"  * Answer: {chat_data['answer'][:80]}...")
    print(f"  * Sources ({len(chat_data.get('sources', []))}): {[s['document'] for s in chat_data.get('sources', [])]}")
    assert all(s["document"] == "DBMS_Notes.pdf" for s in chat_data.get("sources", []))

    # Test POST /api/search (Fast Document Search)
    print("\n[OK] Testing POST /api/search (Fast hybrid search without LLM)...")
    search_res = client.post('/api/search', json={
        "query": "ACID",
        "selected_documents": ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"]
    })
    assert search_res.status_code == 200
    search_data = search_res.get_json()
    assert search_data["success"] is True
    print(f"  * Fast Search returned {len(search_data.get('results', []))} results in {search_data.get('elapsed_seconds')}s.")

    # Test PDF Viewer Endpoint
    print("\n[OK] Testing PDF Viewer inline endpoint (/view_pdf/DBMS_Notes.pdf)...")
    pdf_res = client.get('/view_pdf/DBMS_Notes.pdf')
    assert pdf_res.status_code == 200
    assert pdf_res.mimetype == 'application/pdf'
    print(f"  * PDF serve returned HTTP {pdf_res.status_code} with MIME: {pdf_res.mimetype}")

    print("\n" + "=" * 70)
    print("ALL HYBRID SEARCH + RERANKING PIPELINE TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
