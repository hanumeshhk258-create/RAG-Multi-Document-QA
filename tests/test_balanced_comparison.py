"""
tests/test_balanced_comparison.py
Regression test suite for Balanced Multi-Document Comparison Mode.
Verifies fair per-document candidate retrieval, starvation prevention,
relevance safety, and comparison subtype preservation.
"""

import os
import sys
import tempfile
import shutil
from langchain_core.documents import Document

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.embeddings import get_embedding_model
from rag.vectorstore import build_vectorstore, clear_vectorstore
from rag.retriever import (
    detect_comparison_question,
    retrieve_multi_document_context,
    retrieve_relevant_chunks
)


def run_balanced_comparison_tests():
    print("=" * 60)
    print("RUNNING BALANCED MULTI-DOCUMENT COMPARISON REGRESSION SUITE")
    print("=" * 60)

    # 1. Comparison Subtype Detection Tests
    print("\n[TEST 1] Testing Comparison Intent & Subtype Detection...")
    res_diff = detect_comparison_question("What are the key differences between SQL and NoSQL?")
    assert res_diff.is_comparison is True, "Expected DIFFERENCES query to be detected as comparison"
    assert res_diff.subtype == "DIFFERENCES", f"Expected DIFFERENCES, got {res_diff.subtype}"

    res_sim = detect_comparison_question("What are the similarities between these documents?")
    assert res_sim.is_comparison is True, "Expected SIMILARITIES query to be detected as comparison"
    assert res_sim.subtype == "SIMILARITIES", f"Expected SIMILARITIES, got {res_sim.subtype}"

    res_sum = detect_comparison_question("Summarize all the selected documents in detail")
    assert res_sum.is_comparison is True, "Expected SUMMARY_ALL query to be detected as comparison"
    assert res_sum.subtype == "SUMMARY_ALL", f"Expected SUMMARY_ALL, got {res_sum.subtype}"

    res_norm = detect_comparison_question("What is a relational database primary key?")
    assert res_norm.is_comparison is False, "Expected normal query to NOT be detected as comparison"
    print("  Comparison intent and subtypes (DIFFERENCES, SIMILARITIES, SUMMARY_ALL) verified.")

    # Prepare mock chunks across 3 documents
    doc_a_chunks = [
        Document(
            page_content="SQL databases are relational and use structured schemas with tables. "
                         "They guarantee ACID properties: Atomicity, Consistency, Isolation, and Durability. "
                         "Transactions ensure complete integrity for critical banking and enterprise systems.",
            metadata={"document": "doc_a_sql.pdf", "source": "doc_a_sql.pdf", "page": 1, "chunk_id": "a_1"}
        ),
        Document(
            page_content="Relational schemas define foreign keys and strict constraints. "
                         "SQL queries use declarative syntax with SELECT, JOIN, GROUP BY, and HAVING.",
            metadata={"document": "doc_a_sql.pdf", "source": "doc_a_sql.pdf", "page": 2, "chunk_id": "a_2"}
        ),
    ]

    doc_b_chunks = [
        Document(
            page_content="NoSQL databases provide flexible schema designs including document, key-value, "
                         "and graph data models. They scale horizontally across distributed clusters "
                         "and prioritize high availability and partition tolerance over strict immediate consistency.",
            metadata={"document": "doc_b_nosql.pdf", "source": "doc_b_nosql.pdf", "page": 1, "chunk_id": "b_1"}
        ),
        Document(
            page_content="Eventual consistency is a common model in distributed NoSQL databases. "
                         "Popular examples include MongoDB, Cassandra, DynamoDB, and Redis.",
            metadata={"document": "doc_b_nosql.pdf", "source": "doc_b_nosql.pdf", "page": 2, "chunk_id": "b_2"}
        ),
    ]

    doc_c_unrelated_chunks = [
        Document(
            page_content="Photosynthesis is the biological process by which green plants convert sunlight, "
                         "carbon dioxide, and water into glucose and oxygen through chlorophyll pigments in chloroplasts.",
            metadata={"document": "doc_c_biology.pdf", "source": "doc_c_biology.pdf", "page": 1, "chunk_id": "c_1"}
        )
    ]

    all_chunks = doc_a_chunks + doc_b_chunks + doc_c_unrelated_chunks
    doc_names = ["doc_a_sql.pdf", "doc_b_nosql.pdf", "doc_c_biology.pdf"]

    print("\n[SETUP] Initializing embedding model and test FAISS index...")
    embedding_model = get_embedding_model()
    test_vs_dir = tempfile.mkdtemp(prefix="test_balanced_comp_")

    try:
        vectorstore = build_vectorstore(all_chunks, embedding_model, doc_names=doc_names)
        assert vectorstore is not None, "Failed to build test vectorstore"

        # 2. Test A: Two Relevant Documents (Fair Opportunity / Starvation Prevention)
        print("\n[TEST 2] Testing Balanced Retrieval with Two Relevant Documents...")
        query = "Compare data consistency models and architectures in SQL and NoSQL"
        selected = ["doc_a_sql.pdf", "doc_b_nosql.pdf"]

        retrieved_items, grouped_context, q_type, rewritten_q, stats = retrieve_multi_document_context(
            vectorstore=vectorstore,
            query=query,
            comparison_topics=["SQL", "NoSQL", "data consistency"],
            selected_documents=selected,
            candidate_pool_size=16,
            final_k_per_doc=3,
            top_k_final=6,
            comparison_subtype="DIFFERENCES"
        )

        assert q_type == "COMPARISON_QUESTION"
        assert len(retrieved_items) >= 2, f"Expected at least 2 chunks, got {len(retrieved_items)}"
        assert "doc_a_sql.pdf" in grouped_context, "doc_a_sql.pdf must be in grouped context"
        assert "doc_b_nosql.pdf" in grouped_context, "doc_b_nosql.pdf must be in grouped context"
        assert len(grouped_context["doc_a_sql.pdf"]) >= 1, "doc_a_sql.pdf must have >= 1 chunk"
        assert len(grouped_context["doc_b_nosql.pdf"]) >= 1, "doc_b_nosql.pdf must have >= 1 chunk"
        assert stats.get("documents_compared") == 2, f"Expected 2 documents compared, got {stats.get('documents_compared')}"
        assert stats.get("comparison_subtype") == "DIFFERENCES"

        # Verify metadata integrity on all retrieved items
        for item in retrieved_items:
            doc_obj, score, rel_pct = item
            assert doc_obj.metadata.get("document") in selected
            assert "page" in doc_obj.metadata
            assert "chunk_id" in doc_obj.metadata
            assert isinstance(score, float)
            assert isinstance(rel_pct, int)

        print(f"  Balanced retrieval verified: Both docs represented cleanly ({list(grouped_context.keys())}).")
        print(f"  doc_a chunks: {len(grouped_context['doc_a_sql.pdf'])}, doc_b chunks: {len(grouped_context['doc_b_nosql.pdf'])}")

        # 3. Test B: One Relevant Document + One Unrelated Document (Relevance Safety)
        print("\n[TEST 3] Testing Relevance Safety with One Relevant and One Unrelated Document...")
        query_sql_only = "Explain ACID transaction properties and durability guarantees"
        selected_with_unrelated = ["doc_a_sql.pdf", "doc_c_biology.pdf"]

        items_safe, grouped_safe, _, _, stats_safe = retrieve_multi_document_context(
            vectorstore=vectorstore,
            query=query_sql_only,
            comparison_topics=["ACID", "durability"],
            selected_documents=selected_with_unrelated,
            candidate_pool_size=16,
            final_k_per_doc=3,
            top_k_final=6,
            comparison_subtype="COMMON_INFO"
        )

        # No exception should be thrown
        assert "doc_a_sql.pdf" in grouped_safe, "Relevant doc_a_sql.pdf should be retrieved"
        assert len(grouped_safe["doc_a_sql.pdf"]) >= 1
        # Unrelated biology doc must NOT be forced into context with noise
        doc_c_chunks = grouped_safe.get("doc_c_biology.pdf", [])
        assert len(doc_c_chunks) == 0, (
            f"Relevance safety violated: Unrelated doc_c_biology.pdf received {len(doc_c_chunks)} chunks for an ACID query"
        )
        print("  Relevance safety verified: Unrelated document was not injected into comparison context.")

        # 4. Test C: Single Document Selected (Preserves Existing Single-Doc Behavior)
        print("\n[TEST 4] Testing Single Document Selected with Comparison Query...")
        items_single, grouped_single, _, _, _ = retrieve_multi_document_context(
            vectorstore=vectorstore,
            query="Compare table features",
            comparison_topics=[],
            selected_documents=["doc_a_sql.pdf"],
            candidate_pool_size=16,
            final_k_per_doc=3,
            top_k_final=6,
            comparison_subtype="COMPARISON_TABLE"
        )
        assert len(grouped_single) <= 1
        if grouped_single:
            assert list(grouped_single.keys()) == ["doc_a_sql.pdf"]
        print("  Single-document selection preserved exactly.")

        # 5. Test D: Non-Comparison Retrieval Unchanged
        print("\n[TEST 5] Testing Non-Comparison Retrieval Unchanged...")
        chunks_res, q_type, term, rewritten_q, stats_norm = retrieve_relevant_chunks(
            vectorstore=vectorstore,
            query="What is photosynthesis?",
            selected_documents=["doc_c_biology.pdf"],
            top_k=2
        )
        assert len(chunks_res) > 0, "Expected non-comparison retrieval to return chunks"
        top_doc = chunks_res[0][0]
        assert top_doc.metadata.get("document") == "doc_c_biology.pdf"
        assert "photosynthesis" in top_doc.page_content.lower()
        print("  Non-comparison retrieval verified unchanged.")

    finally:
        # Cleanup temporary vectorstore
        if os.path.exists(test_vs_dir):
            shutil.rmtree(test_vs_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("ALL BALANCED COMPARISON REGRESSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_balanced_comparison_tests()
