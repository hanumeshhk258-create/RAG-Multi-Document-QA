"""
Comprehensive Test Suite for Table & Structured Data Extraction RAG Feature
Tests:
1. Table extraction from PDF
2. RAG indexing with table chunks
3. Table retrieval and citations
4. Question answering on structured table content
5. Regression test on standard text content (Waterfall Model)
"""
import sys
import os
import time

# Ensure project root in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.table_extractor import extract_tables_from_pdf, format_markdown_table
from rag.pdf_loader import load_pdf_file, process_multiple_pdfs
from rag.chunker import split_documents
from rag.vectorstore import build_vectorstore
from rag.embeddings import get_embedding_model
from rag.retriever import adaptive_retrieval_pipeline

def run_tests():
    print("=" * 70)
    print("RUNNING TABLE & STRUCTURED DATA EXTRACTION TEST SUITE")
    print("=" * 70)

    # Step 1: Generate table PDF if not exists
    from create_table_pdf import generate_table_pdf
    generate_table_pdf()

    # Step 2: Test extract_tables_from_pdf
    print("\n--- TEST 1: Unit Extraction with pdfplumber ---")
    pdf_path = "documents/ML_Algorithms_Comparison.pdf"
    table_docs, table_metas = extract_tables_from_pdf(pdf_path, "ML_Algorithms_Comparison.pdf")
    print(f"Extracted {len(table_docs)} table document(s) from {pdf_path}")
    assert len(table_docs) > 0, "Expected at least 1 table extracted from ML_Algorithms_Comparison.pdf"
    
    first_tbl = table_docs[0]
    meta = first_tbl.metadata
    print(f"Table Metadata: {meta}")
    assert meta.get("is_table") is True, "Metadata is_table should be True"
    assert "KNN" in first_tbl.page_content, "Expected 'KNN' in table content"
    assert "K-Means" in first_tbl.page_content, "Expected 'K-Means' in table content"
    assert "Algorithm" in meta.get("columns", []), "Expected 'Algorithm' in columns"
    print("[OK] Test 1 Passed: Table extracted with columns, rows, and structured representations.")

    # Step 3: Test loading all PDFs and chunking
    print("\n--- TEST 2: PDF Loader & Chunker Integration ---")
    pdf_files = [f for f in sorted(os.listdir("documents")) if f.lower().endswith('.pdf')]
    raw_docs = []
    for pdf_name in pdf_files:
        pdf_path = os.path.join("documents", pdf_name)
        docs, _ = load_pdf_file(pdf_path, pdf_name)
        raw_docs.extend(docs)

    print(f"Loaded {len(raw_docs)} total document objects from documents/ directory across {len(pdf_files)} PDFs.")
    table_count = sum(1 for d in raw_docs if getattr(d, "metadata", {}).get("is_table", False))
    print(f"Found {table_count} table chunk(s) among loaded documents.")
    assert table_count > 0, "Expected table chunks in loaded documents."

    chunks = split_documents(raw_docs)
    print(f"Total chunks created: {len(chunks)}")
    tbl_chunks = [c for c in chunks if c.metadata.get("is_table", False)]
    print(f"Table chunks preserved in chunker: {len(tbl_chunks)}")
    assert len(tbl_chunks) > 0, "Expected table chunks preserved intact."
    print("[OK] Test 2 Passed: Documents and table chunks loaded and chunked properly.")

    # Step 4: Build Hybrid Index
    print("\n--- TEST 3: Indexing Vector Store ---")
    embed_model = get_embedding_model()
    vectorstore = build_vectorstore(chunks, embed_model)
    assert vectorstore is not None, "FAISS vectorstore should not be None"
    print("[OK] Test 3 Passed: Hybrid vector store successfully built.")

    # Step 5: Test Table Queries
    print("\n--- TEST 4: Table Query 1 ('What is the advantage of KNN?') ---")
    res1 = adaptive_retrieval_pipeline(
        query="What is the advantage of KNN?",
        vectorstore=vectorstore,
        selected_documents=["ML_Algorithms_Comparison.pdf"]
    )
    print(f"Answer: {res1.get('answer')}")
    print(f"Sources: {res1.get('sources')}")
    print(f"Structured Data: {res1.get('structured_data')}")
    assert res1.get("answer") is not None, "Expected an answer"
    assert "simple" in res1.get("answer", "").lower(), "Expected 'simple' in answer for KNN advantage"
    print("[OK] Table Query 1 Passed.")

    print("\n--- TEST 5: Table Query 2 ('Which algorithm is unsupervised?') ---")
    res2 = adaptive_retrieval_pipeline(
        query="Which algorithm is unsupervised?",
        vectorstore=vectorstore,
        selected_documents=["ML_Algorithms_Comparison.pdf"]
    )
    print(f"Answer: {res2.get('answer')}")
    assert res2.get("answer") is not None, "Expected an answer"
    ans_lower = res2.get("answer", "").lower()
    assert "k-means" in ans_lower or "pca" in ans_lower or "dbscan" in ans_lower, "Expected unsupervised algorithm name in answer"
    print("[OK] Table Query 2 Passed.")

    print("\n--- TEST 6: Table Query 3 ('Compare KNN and K-Means.') ---")
    res3 = adaptive_retrieval_pipeline(
        query="Compare KNN and K-Means.",
        vectorstore=vectorstore,
        selected_documents=["ML_Algorithms_Comparison.pdf"]
    )
    print(f"Answer: {res3.get('answer')}")
    ans3_lower = res3.get("answer", "").lower()
    assert res3.get("answer") is not None
    assert "supervised" in ans3_lower or "clustering" in ans3_lower or "knn" in ans3_lower
    print("[OK] Table Query 3 Passed.")

    # Step 6: Test Text-Only Regression Query
    print("\n--- TEST 7: Text-Only Query ('What is Waterfall Model?') ---")
    res4 = adaptive_retrieval_pipeline(
        query="What is Waterfall Model?",
        vectorstore=vectorstore,
        selected_documents=["CommerceOS_Complete_Project_Report.pdf"]
    )
    print(f"Answer: {res4.get('answer')[:200]}...")
    assert res4.get("answer") is not None
    print("[OK] Test 7 Passed: Standard text QA regression verified.")

    print("\n" + "=" * 70)
    print("ALL TABLE EXTRACTION & RAG PIPELINE TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
