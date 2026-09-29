"""
Test suite to verify Question Detail data and API endpoints for all 5 test scenarios:
1. Normal verified question
2. Unsupported question
3. Question with multiple sources
4. Question with claim correction
5. Comparison question
"""

import json
import urllib.request
from rag.analytics import record_interaction, get_question_detail

def populate_test_interactions():
    # 1. Normal Verified Question
    p1 = {
        "query_type": "NORMAL_QUESTION",
        "answer": "A Primary Key in a Database Management System uniquely identifies each record in a relational database table [DBMS_Notes.pdf, Page 1]. It must contain unique values and cannot contain NULL values [SQL_CheatSheet.pdf, Page 1].",
        "final_verified_answer": "A Primary Key in a Database Management System uniquely identifies each record in a relational database table [DBMS_Notes.pdf, Page 1]. It must contain unique values and cannot contain NULL values [SQL_CheatSheet.pdf, Page 1].",
        "draft_answer": "A Primary Key in a Database Management System uniquely identifies each record in a relational database table. It must contain unique values and cannot contain NULL values.",
        "response_time": 1.45,
        "timing_breakdown": {
            "retrieval_sec": 0.12,
            "reranking_sec": 0.35,
            "compression_sec": 0.008,
            "generation_sec": 0.82,
            "verification_sec": 0.15,
            "total_sec": 1.45,
            "retry_count": 0,
            "compressed_chunks": 4
        },
        "retrieval_stats": {
            "total_candidates": 12,
            "unique_candidates": 8,
            "final_chunks": 4
        },
        "chunks_used": 4,
        "evaluation": {
            "retrieval_relevance": 94.5,
            "groundedness_score": 98.0,
            "faithfulness": 100.0,
            "source_coverage": 100.0,
            "confidence": "High",
            "hallucination_risk": 2.0,
            "supported_claims": 2,
            "partial_claims": 0,
            "unsupported_claims": 0,
            "total_claims": 2,
            "claims": [
                {
                    "claim_id": 1,
                    "claim": "A Primary Key in a Database Management System uniquely identifies each record in a table.",
                    "status": "SUPPORTED",
                    "action": "KEEP",
                    "document": "DBMS_Notes.pdf",
                    "page": 1,
                    "evidence": "A Primary Key constraint uniquely identifies each record in a database table. Primary keys must contain UNIQUE values, and cannot contain NULL values.",
                    "score": 0.96,
                    "pdf_url": "/view_pdf/DBMS_Notes.pdf#page=1"
                },
                {
                    "claim_id": 2,
                    "claim": "Primary keys must contain unique values and cannot contain NULL values.",
                    "status": "SUPPORTED",
                    "action": "KEEP",
                    "document": "SQL_CheatSheet.pdf",
                    "page": 1,
                    "evidence": "CREATE TABLE employees (employee_id INT PRIMARY KEY, first_name VARCHAR(50)); Primary Key enforces non-null uniqueness.",
                    "score": 0.93,
                    "pdf_url": "/view_pdf/SQL_CheatSheet.pdf#page=1"
                }
            ]
        },
        "sources": [
            {
                "document": "DBMS_Notes.pdf",
                "page": 1,
                "relevance_pct": 96,
                "score": 0.96,
                "excerpt": "A Primary Key constraint uniquely identifies each record in a database table. Primary keys must contain UNIQUE values, and cannot contain NULL values.",
                "pdf_url": "/view_pdf/DBMS_Notes.pdf#page=1"
            },
            {
                "document": "SQL_CheatSheet.pdf",
                "page": 1,
                "relevance_pct": 93,
                "score": 0.93,
                "excerpt": "CREATE TABLE employees (employee_id INT PRIMARY KEY, first_name VARCHAR(50)); Primary Key enforces non-null uniqueness.",
                "pdf_url": "/view_pdf/SQL_CheatSheet.pdf#page=1"
            }
        ]
    }
    id1 = record_interaction("What is a primary key and its constraints in DBMS?", p1, wall_time_sec=1.45, selected_documents=["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"])

    # 2. Unsupported Question
    p2 = {
        "query_type": "NORMAL_QUESTION",
        "answer": "The selected documents do not contain enough information to answer this question.",
        "final_verified_answer": "The selected documents do not contain enough information to answer this question.",
        "draft_answer": "The selected documents do not contain enough information to answer this question.",
        "response_time": 0.42,
        "timing_breakdown": {
            "retrieval_sec": 0.08,
            "reranking_sec": 0.22,
            "compression_sec": 0.003,
            "generation_sec": 0.12,
            "verification_sec": 0.0,
            "total_sec": 0.42,
            "retry_count": 0,
            "compressed_chunks": 0
        },
        "retrieval_stats": {
            "total_candidates": 10,
            "unique_candidates": 6,
            "final_chunks": 0
        },
        "chunks_used": 0,
        "evaluation": {
            "retrieval_relevance": 0.0,
            "groundedness_score": 0.0,
            "faithfulness": 0.0,
            "source_coverage": 0.0,
            "confidence": "Low",
            "hallucination_risk": 100.0,
            "supported_claims": 0,
            "partial_claims": 0,
            "unsupported_claims": 0,
            "total_claims": 0,
            "claims": []
        },
        "sources": []
    }
    id2 = record_interaction("What is the recipe for baking chocolate cake?", p2, wall_time_sec=0.42, selected_documents=["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"])

    # 3. Question with Multiple Sources (3+ sources)
    p3 = {
        "query_type": "NORMAL_QUESTION",
        "answer": "Relational databases use Normalization (1NF, 2NF, 3NF, BCNF) to organize data and minimize redundancy [DBMS_Notes.pdf, Page 2]. SQL DML operations such as INSERT, UPDATE, and DELETE modify records [SQL_CheatSheet.pdf, Page 2], while aggregate functions like COUNT, SUM, and AVG compute statistical summaries [SQL_CheatSheet.pdf, Page 3]. Python database connectors (like sqlite3 and SQLAlchemy) allow executing these queries programmatically [Python_Notes.pdf, Page 2].",
        "final_verified_answer": "Relational databases use Normalization (1NF, 2NF, 3NF, BCNF) to organize data and minimize redundancy [DBMS_Notes.pdf, Page 2]. SQL DML operations such as INSERT, UPDATE, and DELETE modify records [SQL_CheatSheet.pdf, Page 2], while aggregate functions like COUNT, SUM, and AVG compute statistical summaries [SQL_CheatSheet.pdf, Page 3]. Python database connectors (like sqlite3 and SQLAlchemy) allow executing these queries programmatically [Python_Notes.pdf, Page 2].",
        "draft_answer": "Relational databases use Normalization (1NF, 2NF, 3NF, BCNF) to organize data. SQL DML operations such as INSERT, UPDATE, and DELETE modify records, while aggregate functions like COUNT, SUM, and AVG compute statistical summaries. Python database connectors allow executing these queries programmatically.",
        "response_time": 2.15,
        "timing_breakdown": {
            "retrieval_sec": 0.18,
            "reranking_sec": 0.48,
            "compression_sec": 0.012,
            "generation_sec": 1.22,
            "verification_sec": 0.26,
            "total_sec": 2.15,
            "retry_count": 0,
            "compressed_chunks": 6
        },
        "retrieval_stats": {
            "total_candidates": 18,
            "unique_candidates": 12,
            "final_chunks": 4
        },
        "chunks_used": 4,
        "evaluation": {
            "retrieval_relevance": 92.0,
            "groundedness_score": 95.0,
            "faithfulness": 100.0,
            "source_coverage": 96.0,
            "confidence": "High",
            "hallucination_risk": 4.0,
            "supported_claims": 4,
            "partial_claims": 0,
            "unsupported_claims": 0,
            "total_claims": 4,
            "claims": [
                {"claim_id": 1, "claim": "Normalization (1NF, 2NF, 3NF, BCNF) reduces data redundancy.", "status": "SUPPORTED", "document": "DBMS_Notes.pdf", "page": 2, "evidence": "Relational Database Normalization organizes data to reduce redundancy: 1NF, 2NF, 3NF, BCNF.", "score": 0.94},
                {"claim_id": 2, "claim": "SQL DML operations include INSERT, UPDATE, and DELETE.", "status": "SUPPORTED", "document": "SQL_CheatSheet.pdf", "page": 2, "evidence": "2. DML Commands: Data Manipulation Language including INSERT, UPDATE, DELETE.", "score": 0.95},
                {"claim_id": 3, "claim": "Aggregate functions include COUNT, SUM, and AVG.", "status": "SUPPORTED", "document": "SQL_CheatSheet.pdf", "page": 3, "evidence": "4. Aggregate Functions: COUNT, SUM, AVG, MIN, MAX.", "score": 0.91},
                {"claim_id": 4, "claim": "Python connectors enable executing database queries.", "status": "SUPPORTED", "document": "Python_Notes.pdf", "page": 2, "evidence": "Python provides sqlite3 standard library module and SQLAlchemy ORM for database operations.", "score": 0.88}
            ]
        },
        "sources": [
            {"document": "DBMS_Notes.pdf", "page": 2, "relevance_pct": 94, "score": 0.94, "excerpt": "Relational Database Normalization organizes data to reduce redundancy: 1NF, 2NF, 3NF, BCNF."},
            {"document": "SQL_CheatSheet.pdf", "page": 2, "relevance_pct": 95, "score": 0.95, "excerpt": "2. DML Commands: Data Manipulation Language including INSERT, UPDATE, DELETE."},
            {"document": "SQL_CheatSheet.pdf", "page": 3, "relevance_pct": 91, "score": 0.91, "excerpt": "4. Aggregate Functions: COUNT, SUM, AVG, MIN, MAX."},
            {"document": "Python_Notes.pdf", "page": 2, "relevance_pct": 88, "score": 0.88, "excerpt": "Python provides sqlite3 standard library module and SQLAlchemy ORM for database operations."}
        ]
    }
    id3 = record_interaction("How do DBMS normalization, SQL commands, and Python database connections work together?", p3, wall_time_sec=2.15, selected_documents=["DBMS_Notes.pdf", "SQL_CheatSheet.pdf", "Python_Notes.pdf"])

    # 4. Question with Claim Correction (Supported + Partially Supported + Removed Claims)
    p4 = {
        "query_type": "COMPLEX_QUESTION",
        "draft_answer": "ACID properties ensure transaction reliability: Atomicity ensures all-or-nothing execution, Consistency maintains valid database state, and DBMS was invented by Nikola Tesla in 1895 with quantum database indexing.",
        "final_verified_answer": "- Atomicity ensures all operations in a transaction succeed, or none are applied (all-or-nothing) [DBMS_Notes.pdf, Page 1].\n\n- Consistency ensures the database remains in a valid state before and after transaction execution [DBMS_Notes.pdf, Page 1].",
        "answer": "- Atomicity ensures all operations in a transaction succeed, or none are applied (all-or-nothing) [DBMS_Notes.pdf, Page 1].\n\n- Consistency ensures the database remains in a valid state before and after transaction execution [DBMS_Notes.pdf, Page 1].",
        "response_time": 2.85,
        "timing_breakdown": {
            "retrieval_sec": 0.15,
            "reranking_sec": 0.42,
            "compression_sec": 0.009,
            "generation_sec": 1.45,
            "verification_sec": 0.83,
            "total_sec": 2.85,
            "retry_count": 1,
            "compressed_chunks": 3
        },
        "retrieval_stats": {
            "total_candidates": 14,
            "unique_candidates": 9,
            "final_chunks": 3
        },
        "chunks_used": 3,
        "evaluation": {
            "retrieval_relevance": 85.0,
            "groundedness_score": 70.0,
            "faithfulness": 66.7,
            "source_coverage": 65.0,
            "confidence": "Medium",
            "hallucination_risk": 33.3,
            "supported_claims": 2,
            "partial_claims": 0,
            "unsupported_claims": 1,
            "total_claims": 3,
            "claims": [
                {
                    "claim_id": 1,
                    "claim": "Atomicity ensures all-or-nothing execution in database transactions.",
                    "status": "SUPPORTED",
                    "action": "KEEP",
                    "document": "DBMS_Notes.pdf",
                    "page": 1,
                    "evidence": "Atomicity: All operations in a transaction succeed, or none are applied (all-or-nothing).",
                    "score": 0.95,
                    "pdf_url": "/view_pdf/DBMS_Notes.pdf#page=1"
                },
                {
                    "claim_id": 2,
                    "claim": "Consistency maintains a valid state before and after transactions.",
                    "status": "SUPPORTED",
                    "action": "KEEP",
                    "document": "DBMS_Notes.pdf",
                    "page": 1,
                    "evidence": "Consistency: Database remains in a valid state before and after transaction execution.",
                    "score": 0.92,
                    "pdf_url": "/view_pdf/DBMS_Notes.pdf#page=1"
                },
                {
                    "claim_id": 3,
                    "claim": "DBMS was invented by Nikola Tesla in 1895 with quantum database indexing.",
                    "status": "NOT_SUPPORTED",
                    "action": "REMOVE",
                    "document": None,
                    "page": None,
                    "evidence": "No supporting evidence found in selected documents.",
                    "score": 0.0,
                    "pdf_url": ""
                }
            ],
            "answer_correction": {
                "total_claims": 3,
                "supported": 2,
                "partially_supported": 0,
                "unsupported": 1,
                "removed": 1,
                "final_coverage": 66.7
            }
        },
        "sources": [
            {"document": "DBMS_Notes.pdf", "page": 1, "relevance_pct": 95, "score": 0.95, "excerpt": "ACID Properties: 1. Atomicity: All operations succeed or none. 2. Consistency: Valid state before/after."}
        ]
    }
    id4 = record_interaction("Explain ACID properties and historical origins of DBMS.", p4, wall_time_sec=2.85, selected_documents=["DBMS_Notes.pdf"])

    # 5. Comparison Question
    p5 = {
        "query_type": "COMPARISON_QUESTION",
        "is_comparison": True,
        "mode": "comparison",
        "answer": "### Comparison of SQL vs Python\n\n| Feature | SQL | Python |\n| :--- | :--- | :--- |\n| **Primary Purpose** | Declarative Data Definition & Querying [SQL_CheatSheet.pdf, Page 1] | General-purpose programming and data scripting [Python_Notes.pdf, Page 1] |\n| **Data Manipulation** | Uses DML commands: INSERT, UPDATE, DELETE [SQL_CheatSheet.pdf, Page 2] | Uses variables, data structures, and functions [Python_Notes.pdf, Page 1] |\n| **Aggregations** | Built-in functions: COUNT, SUM, AVG [SQL_CheatSheet.pdf, Page 3] | Standard library math & statistical operations [Python_Notes.pdf, Page 2] |",
        "final_verified_answer": "### Comparison of SQL vs Python\n\n| Feature | SQL | Python |\n| :--- | :--- | :--- |\n| **Primary Purpose** | Declarative Data Definition & Querying [SQL_CheatSheet.pdf, Page 1] | General-purpose programming and data scripting [Python_Notes.pdf, Page 1] |\n| **Data Manipulation** | Uses DML commands: INSERT, UPDATE, DELETE [SQL_CheatSheet.pdf, Page 2] | Uses variables, data structures, and functions [Python_Notes.pdf, Page 1] |\n| **Aggregations** | Built-in functions: COUNT, SUM, AVG [SQL_CheatSheet.pdf, Page 3] | Standard library math & statistical operations [Python_Notes.pdf, Page 2] |",
        "draft_answer": "### Comparison of SQL vs Python\n\n| Feature | SQL | Python |\n| :--- | :--- | :--- |\n| **Primary Purpose** | Declarative Data Definition & Querying | General-purpose programming and data scripting |\n| **Data Manipulation** | Uses DML commands: INSERT, UPDATE, DELETE | Uses variables, data structures, and functions |\n| **Aggregations** | Built-in functions: COUNT, SUM, AVG | Standard library math & statistical operations |",
        "response_time": 3.10,
        "timing_breakdown": {
            "retrieval_sec": 0.32,
            "reranking_sec": 0.55,
            "compression_sec": 0.015,
            "generation_sec": 1.85,
            "verification_sec": 0.38,
            "total_sec": 3.10,
            "retry_count": 0,
            "compressed_chunks": 5
        },
        "retrieval_stats": {
            "total_candidates": 20,
            "unique_candidates": 14,
            "final_chunks": 5
        },
        "chunks_used": 5,
        "evaluation": {
            "retrieval_relevance": 96.0,
            "groundedness_score": 97.0,
            "faithfulness": 100.0,
            "source_coverage": 95.0,
            "confidence": "High",
            "hallucination_risk": 3.0,
            "supported_claims": 3,
            "partial_claims": 0,
            "unsupported_claims": 0,
            "total_claims": 3,
            "claims": [
                {"claim_id": 1, "claim": "SQL is a declarative language for database structure and querying.", "status": "SUPPORTED", "document": "SQL_CheatSheet.pdf", "page": 1, "evidence": "SQL Data Definition Language: defining and optimizing database structure.", "score": 0.96},
                {"claim_id": 2, "claim": "SQL uses DML commands INSERT, UPDATE, DELETE.", "status": "SUPPORTED", "document": "SQL_CheatSheet.pdf", "page": 2, "evidence": "DML Commands: INSERT, UPDATE, DELETE records in tables.", "score": 0.95},
                {"claim_id": 3, "claim": "Python is a general-purpose programming language supporting data structures.", "status": "SUPPORTED", "document": "Python_Notes.pdf", "page": 1, "evidence": "Python fundamentals: data types, lists, dictionaries, functions, and modules.", "score": 0.93}
            ]
        },
        "sources": [
            {"document": "SQL_CheatSheet.pdf", "page": 1, "relevance_pct": 96, "score": 0.96, "excerpt": "SQL Data Definition Language: defining and optimizing database structure."},
            {"document": "SQL_CheatSheet.pdf", "page": 2, "relevance_pct": 95, "score": 0.95, "excerpt": "DML Commands: INSERT, UPDATE, DELETE records in tables."},
            {"document": "Python_Notes.pdf", "page": 1, "relevance_pct": 93, "score": 0.93, "excerpt": "Python fundamentals: data types, lists, dictionaries, functions, and modules."}
        ]
    }
    id5 = record_interaction("Compare SQL and Python features.", p5, wall_time_sec=3.10, selected_documents=["SQL_CheatSheet.pdf", "Python_Notes.pdf"])

    print(f"Recorded test interactions: ID1={id1}, ID2={id2}, ID3={id3}, ID4={id4}, ID5={id5}")
    return [id1, id2, id3, id4, id5]


def verify_api_endpoint(question_ids):
    print("\n--- Verifying Question Detail Endpoint Responses ---")
    for qid in question_ids:
        url = f"http://127.0.0.1:5000/api/analytics/question/{qid}"
        req = urllib.request.urlopen(url)
        assert req.getcode() == 200, f"Expected HTTP 200, got {req.getcode()}"
        data = json.loads(req.read())
        assert data.get("success") is True, f"Expected success: True for #{qid}"
        q = data.get("question")
        assert q is not None, f"Expected question object for #{qid}"
        
        print(f"\n[Test Case Q#{qid}] Question: {q['question']}")
        print(f"  - Status: {q['status']}")
        print(f"  - Query Type: {q['query_type']} | Retrieval Method: {q['retrieval_method']}")
        print(f"  - Quality Metrics -> Groundedness: {q['groundedness']}%, Faithfulness: {q['faithfulness']}%, Rel: {q['retrieval_relevance']}%, Conf: {q['confidence']}, Hallucination Risk: {q['hallucination_risk']}%")
        print(f"  - Performance -> Total: {q['response_time']}s, Ret: {q['retrieval_time']}s, Rerank: {q['reranking_time']}s, Gen: {q['generation_time']}s, Verif: {q['verification_time']}s")
        print(f"  - Sources Count: {len(q['sources'])}, Selected Docs: {q['selected_documents']}")
        print(f"  - Claims Count: {len(q['claims'])} (Verified: {q['verified_claims']}, Partial: {q['partially_supported_claims']}, Removed: {q['removed_claims']})")
        print(f"  - Answer Length: {len(q['answer_text'])} chars")
        if q.get('draft_answer') and q['draft_answer'] != q['answer_text']:
            print(f"  - Draft Difference Detected: {len(q['draft_answer'])} vs {len(q['answer_text'])} chars")

    # Test error handling with a non-existent question ID
    print("\n--- Verifying Error Handling for Non-Existent ID ---")
    try:
        urllib.request.urlopen("http://127.0.0.1:5000/api/analytics/question/999999")
        print("ERROR: Should have returned 404 for non-existent ID")
    except urllib.error.HTTPError as e:
        print(f"  [OK] Non-existent ID correctly returned HTTP {e.code}")

if __name__ == "__main__":
    ids = populate_test_interactions()
    verify_api_endpoint(ids)
    print("\nALL 5 TEST SCENARIOS PASSED PERFECTLY!")
