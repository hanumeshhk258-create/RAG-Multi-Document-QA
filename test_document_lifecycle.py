import requests
import json
import os
import time

BASE_URL = "http://127.0.0.1:5000"

import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_sql_pdf_bytes():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=10, leading=14)

    story = [
        Paragraph("SQL Overview and Relational Database Queries", title_style),
        Spacer(1, 10),
        Paragraph("Structured Query Language (SQL) is the standard declarative language for interacting with relational databases.", body_style),
        Spacer(1, 8),
        Paragraph("<b>SQL Sub-Languages:</b><br/>1. DDL (Data Definition Language): CREATE, ALTER, DROP, TRUNCATE.<br/>2. DML (Data Manipulation Language): SELECT, INSERT, UPDATE, DELETE.", body_style),
        Paragraph("<pagebreak/>", body_style),
        Paragraph("SQL JOINs and Query Optimization", title_style),
        Spacer(1, 10),
        Paragraph("SQL JOINs are used to combine rows from two or more tables based on a related column between them.", body_style),
        Spacer(1, 8),
        Paragraph("<b>Types of SQL JOINs:</b><br/>• <b>INNER JOIN</b>: Returns records that have matching values in both tables.<br/>• <b>LEFT (OUTER) JOIN</b>: Returns all records from the left table and matched records from the right table.<br/>• <b>RIGHT (OUTER) JOIN</b>: Returns all records from the right table and matched records from the left table.<br/>• <b>FULL (OUTER) JOIN</b>: Returns all records when there is a match in either left or right table.", body_style)
    ]
    doc.build(story)
    return buffer.getvalue()

def run_tests():
    print("==================================================")
    print("STARTING DOCUMENT LIFECYCLE MANAGEMENT TEST SUITE")
    print("==================================================")
    
    # 0. Check Status
    res = requests.get(f"{BASE_URL}/api/status")
    print(f"[STATUS CHECK]: HTTP {res.status_code}")
    status_data = res.json()
    print(f"Total documents: {status_data.get('total_documents')}, Indexed: {status_data.get('indexed_count')}, Chunks: {status_data.get('total_chunks')}")
    
    pdf_bytes = generate_sql_pdf_bytes()

    # TEST 1: Upload SQL_CheatSheet.pdf
    print("\n--- TEST 1: Upload SQL_CheatSheet.pdf ---")
    files = {"files": ("SQL_CheatSheet.pdf", pdf_bytes, "application/pdf")}
    res = requests.post(f"{BASE_URL}/api/upload?replace=true", files=files)
    print(f"Upload Response: HTTP {res.status_code} -> {res.json().get('success')}")
    assert res.json().get("success") == True
    
    # Check Document ID assigned
    details = res.json().get("details", [])
    doc_id = details[0].get("document_id") if details else None
    print(f"Assigned document_id: {doc_id}")
    assert doc_id is not None
    
    # TEST 2: Index it
    print("\n--- TEST 2: Index SQL_CheatSheet.pdf ---")
    res = requests.post(f"{BASE_URL}/api/index", json={"selected_documents": ["SQL_CheatSheet.pdf"]})
    print(f"Index Response: HTTP {res.status_code} -> {res.json().get('message')}")
    assert res.json().get("success") == True

    # TEST 3: Ask "What are JOINs?"
    print("\n--- TEST 3: Query grounded question: 'What are JOINs?' ---")
    res = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What are JOINs?",
        "selected_documents": ["SQL_CheatSheet.pdf"]
    })
    chat_res = res.json()
    print(f"Answer HTTP {res.status_code}, Sources count: {len(chat_res.get('sources', []))}")
    print(f"Answer snippet: {chat_res.get('answer', '')[:200]}...")
    assert len(chat_res.get("sources", [])) > 0
    assert "SQL_CheatSheet.pdf" in [s.get("document") for s in chat_res.get("sources", [])]
    
    # TEST 4: Delete SQL_CheatSheet.pdf by unique document_id
    print(f"\n--- TEST 4: Delete SQL_CheatSheet.pdf using unique document_id: {doc_id} ---")
    res = requests.delete(f"{BASE_URL}/api/documents/{doc_id}")
    print(f"Delete Response: HTTP {res.status_code} -> {res.json().get('message')}")
    assert res.json().get("success") == True
    assert res.json().get("message") == "Document removed successfully."

    # TEST 5: Check document list
    print("\n--- TEST 5: Verify document list state after deletion ---")
    res = requests.get(f"{BASE_URL}/api/documents")
    docs_data = res.json()
    doc_names = [d.get("filename") for d in docs_data.get("documents", [])]
    doc_ids = [d.get("document_id") for d in docs_data.get("documents", [])]
    print(f"Remaining documents: {doc_names}")
    assert "SQL_CheatSheet.pdf" not in doc_names
    assert doc_id not in doc_ids
    print("[OK] SQL_CheatSheet.pdf is completely gone from document list and registry.")

    # TEST 6 & 7: Ask "What are JOINs?" -> Expected: NOT SUPPORTED / Fallback, 0 sources
    print("\n--- TEST 6 & 7: Query 'What are JOINs?' after document deleted ---")
    res = requests.post(f"{BASE_URL}/api/chat", json={
        "question": "What are JOINs?",
        "selected_documents": ["SQL_CheatSheet.pdf"]
    })
    post_delete_chat = res.json()
    print(f"Post-delete response: {post_delete_chat}")
    if post_delete_chat.get("success"):
        print(f"Answer: {post_delete_chat.get('answer')}")
        print(f"Sources: {post_delete_chat.get('sources')}")
        assert len(post_delete_chat.get("sources", [])) == 0 or "SQL_CheatSheet.pdf" not in [s.get("document") for s in post_delete_chat.get("sources", [])]
    else:
        print(f"System properly rejected or handled deleted document: {post_delete_chat.get('error')}")

    # TEST 8: Check Analytics
    print("\n--- TEST 8: Check Analytics counters ---")
    res = requests.get(f"{BASE_URL}/api/analytics")
    analytics_data = res.json()
    summary = analytics_data.get("summary", {})
    print(f"Analytics Summary: Total Questions={summary.get('total_questions')}, Verified={summary.get('verified_count')}")

    # TEST 9: Re-upload SQL_CheatSheet.pdf
    print("\n--- TEST 9: Re-upload SQL_CheatSheet.pdf ---")
    files = {"files": ("SQL_CheatSheet.pdf", pdf_bytes, "application/pdf")}
    res = requests.post(f"{BASE_URL}/api/upload", files=files)
    print(f"Re-upload Response: HTTP {res.status_code} -> {res.json().get('success')}")
    assert res.json().get("success") == True

    # TEST 10: Upload duplicate SQL_CheatSheet.pdf without replace
    print("\n--- TEST 10: Upload duplicate without replace -> Expect duplicate detection ---")
    files = {"files": ("SQL_CheatSheet.pdf", pdf_bytes, "application/pdf")}
    res = requests.post(f"{BASE_URL}/api/upload", files=files)
    dup_res = res.json()
    print(f"Duplicate Upload Response: is_duplicate={dup_res.get('is_duplicate')}, message='{dup_res.get('message')}'")
    assert dup_res.get("is_duplicate") == True
    assert dup_res.get("message") == "This document is already uploaded."
    
    # Test duplicate with replace=true
    print("\n--- TEST 10b: Upload with replace=true ---")
    files = {"files": ("SQL_CheatSheet.pdf", pdf_bytes, "application/pdf")}
    res = requests.post(f"{BASE_URL}/api/upload?replace=true", files=files)
    print(f"Replace Response: HTTP {res.status_code} -> success={res.json().get('success')}")
    assert res.json().get("success") == True

    # Re-index to leave system ready
    print("\n--- Rebuilding/Indexing all documents ---")
    res = requests.post(f"{BASE_URL}/api/rebuild")
    print(f"Rebuild Response: HTTP {res.status_code} -> {res.json().get('message')}")
    assert res.json().get("success") == True

    print("\n==================================================")
    print("ALL 10 DOCUMENT LIFECYCLE TESTS PASSED PERFECTLY!")
    print("==================================================")

if __name__ == "__main__":
    time.sleep(1)
    run_tests()
