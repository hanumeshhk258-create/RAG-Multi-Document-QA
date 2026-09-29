import urllib.request
import json
import time
import sys

BASE_URL = "http://127.0.0.1:5000"

def request_json(endpoint, method="GET", data=None):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    body = json.dumps(data).encode("utf-8") if data is not None else None
    
    with urllib.request.urlopen(req, data=body, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))

def request_text(endpoint):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8"), resp.headers.get("Content-Type")

def run_tests():
    print("==================================================")
    print("RUNNING RAG ANALYTICS & EVALUATION DASHBOARD SUITE")
    print("==================================================")

    # 1. Check Status
    status = request_json("/api/status")
    print(f"[*] System status: {status.get('status')} | Indexed Chunks: {status.get('total_chunks')} | Documents: {status.get('total_documents')}")

    # 2. Reset Analytics to clean slate
    clear_res = request_json("/api/analytics/clear", method="POST")
    print(f"[*] Analytics cleared: {clear_res.get('success')}")

    # 3. Ask Question 1: Normal Question
    print("\n[1/4] Sending Normal Question: 'What is a primary key in DBMS?'")
    t0 = time.time()
    q1 = request_json("/api/chat", method="POST", data={
        "question": "What is a primary key in DBMS?",
        "selected_documents": ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"]
    })
    print(f"   -> Response time: {round(time.time() - t0, 2)}s | Verified: {q1.get('answer_correction', {}).get('supported', 0)} claims")

    # 4. Ask Question 2: Comparison Question
    print("\n[2/4] Sending Comparison Question: 'Compare SQL and Python.'")
    t0 = time.time()
    q2 = request_json("/api/chat", method="POST", data={
        "question": "Compare SQL and Python.",
        "selected_documents": ["SQL_CheatSheet.pdf", "Python_Notes.pdf"]
    })
    print(f"   -> Response time: {round(time.time() - t0, 2)}s | Comparison mode: {q2.get('mode')}")

    # 5. Ask Question 3: Unsupported Question
    print("\n[3/4] Sending Unsupported Question: 'What is the recipe for chocolate cake with strawberry frosting?'")
    t0 = time.time()
    q3 = request_json("/api/chat", method="POST", data={
        "question": "What is the recipe for chocolate cake with strawberry frosting?",
        "selected_documents": ["SQL_CheatSheet.pdf"]
    })
    print(f"   -> Response time: {round(time.time() - t0, 2)}s | Answer: {q3.get('answer')[:60]}...")

    # 6. Ask Question 4: Follow-up Question
    print("\n[4/4] Sending Follow-up Question: 'What are its types?'")
    t0 = time.time()
    q4 = request_json("/api/chat", method="POST", data={
        "question": "What are its types?",
        "selected_documents": ["DBMS_Notes.pdf", "SQL_CheatSheet.pdf"],
        "conversation": [
            {"role": "user", "content": "What is a primary key in DBMS?"},
            {"role": "assistant", "content": q1.get("answer", "")}
        ]
    })
    print(f"   -> Response time: {round(time.time() - t0, 2)}s | Followup resolved topic: {q4.get('context_topic')}")

    # 7. Check Analytics Dashboard API
    print("\n==================================================")
    print("VERIFYING ANALYTICS DASHBOARD API (/api/analytics)")
    print("==================================================")
    analytics = request_json("/api/analytics")
    assert analytics.get("success") is True, "Analytics API call failed!"
    
    summary = analytics.get("summary", {})
    charts = analytics.get("charts", {})
    history = analytics.get("history", [])

    print(f"[*] Total Questions: {summary.get('total_questions')}")
    print(f"[*] Avg Response Time: {summary.get('avg_response_time')}s")
    print(f"[*] Avg Retrieval Relevance: {summary.get('avg_retrieval_relevance')}%")
    print(f"[*] Avg Groundedness: {summary.get('avg_groundedness')}%")
    print(f"[*] Avg Faithfulness: {summary.get('avg_faithfulness')}%")
    print(f"[*] Avg Source Coverage: {summary.get('avg_source_coverage')}%")
    print(f"[*] Verified Answers Count: {summary.get('verified_count')}")
    print(f"[*] Unsupported Answers Count: {summary.get('unsupported_count')}")
    print(f"[*] Avg Retry Count: {summary.get('avg_retry_count')}")

    assert summary.get("total_questions") >= 4, f"Expected at least 4 questions recorded, got {summary.get('total_questions')}"
    assert len(charts.get("response_times", [])) >= 4, "Expected 4+ chart response time points"
    assert len(history) >= 4, "Expected 4+ history items"

    # 8. Check Question Detail Endpoint
    target_q_id = history[-1]["id"] if history else 1
    print("\n==================================================")
    print(f"VERIFYING QUESTION DETAIL API (/api/analytics/question/{target_q_id})")
    print("==================================================")
    q_detail = request_json(f"/api/analytics/question/{target_q_id}")
    assert q_detail.get("success") is True, "Question detail API failed!"
    q_item = q_detail.get("question", {})
    print(f"[*] Question #{target_q_id}: '{q_item.get('question')}'")
    print(f"[*] Status: {q_item.get('status')}")
    print(f"[*] Timing Breakdown: {q_item.get('timing_breakdown')}")
    print(f"[*] Sources ({len(q_item.get('sources', []))}): {[s.get('document') for s in q_item.get('sources', [])]}")
    print(f"[*] Verified Claims: {q_item.get('verified_claims')} | Removed Claims: {q_item.get('removed_claims')}")

    # 9. Check CSV Export Endpoint
    print("\n==================================================")
    print("VERIFYING CSV EXPORT API (/api/analytics/export)")
    print("==================================================")
    csv_text, content_type = request_text("/api/analytics/export")
    print(f"[*] Content-Type: {content_type}")
    print(f"[*] CSV Preview:\n{csv_text[:350]}...")
    assert "ID,Timestamp,Question" in csv_text, "Invalid CSV header!"
    assert len(csv_text.strip().split("\n")) >= 5, "Expected at least header + 4 rows in CSV!"

    print("\n==================================================")
    print("ALL ANALYTICS DASHBOARD TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
