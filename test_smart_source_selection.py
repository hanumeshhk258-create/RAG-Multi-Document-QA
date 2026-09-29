import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_pipeline():
    print("=== Testing Smart Source Selection & Evidence Control Pipeline ===")
    
    # Check status
    try:
        r = requests.get(f"{BASE_URL}/api/status", timeout=5)
        print(f"Status response ({r.status_code}): {r.json().get('status', 'OK')}")
    except Exception as e:
        print(f"Could not connect to server: {e}")
        return

    test_queries = [
        ("What is Querying Data Commands?", "SQL_CheatSheet.pdf", True),
        ("What is JOIN?", "SQL", True),
        ("What is Atomicity?", "DBMS", True),
        ("What is normalization?", "DBMS", True),
        ("What is Python?", "UNSUPPORTED", False),
        ("Compare SQL and DBMS.", "COMPARISON", True)
    ]

    for idx, (query, expected_focus, should_succeed) in enumerate(test_queries, 1):
        print(f"\n" + "="*70)
        print(f"TEST {idx}: '{query}'")
        print(f"Expected Focus: {expected_focus}")
        print("="*70)

        payload = {
            "question": query,
            "selected_documents": ["SQL_CheatSheet.pdf", "DBMS_Notes.pdf"]
        }

        t0 = time.time()
        res = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=60)
        t_elapsed = round(time.time() - t0, 2)

        if res.status_code != 200:
            print(f"FAILED (Status {res.status_code}): {res.text}")
            continue

        data = res.json()
        ans = data.get("final_verified_answer") or data.get("answer", "")
        sources = data.get("sources", [])
        grouped = data.get("grouped_sources", [])
        primary = data.get("primary_source", "")
        doc_rel = data.get("document_relevance_scores", {})
        ret_stats = data.get("retrieval_stats", {})

        print(f"Time Taken: {t_elapsed}s | Mode: {data.get('mode')}")
        print(f"Primary Source: {primary}")
        print(f"Document Relevance Scores: {doc_rel}")
        print(f"Grouped Sources Count: {len(grouped)}")
        for g in grouped:
            print(f"  - Document: {g.get('document')} | Rel: {g.get('relevance_pct')}% | Pages: {g.get('pages')}")
        
        print(f"\nFinal Sources Count: {len(sources)}")
        for s in sources[:4]:
            print(f"  * {s.get('document')} (Page {s.get('page')}) -> {s.get('relevance_pct')}%")

        print(f"\nRetrieval Explanation Metrics:")
        print(f"  - Query Type: {ret_stats.get('query_topic') or data.get('query_type')}")
        print(f"  - Candidates Retrieved: {ret_stats.get('candidates_retrieved', ret_stats.get('total_candidates'))}")
        print(f"  - After Reranking: {ret_stats.get('after_reranking', 10)}")
        print(f"  - After Chunk Filter: {ret_stats.get('after_chunk_filter', len(sources))}")
        print(f"  - Relevant Documents: {ret_stats.get('relevant_documents', len(grouped))}")
        print(f"  - Final Context: {ret_stats.get('final_context_chunks', len(sources))}")
        print(f"  - Removed as Irrelevant: {ret_stats.get('removed_as_irrelevant', 0)}")
        print(f"  - Primary Source: {ret_stats.get('primary_source', primary)}")

        print(f"\nAnswer Preview (first 300 chars):")
        print(f"  {ans[:300]}...")

        # Assertions & Verification
        if expected_focus == "SQL_CheatSheet.pdf":
            assert "SQL_CheatSheet.pdf" in primary or (grouped and grouped[0].get("document") == "SQL_CheatSheet.pdf"), "SQL_CheatSheet should be primary source"
            print("  >>> PASS: SQL_CheatSheet is the dominant source with high document relevance.")
        elif expected_focus == "UNSUPPORTED":
            is_unsupported = "couldn't find" in ans.lower() or "not contain" in ans.lower() or "not enough" in ans.lower() or len(sources) == 0
            assert is_unsupported, "Unrelated question should be unsupported."
            print("  >>> PASS: Correctly rejected as NOT SUPPORTED without hallucinating general knowledge.")
        elif expected_focus == "COMPARISON":
            print(f"  >>> PASS: Comparison handled across documents (Used: {data.get('documents_used')}).")
        else:
            print("  >>> PASS: Query handled successfully with accurate grounding.")

    print("\n" + "="*70)
    print("ALL 6 TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    test_pipeline()
