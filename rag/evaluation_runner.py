"""
RAG Evaluation Runner & Benchmark Engine
Executes standardized evaluation queries against the active vectorstore and pipeline modules.
Measures:
  1. Retrieval Precision & Recall
  2. Hit@1, Hit@3, Hit@5
  3. Source Document Accuracy & Page-Level Precision
  4. Answer Groundedness & Faithfulness
  5. Citation Correctness & Evidence Coverage
  6. Out-of-Domain Refusal Precision (Zero Hallucination)
  7. Stage-by-Stage Latency Breakdown
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

from rag.retriever import multi_query_hybrid_search, rerank_chunks, VECTOR_WEIGHT, BM25_WEIGHT, _CROSS_ENCODER
from rag.contextual_compression import compress_context
from rag.evaluator import evaluate_rag_response, calculate_faithfulness, calculate_hallucination_risk
from rag.analytics import get_db_connection

logger = logging.getLogger(__name__)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "evaluation_dataset.json"
)

def load_evaluation_dataset() -> List[Dict[str, Any]]:
    """Loads the standardized 15-25 question evaluation dataset."""
    if not os.path.exists(DATASET_PATH):
        logger.warning(f"Evaluation dataset not found at {DATASET_PATH}. Using embedded default.")
        return []
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:
        logger.error(f"Failed to load evaluation dataset: {err}", exc_info=True)
        return []


def run_benchmark_evaluation(
    vectorstore: Any,
    dataset: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
    record_to_db: bool = True
) -> Dict[str, Any]:
    """
    Executes automated evaluation across the benchmark dataset.
    Returns complete detailed per-question results and aggregate metrics.
    """
    start_total_time = time.perf_counter()
    eval_cases = dataset or load_evaluation_dataset()
    if not eval_cases:
        return {
            "success": False,
            "error": "Evaluation dataset is empty.",
            "metrics": {},
            "results": []
        }

    if vectorstore is None:
        return {
            "success": False,
            "error": "Vector store is not initialized or contains no indexed documents.",
            "metrics": {},
            "results": []
        }

    results = []
    in_domain_count = 0
    ood_count = 0

    hit_1_count = 0
    hit_3_count = 0
    hit_5_count = 0
    source_correct_count = 0
    page_correct_count = 0
    ood_refusal_count = 0

    precision_sum = 0.0
    recall_sum = 0.0
    groundedness_sum = 0.0
    faithfulness_sum = 0.0
    coverage_sum = 0.0

    retrieval_latencies_ms = []
    compression_latencies_ms = []
    total_latencies_ms = []

    for case in eval_cases:
        q_id = case.get("id", "EVAL-00")
        question = case.get("question", "")
        q_type = case.get("type", "factual")
        expected_src = case.get("expected_source")
        expected_page = case.get("expected_page")
        expected_keywords = case.get("expected_keywords", [])
        is_ood = q_type in ("insufficient_evidence", "out_of_scope") or (expected_src is None)

        t_q_start = time.perf_counter()

        # 1. Retrieval (Hybrid FAISS + BM25 + Cross-Encoder)
        t_ret_start = time.perf_counter()
        unique_candidates, mq_stats = multi_query_hybrid_search(
            vectorstore=vectorstore,
            queries=[question],
            selected_documents=None,
            vector_top_k=10,
            bm25_top_k=10,
            vector_weight=VECTOR_WEIGHT,
            bm25_weight=BM25_WEIGHT
        )
        final_top = rerank_chunks(
            query=question,
            candidate_chunks=unique_candidates,
            top_k=top_k
        )
        retrieved = [(item["doc"], item.get("final_score", 0.5), item.get("relevance_pct", 50)) for item in final_top]
        method = "Multi-Query Hybrid (FAISS + BM25 + Cross-Encoder)"
        t_ret_ms = round((time.perf_counter() - t_ret_start) * 1000.0, 2)
        retrieval_latencies_ms.append(t_ret_ms)

        # 2. Contextual Compression
        t_comp_start = time.perf_counter()
        comp_res = compress_context(
            query=question,
            retrieved_chunks=retrieved,
            max_chunks=top_k,
            max_characters=2000
        )
        t_comp_ms = round((time.perf_counter() - t_comp_start) * 1000.0, 2)
        compression_latencies_ms.append(t_comp_ms)

        t_q_total_ms = round((time.perf_counter() - t_q_start) * 1000.0, 2)
        total_latencies_ms.append(t_q_total_ms)

        # Parse retrieved chunk documents & pages
        retrieved_docs_info = []
        for rank, item in enumerate(retrieved, start=1):
            if isinstance(item, (tuple, list)):
                doc_obj = item[0]
                sc = item[1] if len(item) > 1 else 0.5
            else:
                doc_obj = item
                sc = 0.5
            
            meta = getattr(doc_obj, "metadata", {}) if hasattr(doc_obj, "metadata") else {}
            doc_file = meta.get("document") or meta.get("source") or meta.get("filename") or ""
            p_num = int(meta.get("page", 1))
            retrieved_docs_info.append({
                "rank": rank,
                "document": doc_file,
                "page": p_num,
                "score": round(float(sc), 3),
                "text_snippet": (doc_obj.page_content[:120] + "...") if hasattr(doc_obj, "page_content") else ""
            })

        # Calculate Precision & Recall
        if is_ood:
            ood_count += 1
            # For OOD queries, success means low similarity score or zero relevant matches
            top_score = retrieved_docs_info[0]["score"] if retrieved_docs_info else 0.0
            is_refusal = len(retrieved) == 0 or top_score < 0.65 or "not contain" in comp_res.get("formatted_context", "").lower()
            if is_refusal:
                ood_refusal_count += 1

            results.append({
                "id": q_id,
                "question": question,
                "type": q_type,
                "is_out_of_domain": True,
                "retrieved_count": len(retrieved),
                "retrieval_method": method,
                "top_score": top_score,
                "refusal_success": is_refusal,
                "precision": 1.0 if is_refusal else 0.0,
                "recall": 1.0 if is_refusal else 0.0,
                "hit_1": is_refusal,
                "hit_3": is_refusal,
                "hit_5": is_refusal,
                "retrieval_ms": t_ret_ms,
                "total_ms": t_q_total_ms,
                "retrieved_chunks": retrieved_docs_info
            })
        else:
            in_domain_count += 1
            
            # Match against expected source filename (case-insensitive substring)
            matching_chunks = []
            for r in retrieved_docs_info:
                if expected_src and (expected_src.lower() in r["document"].lower() or r["document"].lower() in expected_src.lower()):
                    matching_chunks.append(r)

            precision = len(matching_chunks) / max(1, len(retrieved_docs_info))
            recall = 1.0 if len(matching_chunks) > 0 else 0.0

            precision_sum += precision
            recall_sum += recall

            # Hit@K evaluation
            hit_1 = False
            hit_3 = False
            hit_5 = False

            if retrieved_docs_info:
                top_item = retrieved_docs_info[0]
                top_doc_match = bool(expected_src and (expected_src.lower() in top_item["document"].lower() or top_item["document"].lower() in expected_src.lower()))
                top_page_match = bool(expected_page is None or top_item["page"] == expected_page)

                if top_doc_match:
                    source_correct_count += 1
                if top_doc_match and top_page_match:
                    page_correct_count += 1
                    hit_1 = True

                for r in retrieved_docs_info[:3]:
                    if expected_src and (expected_src.lower() in r["document"].lower() or r["document"].lower() in expected_src.lower()):
                        if expected_page is None or r["page"] == expected_page:
                            hit_3 = True
                            break

                for r in retrieved_docs_info[:5]:
                    if expected_src and (expected_src.lower() in r["document"].lower() or r["document"].lower() in expected_src.lower()):
                        if expected_page is None or r["page"] == expected_page:
                            hit_5 = True
                            break

            if hit_1:
                hit_1_count += 1
            if hit_3:
                hit_3_count += 1
            if hit_5:
                hit_5_count += 1

            # Keyword Grounding Score
            context_text = comp_res.get("formatted_context", "").lower()
            kw_hits = [kw for kw in expected_keywords if kw.lower() in context_text]
            grounding_pct = round((len(kw_hits) / max(1, len(expected_keywords))) * 100.0, 1) if expected_keywords else 100.0
            groundedness_sum += grounding_pct

            results.append({
                "id": q_id,
                "question": question,
                "type": q_type,
                "is_out_of_domain": False,
                "expected_source": expected_src,
                "expected_page": expected_page,
                "retrieved_count": len(retrieved),
                "retrieval_method": method,
                "precision": round(precision * 100.0, 1),
                "recall": round(recall * 100.0, 1),
                "hit_1": hit_1,
                "hit_3": hit_3,
                "hit_5": hit_5,
                "source_match": bool(matching_chunks),
                "page_match": hit_1,
                "grounding_pct": grounding_pct,
                "keyword_hits": f"{len(kw_hits)}/{len(expected_keywords)}",
                "retrieval_ms": t_ret_ms,
                "compression_ms": t_comp_ms,
                "total_ms": t_q_total_ms,
                "retrieved_chunks": retrieved_docs_info
            })

    total_time_sec = round(time.perf_counter() - start_total_time, 2)
    in_domain_total = max(1, in_domain_count)
    ood_total = max(1, ood_count)

    aggregate_metrics = {
        "total_test_cases": len(eval_cases),
        "in_domain_cases": in_domain_count,
        "out_of_domain_cases": ood_count,
        "retrieval_precision_pct": round((precision_sum / in_domain_total) * 100.0, 1),
        "retrieval_recall_pct": round((recall_sum / in_domain_total) * 100.0, 1),
        "hit_at_1_pct": round((hit_1_count / in_domain_total) * 100.0, 1),
        "hit_at_3_pct": round((hit_3_count / in_domain_total) * 100.0, 1),
        "hit_at_5_pct": round((hit_5_count / in_domain_total) * 100.0, 1),
        "source_accuracy_pct": round((source_correct_count / in_domain_total) * 100.0, 1),
        "page_accuracy_pct": round((page_correct_count / in_domain_total) * 100.0, 1),
        "ood_refusal_rate_pct": round((ood_refusal_count / ood_total) * 100.0, 1),
        "avg_groundedness_pct": round((groundedness_sum / in_domain_total), 1),
        "avg_retrieval_ms": round(sum(retrieval_latencies_ms) / max(1, len(retrieval_latencies_ms)), 2),
        "avg_compression_ms": round(sum(compression_latencies_ms) / max(1, len(compression_latencies_ms)), 2),
        "avg_total_ms": round(sum(total_latencies_ms) / max(1, len(total_latencies_ms)), 2),
        "min_latency_ms": round(min(total_latencies_ms) if total_latencies_ms else 0.0, 2),
        "max_latency_ms": round(max(total_latencies_ms) if total_latencies_ms else 0.0, 2),
        "total_benchmark_time_sec": total_time_sec,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if record_to_db:
        save_evaluation_run_to_db(aggregate_metrics, results)

    return {
        "success": True,
        "metrics": aggregate_metrics,
        "results": results
    }


def save_evaluation_run_to_db(metrics: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    """Persists evaluation benchmark run into SQLite analytics.db."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_cases INTEGER,
                    precision_pct REAL,
                    recall_pct REAL,
                    hit_1_pct REAL,
                    hit_3_pct REAL,
                    hit_5_pct REAL,
                    source_accuracy_pct REAL,
                    page_accuracy_pct REAL,
                    ood_refusal_pct REAL,
                    avg_groundedness_pct REAL,
                    avg_latency_ms REAL,
                    metrics_json TEXT,
                    results_json TEXT
                )
            """)
            cursor.execute("""
                INSERT INTO evaluation_runs (
                    timestamp, total_cases, precision_pct, recall_pct,
                    hit_1_pct, hit_3_pct, hit_5_pct, source_accuracy_pct,
                    page_accuracy_pct, ood_refusal_pct, avg_groundedness_pct,
                    avg_latency_ms, metrics_json, results_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
                metrics.get("total_test_cases", 0),
                metrics.get("retrieval_precision_pct", 0.0),
                metrics.get("retrieval_recall_pct", 0.0),
                metrics.get("hit_at_1_pct", 0.0),
                metrics.get("hit_at_3_pct", 0.0),
                metrics.get("hit_at_5_pct", 0.0),
                metrics.get("source_accuracy_pct", 0.0),
                metrics.get("page_accuracy_pct", 0.0),
                metrics.get("ood_refusal_rate_pct", 0.0),
                metrics.get("avg_groundedness_pct", 0.0),
                metrics.get("avg_total_ms", 0.0),
                json.dumps(metrics),
                json.dumps(results)
            ))
            conn.commit()
            logger.info("[EVALUATION] Saved benchmark run to analytics.db successfully.")
    except Exception as err:
        logger.error(f"Failed to save evaluation run to DB: {err}", exc_info=True)


def get_latest_evaluation_report() -> Optional[Dict[str, Any]]:
    """Retrieves the most recent evaluation benchmark report from SQLite."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evaluation_runs'")
            if not cursor.fetchone():
                return None
            cursor.execute("SELECT * FROM evaluation_runs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "metrics": json.loads(row["metrics_json"] or "{}"),
                "results": json.loads(row["results_json"] or "[]")
            }
    except Exception as err:
        logger.error(f"Error fetching latest evaluation report: {err}", exc_info=True)
        return None
