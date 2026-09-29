import os
import sqlite3
import json
import time
import csv
import io
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analytics.db")

def get_db_connection():
    """Returns a SQLite connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_analytics_db():
    """Initializes the SQLite analytics database table if it doesn't exist."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    question TEXT NOT NULL,
                    query_type TEXT DEFAULT 'NORMAL_QUESTION',
                    response_time REAL DEFAULT 0.0,
                    retrieval_time REAL DEFAULT 0.0,
                    reranking_time REAL DEFAULT 0.0,
                    compression_time REAL DEFAULT 0.0,
                    generation_time REAL DEFAULT 0.0,
                    verification_time REAL DEFAULT 0.0,
                    chunks_retrieved INTEGER DEFAULT 0,
                    chunks_selected INTEGER DEFAULT 0,
                    retrieval_relevance REAL DEFAULT 0.0,
                    groundedness REAL DEFAULT 0.0,
                    faithfulness REAL DEFAULT 100.0,
                    source_coverage REAL DEFAULT 0.0,
                    confidence TEXT DEFAULT 'Medium',
                    hallucination_risk REAL DEFAULT 0.0,
                    retry_count INTEGER DEFAULT 0,
                    verified_claims INTEGER DEFAULT 0,
                    partially_supported_claims INTEGER DEFAULT 0,
                    removed_claims INTEGER DEFAULT 0,
                    total_claims INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'VERIFIED',
                    cache_status TEXT DEFAULT 'MISS',
                    selected_documents TEXT DEFAULT '[]',
                    sources_json TEXT DEFAULT '[]',
                    claims_json TEXT DEFAULT '[]',
                    timing_breakdown_json TEXT DEFAULT '{}',
                    answer_text TEXT DEFAULT '',
                    draft_answer TEXT DEFAULT '',
                    context_topic TEXT DEFAULT '',
                    is_followup INTEGER DEFAULT 0,
                    is_comparison INTEGER DEFAULT 0,
                    is_decomposed INTEGER DEFAULT 0
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    interaction_id INTEGER,
                    msg_id TEXT,
                    feedback TEXT NOT NULL,
                    comment TEXT DEFAULT ''
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON user_feedback(timestamp)")
            conn.commit()
            logger.info("Analytics database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize analytics database: {e}", exc_info=True)

# Initialize on module import
init_analytics_db()


def record_user_feedback(
    interaction_id: Optional[int] = None,
    msg_id: Optional[str] = None,
    feedback: str = "helpful",
    comment: str = ""
) -> bool:
    """Records user feedback (helpful or unhelpful) for a question response."""
    try:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        clean_fb = "helpful" if "help" in feedback.lower() and "un" not in feedback.lower() else "unhelpful"
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_feedback (timestamp, interaction_id, msg_id, feedback, comment)
                VALUES (?, ?, ?, ?, ?)
            """, (ts, interaction_id, msg_id or "", clean_fb, comment or ""))
            conn.commit()
            logger.info(f"[ANALYTICS] Recorded user feedback: '{clean_fb}' for msg {msg_id}")
            return True
    except Exception as err:
        logger.error(f"[ANALYTICS] Failed to record user feedback: {err}", exc_info=True)
        return False


def get_user_feedback_summary() -> Dict[str, Any]:
    """Returns total feedback count, helpful count, unhelpful count, and satisfaction rate."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_feedback,
                    SUM(CASE WHEN feedback = 'helpful' THEN 1 ELSE 0 END) as helpful_count,
                    SUM(CASE WHEN feedback = 'unhelpful' THEN 1 ELSE 0 END) as unhelpful_count
                FROM user_feedback
            """)
            row = cursor.fetchone()
            total = int(row["total_feedback"] or 0)
            helpful = int(row["helpful_count"] or 0)
            unhelpful = int(row["unhelpful_count"] or 0)
            satisfaction = round((helpful / max(1, total)) * 100.0, 1) if total > 0 else 100.0

            return {
                "total_feedback": total,
                "helpful_count": helpful,
                "unhelpful_count": unhelpful,
                "satisfaction_rate": satisfaction
            }
    except Exception as err:
        logger.error(f"[ANALYTICS] Error getting feedback summary: {err}", exc_info=True)
        return {
            "total_feedback": 0,
            "helpful_count": 0,
            "unhelpful_count": 0,
            "satisfaction_rate": 100.0
        }


def record_interaction(
    question: str,
    response_payload: Dict[str, Any],
    wall_time_sec: float = 0.0,
    selected_documents: Optional[List[str]] = None
) -> Optional[int]:
    """
    Safely records a question-answer interaction with all RAG metrics.
    Runs non-blockingly and ensures no exceptions can bubble up or crash chat.
    """
    if not question or not isinstance(response_payload, dict):
        return None

    try:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        query_type = response_payload.get("query_type", "NORMAL_QUESTION")
        
        # Timing metrics
        timing = response_payload.get("timing_breakdown", {})
        resp_time = float(response_payload.get("response_time") or wall_time_sec or 0.0)
        
        ret_time = float(timing.get("retrieval_sec", 0.0))
        if ret_time == 0.0 and "retrieval_ms" in timing:
            ret_time = round(float(timing.get("retrieval_ms", 0.0)) / 1000.0, 3)

        rerank_time = float(timing.get("reranking_sec", 0.0))
        if rerank_time == 0.0 and "reranking_ms" in timing:
            rerank_time = round(float(timing.get("reranking_ms", 0.0)) / 1000.0, 3)

        comp_time = float(timing.get("compression_sec", 0.0))
        if comp_time == 0.0 and "compression_ms" in timing:
            comp_time = round(float(timing.get("compression_ms", 0.0)) / 1000.0, 3)

        gen_time = timing.get("generation_sec", 0.0)
        gen_time_float = float(gen_time) if isinstance(gen_time, (int, float)) else 0.0

        verif_time = timing.get("verification_sec", 0.0)
        verif_time_float = float(verif_time) if isinstance(verif_time, (int, float)) else 0.0

        # Chunks metrics
        ret_stats = response_payload.get("retrieval_stats", {})
        chunks_retrieved = int(ret_stats.get("total_candidates", ret_stats.get("deduped_candidates", response_payload.get("fused_candidates", len(response_payload.get("sources", []))))))
        chunks_selected = int(response_payload.get("chunks_used", len(response_payload.get("sources", []))))

        # Evaluation & Groundedness metrics
        eval_data = response_payload.get("evaluation", {})
        ans_corr = response_payload.get("answer_correction", {}) or eval_data.get("answer_correction", {})

        ret_relevance = float(eval_data.get("retrieval_relevance", 90.0))
        groundedness = float(eval_data.get("groundedness_score", 95.0))
        faithfulness = float(eval_data.get("faithfulness", eval_data.get("faithfulness_score", 100.0)))
        source_cov = float(eval_data.get("source_coverage", 100.0))
        confidence = str(eval_data.get("confidence", "High"))
        hall_risk = float(eval_data.get("hallucination_risk", eval_data.get("hallucination_risk_score", 5.0)))
        
        adaptive_data = response_payload.get("adaptive_retrieval", {})
        retry_count = int(adaptive_data.get("retries_performed", timing.get("retry_count", 0)))

        # Claims breakdown
        claims_list = eval_data.get("claims", []) or ans_corr.get("claims_correction", [])
        verified_claims = int(ans_corr.get("supported", eval_data.get("supported_claims", 0)))
        part_claims = int(ans_corr.get("partially_supported", eval_data.get("partial_claims", 0)))
        removed_claims = int(ans_corr.get("removed", ans_corr.get("unsupported", eval_data.get("unsupported_claims", 0))))
        total_claims = int(ans_corr.get("total_claims", eval_data.get("total_claims", len(claims_list))))

        # Answer status determination
        ans_text = response_payload.get("final_verified_answer") or response_payload.get("answer") or ""
        draft_text = response_payload.get("draft_answer") or ans_text
        is_fallback = (
            not ans_text or
            ans_text.startswith("The selected documents do not") or
            ans_text.startswith("I couldn't find") or
            ans_text.startswith("I could not find") or
            ans_text.startswith("Not enough supporting") or
            ans_text.startswith("The answer is not available")
        )

        if is_fallback or (total_claims > 0 and verified_claims == 0 and removed_claims > 0):
            status = "UNSUPPORTED"
        elif part_claims > 0 and verified_claims == 0:
            status = "PARTIALLY_SUPPORTED"
        elif eval_data.get("risk_level") == "HIGH" or groundedness < 40:
            status = "HIGH_RISK"
        else:
            status = "VERIFIED"

        cache_status = str(response_payload.get("cache_status", "HIT" if response_payload.get("cache_hit") else "MISS"))
        
        sel_docs = selected_documents or response_payload.get("documents_used", [])
        sources_list = response_payload.get("sources", [])
        
        is_followup = 1 if response_payload.get("is_followup") or response_payload.get("context_used") else 0
        is_comp = 1 if response_payload.get("is_comparison") or response_payload.get("mode") == "comparison" else 0
        is_decomp = 1 if response_payload.get("is_decomposed") or response_payload.get("mode") == "decomposition" else 0
        context_topic = str(response_payload.get("context_topic", ""))

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interactions (
                    timestamp, question, query_type, response_time,
                    retrieval_time, reranking_time, compression_time, generation_time, verification_time,
                    chunks_retrieved, chunks_selected, retrieval_relevance, groundedness,
                    faithfulness, source_coverage, confidence, hallucination_risk,
                    retry_count, verified_claims, partially_supported_claims, removed_claims,
                    total_claims, status, cache_status, selected_documents, sources_json,
                    claims_json, timing_breakdown_json, answer_text, draft_answer,
                    context_topic, is_followup, is_comparison, is_decomposed
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
            """, (
                ts, question, query_type, resp_time,
                ret_time, rerank_time, comp_time, gen_time_float, verif_time_float,
                chunks_retrieved, chunks_selected, ret_relevance, groundedness,
                faithfulness, source_cov, confidence, hall_risk,
                retry_count, verified_claims, part_claims, removed_claims,
                total_claims, status, cache_status, json.dumps(sel_docs), json.dumps(sources_list),
                json.dumps(claims_list), json.dumps(timing), ans_text, draft_text,
                context_topic, is_followup, is_comp, is_decomp
            ))
            conn.commit()
            inserted_id = cursor.lastrowid
            logger.info(f"[ANALYTICS] Recorded interaction #{inserted_id} for query '{question[:40]}...' (Status: {status}, Time: {resp_time}s)")
            return inserted_id
    except Exception as e:
        logger.error(f"[ANALYTICS] Failed to record interaction: {e}", exc_info=True)
        return None


def get_analytics_dashboard_data() -> Dict[str, Any]:
    """
    Computes all summary cards, interactive chart data points, document usage frequencies,
    and the recent question history list for the RAG Analytics Dashboard.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Total counts & overall summary stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_questions,
                    MIN(response_time) as min_response_time,
                    AVG(response_time) as avg_response_time,
                    MAX(response_time) as max_response_time,
                    AVG(retrieval_relevance) as avg_retrieval_relevance,
                    AVG(groundedness) as avg_groundedness,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(source_coverage) as avg_source_coverage,
                    AVG(retry_count) as avg_retry_count,
                    AVG(chunks_retrieved) as avg_chunks_retrieved,
                    AVG(chunks_selected) as avg_chunks_selected,
                    SUM(CASE WHEN status = 'VERIFIED' THEN 1 ELSE 0 END) as verified_count,
                    SUM(CASE WHEN status = 'UNSUPPORTED' THEN 1 ELSE 0 END) as unsupported_count,
                    SUM(CASE WHEN status = 'PARTIALLY_SUPPORTED' THEN 1 ELSE 0 END) as partial_count,
                    SUM(CASE WHEN cache_status = 'HIT' THEN 1 ELSE 0 END) as cache_hit_count,
                    SUM(CASE WHEN LOWER(confidence) = 'high' THEN 1 ELSE 0 END) as high_conf_count,
                    SUM(CASE WHEN LOWER(confidence) = 'medium' THEN 1 ELSE 0 END) as med_conf_count,
                    SUM(CASE WHEN LOWER(confidence) = 'low' THEN 1 ELSE 0 END) as low_conf_count,
                    SUM(verified_claims) as total_verified_claims,
                    SUM(removed_claims) as total_unsupported_claims,
                    AVG(retrieval_time) as avg_retrieval_time,
                    AVG(reranking_time) as avg_reranking_time,
                    AVG(compression_time) as avg_compression_time,
                    AVG(generation_time) as avg_generation_time,
                    AVG(verification_time) as avg_verification_time
                FROM interactions
            """)
            summary_row = cursor.fetchone()

            total_q = summary_row["total_questions"] or 0
            cache_hits = int(summary_row["cache_hit_count"] or 0)
            cache_misses = max(0, total_q - cache_hits)
            cache_rate = round((cache_hits / max(1, total_q)) * 100.0, 1)

            verified_c = int(summary_row["verified_count"] or 0)
            partial_c = int(summary_row["partial_count"] or 0)
            unsupported_c = int(summary_row["unsupported_count"] or 0)
            successful_q = verified_c + partial_c
            failed_q = unsupported_c

            # Quality metrics
            citation_success_rate = round(((verified_c + partial_c) / max(1, total_q)) * 100.0, 1)

            # User Feedback
            fb_summary = get_user_feedback_summary()

            # Latest Evaluation Benchmark Report
            latest_eval = None
            try:
                from rag.evaluation_runner import get_latest_evaluation_report
                latest_eval = get_latest_evaluation_report()
            except Exception:
                pass

            summary = {
                "total_questions": total_q,
                "successful_questions": successful_q,
                "failed_questions": failed_q,
                "min_response_time": round(float(summary_row["min_response_time"] or 0.0), 2),
                "avg_response_time": round(float(summary_row["avg_response_time"] or 0.0), 2),
                "max_response_time": round(float(summary_row["max_response_time"] or 0.0), 2),
                "avg_retrieval_relevance": round(float(summary_row["avg_retrieval_relevance"] or 0.0), 1),
                "avg_groundedness": round(float(summary_row["avg_groundedness"] or 0.0), 1),
                "avg_faithfulness": round(float(summary_row["avg_faithfulness"] or 0.0), 1),
                "avg_source_coverage": round(float(summary_row["avg_source_coverage"] or 0.0), 1),
                "avg_retry_count": round(float(summary_row["avg_retry_count"] or 0.0), 2),
                "avg_chunks_retrieved": round(float(summary_row["avg_chunks_retrieved"] or 0.0), 1),
                "avg_chunks_selected": round(float(summary_row["avg_chunks_selected"] or 0.0), 1),
                "verified_count": verified_c,
                "unsupported_count": unsupported_c,
                "partial_count": partial_c,
                "total_verified_claims": int(summary_row["total_verified_claims"] or 0),
                "total_unsupported_claims": int(summary_row["total_unsupported_claims"] or 0),
                "citation_success_rate": citation_success_rate,
                "cache_hit_count": cache_hits,
                "cache_miss_count": cache_misses,
                "cache_hit_rate": cache_rate,
                "confidence_distribution": {
                    "high": int(summary_row["high_conf_count"] or 0),
                    "medium": int(summary_row["med_conf_count"] or 0),
                    "low": int(summary_row["low_conf_count"] or 0)
                },
                "feedback": fb_summary,
                "latest_evaluation": latest_eval.get("metrics") if latest_eval else None,
                "latency_breakdown_ms": {
                    "retrieval_ms": round(float(summary_row["avg_retrieval_time"] or 0.0) * 1000.0, 1),
                    "reranking_ms": round(float(summary_row["avg_reranking_time"] or 0.0) * 1000.0, 1),
                    "compression_ms": round(float(summary_row["avg_compression_time"] or 0.0) * 1000.0, 1),
                    "generation_ms": round(float(summary_row["avg_generation_time"] or 0.0) * 1000.0, 1),
                    "verification_ms": round(float(summary_row["avg_verification_time"] or 0.0) * 1000.0, 1),
                    "total_ms": round(float(summary_row["avg_response_time"] or 0.0) * 1000.0, 1)
                },
                "avg_retrieval_time": round(float(summary_row["avg_retrieval_time"] or 0.0), 3),
                "avg_reranking_time": round(float(summary_row["avg_reranking_time"] or 0.0), 3),
                "avg_compression_time": round(float(summary_row["avg_compression_time"] or 0.0), 3),
                "avg_generation_time": round(float(summary_row["avg_generation_time"] or 0.0), 2),
                "avg_verification_time": round(float(summary_row["avg_verification_time"] or 0.0), 3),
            }

            # 2. Time-series data for Charts (ordered by chronological id ASC)
            cursor.execute("""
                SELECT 
                    id, timestamp, question, response_time,
                    retrieval_relevance, groundedness, faithfulness,
                    source_coverage, status, cache_status, sources_json
                FROM interactions
                ORDER BY id ASC
                LIMIT 50
            """)
            chart_rows = cursor.fetchall()

            chart_labels = []
            response_times = []
            retrieval_relevances = []
            groundedness_scores = []
            faithfulness_scores = []
            source_coverages = []
            status_distribution = {"VERIFIED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0, "HIGH_RISK": 0}
            doc_usage_map: Dict[str, int] = {}

            for r in chart_rows:
                short_q = (r["question"][:22] + "..") if len(r["question"]) > 22 else r["question"]
                chart_labels.append(f"Q{r['id']}: {short_q}")
                response_times.append(round(float(r["response_time"] or 0.0), 2))
                retrieval_relevances.append(round(float(r["retrieval_relevance"] or 0.0), 1))
                groundedness_scores.append(round(float(r["groundedness"] or 0.0), 1))
                faithfulness_scores.append(round(float(r["faithfulness"] or 0.0), 1))
                source_coverages.append(round(float(r["source_coverage"] or 0.0), 1))

                st = r["status"] or "VERIFIED"
                status_distribution[st] = status_distribution.get(st, 0) + 1

                # Extract documents used from sources_json
                try:
                    srcs = json.loads(r["sources_json"] or "[]")
                    seen_in_q = set()
                    for s in srcs:
                        doc = s.get("document") or s.get("document_id") or s.get("filename")
                        if doc and doc not in seen_in_q:
                            seen_in_q.add(doc)
                            doc_usage_map[doc] = doc_usage_map.get(doc, 0) + 1
                except Exception:
                    pass

            # Sort doc usage descending
            sorted_doc_usage = sorted(doc_usage_map.items(), key=lambda x: x[1], reverse=True)
            doc_labels = [item[0] for item in sorted_doc_usage[:10]]
            doc_counts = [item[1] for item in sorted_doc_usage[:10]]

            # 3. Question History Table (latest 100 questions, newest first)
            cursor.execute("""
                SELECT 
                    id, timestamp, question, query_type, response_time,
                    retrieval_time, reranking_time, generation_time, verification_time,
                    chunks_retrieved, chunks_selected, retrieval_relevance, groundedness,
                    faithfulness, source_coverage, confidence, retry_count,
                    verified_claims, total_claims, status, cache_status, sources_json
                FROM interactions
                ORDER BY id DESC
                LIMIT 100
            """)
            history_rows = cursor.fetchall()
            
            history_list = []
            for h in history_rows:
                src_count = 0
                sources_preview = []
                try:
                    s_data = json.loads(h["sources_json"] or "[]")
                    src_count = len(s_data)
                    sources_preview = [f"{s.get('document', '')} (P{s.get('page', 1)})" for s in s_data[:3]]
                except Exception:
                    pass

                history_list.append({
                    "id": h["id"],
                    "timestamp": h["timestamp"],
                    "question": h["question"],
                    "query_type": h["query_type"],
                    "response_time": round(float(h["response_time"] or 0.0), 2),
                    "retrieval_time": round(float(h["retrieval_time"] or 0.0), 3),
                    "reranking_time": round(float(h["reranking_time"] or 0.0), 3),
                    "generation_time": round(float(h["generation_time"] or 0.0), 2),
                    "verification_time": round(float(h["verification_time"] or 0.0), 3),
                    "chunks_retrieved": h["chunks_retrieved"],
                    "chunks_selected": h["chunks_selected"],
                    "relevance": round(float(h["retrieval_relevance"] or 0.0), 1),
                    "groundedness": round(float(h["groundedness"] or 0.0), 1),
                    "faithfulness": round(float(h["faithfulness"] or 0.0), 1),
                    "source_coverage": round(float(h["source_coverage"] or 0.0), 1),
                    "confidence": h["confidence"] or "Medium",
                    "retry_count": h["retry_count"] or 0,
                    "verified_claims": h["verified_claims"] or 0,
                    "total_claims": h["total_claims"] or 0,
                    "status": h["status"] or "VERIFIED",
                    "cache_status": h["cache_status"] or "MISS",
                    "source_count": src_count,
                    "sources_preview": sources_preview
                })

            return {
                "success": True,
                "summary": summary,
                "charts": {
                    "labels": chart_labels,
                    "response_times": response_times,
                    "retrieval_relevances": retrieval_relevances,
                    "groundedness_scores": groundedness_scores,
                    "faithfulness_scores": faithfulness_scores,
                    "source_coverages": source_coverages,
                    "status_distribution": status_distribution,
                    "doc_usage": {
                        "labels": doc_labels,
                        "counts": doc_counts
                    }
                },
                "history": history_list
            }

    except Exception as e:
        logger.error(f"[ANALYTICS] Error fetching dashboard data: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "summary": {
                "total_questions": 0, "avg_response_time": 0.0, "avg_retrieval_relevance": 0.0,
                "avg_groundedness": 0.0, "avg_faithfulness": 0.0, "avg_source_coverage": 0.0,
                "verified_count": 0, "unsupported_count": 0, "partial_count": 0, "avg_retry_count": 0.0,
                "cache_hit_count": 0, "cache_hit_rate": 0.0
            },
            "charts": {"labels": [], "response_times": [], "retrieval_relevances": [], "groundedness_scores": [], "faithfulness_scores": [], "source_coverages": [], "status_distribution": {}, "doc_usage": {"labels": [], "counts": []}},
            "history": []
        }


def get_question_detail(question_id: Any) -> Optional[Dict[str, Any]]:
    """Returns the full detailed metrics, sources, claims, and timings for a specific question."""
    try:
        if isinstance(question_id, str):
            digits = re.findall(r'\d+', question_id)
            if digits:
                question_id = int(digits[0])
            else:
                return None
        else:
            question_id = int(question_id)

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM interactions WHERE id = ?", (question_id,))
            row = cursor.fetchone()
            if not row:
                return None

            query_type = row["query_type"] or "NORMAL_QUESTION"
            is_comp = bool(row["is_comparison"])
            is_decomp = bool(row["is_decomposed"])
            is_follow = bool(row["is_followup"])
            retry_cnt = int(row["retry_count"] or 0)

            # Determine human-readable retrieval method used
            if is_comp or query_type == "COMPARISON_QUESTION":
                retrieval_method = "Multi-Document Comparison Retrieval"
            elif is_decomp or query_type == "DECOMPOSED_QUESTION":
                retrieval_method = "Query Decomposition (Multi-Subquery Hybrid Search)"
            elif is_follow:
                retrieval_method = "Conversational Follow-up Retrieval (Contextual Query Rewriting)"
            elif retry_cnt > 0:
                retrieval_method = f"Adaptive Iterative Retrieval ({retry_cnt} Multi-Query Expansion{'s' if retry_cnt > 1 else ''})"
            elif query_type == "COMPLEX_QUESTION":
                retrieval_method = "Adaptive Hybrid Retrieval (Cross-Encoder Reranked)"
            else:
                retrieval_method = "Hybrid Search (FAISS Dense + BM25 Sparse + Cross-Encoder Reranking)"

            raw_status = row["status"] or "VERIFIED"
            if raw_status in ("UNSUPPORTED", "NOT_SUPPORTED", "UNVERIFIED"):
                formatted_status = "NOT SUPPORTED"
            elif raw_status in ("PARTIALLY_SUPPORTED", "PARTIAL"):
                formatted_status = "PARTIALLY SUPPORTED"
            elif raw_status == "HIGH_RISK":
                formatted_status = "HIGH RISK"
            else:
                formatted_status = "VERIFIED"

            selected_docs = []
            try:
                selected_docs = json.loads(row["selected_documents"] or "[]")
                if isinstance(selected_docs, str):
                    selected_docs = [selected_docs]
            except Exception:
                selected_docs = []

            sources_list = []
            try:
                sources_list = json.loads(row["sources_json"] or "[]")
            except Exception:
                sources_list = []

            claims_list = []
            try:
                claims_list = json.loads(row["claims_json"] or "[]")
            except Exception:
                claims_list = []

            timing_dict = {}
            try:
                timing_dict = json.loads(row["timing_breakdown_json"] or "{}")
            except Exception:
                timing_dict = {}

            chunks_retrieved = row["chunks_retrieved"] or len(sources_list)
            chunks_selected = row["chunks_selected"] or len(sources_list)
            chunks_compressed = timing_dict.get("compressed_chunks", chunks_selected)

            return {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "question": row["question"],
                "query_type": query_type,
                "retrieval_method": retrieval_method,
                "response_time": round(float(row["response_time"] or 0.0), 2),
                "retrieval_time": round(float(row["retrieval_time"] or 0.0), 3),
                "reranking_time": round(float(row["reranking_time"] or 0.0), 3),
                "compression_time": round(float(row["compression_time"] or 0.0), 3),
                "generation_time": round(float(row["generation_time"] or 0.0), 2),
                "verification_time": round(float(row["verification_time"] or 0.0), 3),
                "chunks_retrieved": chunks_retrieved,
                "chunks_selected": chunks_selected,
                "chunks_compressed": chunks_compressed,
                "retrieval_relevance": round(float(row["retrieval_relevance"] or 0.0), 1),
                "groundedness": round(float(row["groundedness"] or 0.0), 1),
                "faithfulness": round(float(row["faithfulness"] or 0.0), 1),
                "source_coverage": round(float(row["source_coverage"] or 0.0), 1),
                "confidence": row["confidence"] or "Medium",
                "hallucination_risk": round(float(row["hallucination_risk"] or 0.0), 1),
                "retry_count": retry_cnt,
                "verified_claims": row["verified_claims"] or 0,
                "partially_supported_claims": row["partially_supported_claims"] or 0,
                "removed_claims": row["removed_claims"] or 0,
                "total_claims": row["total_claims"] or len(claims_list),
                "status": formatted_status,
                "raw_status": raw_status,
                "cache_status": row["cache_status"] or "MISS",
                "selected_documents": selected_docs,
                "sources": sources_list,
                "claims": claims_list,
                "timing_breakdown": timing_dict,
                "answer_text": row["answer_text"] or "",
                "draft_answer": row["draft_answer"] or "",
                "context_topic": row["context_topic"] or "",
                "is_followup": is_follow,
                "is_comparison": is_comp,
                "is_decomposed": is_decomp
            }
    except Exception as e:
        logger.error(f"[ANALYTICS] Error fetching question detail for ID {question_id}: {e}", exc_info=True)
        return None


def export_analytics_csv_data() -> str:
    """Generates complete analytics CSV export string."""
    output = io.StringIO()
    writer = csv.writer(output)

    # CSV Header
    writer.writerow([
        "ID", "Timestamp", "Question", "Query Type", "Response Time (s)",
        "Retrieval Time (s)", "Reranking Time (s)", "Compression Time (s)",
        "Generation Time (s)", "Verification Time (s)", "Chunks Retrieved",
        "Chunks Selected", "Retrieval Relevance (%)", "Groundedness (%)",
        "Faithfulness (%)", "Source Coverage (%)", "Confidence",
        "Hallucination Risk (%)", "Retry Count", "Verified Claims",
        "Partially Supported Claims", "Removed Claims", "Total Claims",
        "Answer Status", "Cache Status", "Selected Documents Count",
        "Sources Count", "Is Follow-up", "Is Comparison", "Is Decomposed"
    ])

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM interactions ORDER BY id ASC")
            for row in cursor.fetchall():
                sel_docs = json.loads(row["selected_documents"] or "[]")
                srcs = json.loads(row["sources_json"] or "[]")

                writer.writerow([
                    row["id"],
                    row["timestamp"],
                    row["question"],
                    row["query_type"],
                    row["response_time"],
                    row["retrieval_time"],
                    row["reranking_time"],
                    row["compression_time"],
                    row["generation_time"],
                    row["verification_time"],
                    row["chunks_retrieved"],
                    row["chunks_selected"],
                    row["retrieval_relevance"],
                    row["groundedness"],
                    row["faithfulness"],
                    row["source_coverage"],
                    row["confidence"],
                    row["hallucination_risk"],
                    row["retry_count"],
                    row["verified_claims"],
                    row["partially_supported_claims"],
                    row["removed_claims"],
                    row["total_claims"],
                    row["status"],
                    row["cache_status"],
                    len(sel_docs),
                    len(srcs),
                    "Yes" if row["is_followup"] else "No",
                    "Yes" if row["is_comparison"] else "No",
                    "Yes" if row["is_decomposed"] else "No"
                ])
    except Exception as e:
        logger.error(f"[ANALYTICS] Error exporting CSV: {e}", exc_info=True)

    return output.getvalue()


def clear_all_analytics() -> bool:
    """Clears all stored analytics interactions."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM interactions")
            conn.commit()
            logger.info("[ANALYTICS] All analytics data cleared.")
            return True
    except Exception as e:
        logger.error(f"[ANALYTICS] Failed to clear analytics: {e}", exc_info=True)
        return False
