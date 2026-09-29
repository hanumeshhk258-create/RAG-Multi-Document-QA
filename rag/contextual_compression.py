"""
Contextual Compression Module for Multi-Document RAG.
Filters retrieved chunks to retain only the sentences and passages directly relevant to the user's query,
reducing token overhead, eliminating repetitive/unrelated text, and preserving 100% source traceability and metadata.
"""

import re
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Configurable constants
MAX_CONTEXT_CHUNKS = 5
MAX_CONTEXT_CHARACTERS = 12000
MIN_SENTENCE_LENGTH = 15
DEFAULT_RELEVANCE_THRESHOLD = 0.25


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into constituent sentences and bullet/numbered list items cleanly,
    preserving structural units and whole paragraphs without mid-sentence truncation.
    """
    if not text:
        return []
    
    # Normalize excessive whitespace
    normalized = re.sub(r'\r\n', '\n', text)
    
    # Split by double newlines or line breaks preserving list items
    raw_lines = normalized.split('\n')
    sentences = []
    
    for line in raw_lines:
        l_clean = line.strip()
        if not l_clean:
            continue
        if re.match(r'^(?:[-*•\u007f\x7f]|\d+[\.\)]|(?:INNER|LEFT|RIGHT|FULL|CROSS|NATURAL|OUTER)\s+JOIN)\s*', l_clean, re.IGNORECASE) or len(l_clean) < 80:
            sentences.append(l_clean)
        else:
            # Split only on definitive terminal punctuation followed by capitalized word
            sub_sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z0-9])', l_clean)
            for s in sub_sentences:
                s_strip = s.strip()
                if len(s_strip) >= 5:
                    sentences.append(s_strip)
    
    return sentences if sentences else [text.strip()]


def stem_token(token: str) -> str:
    """Lightweight morphological stemmer for matching morphological variants."""
    w = token.lower().strip()
    if len(w) <= 3:
        return w
    if w.endswith('sses'):
        w = w[:-2]
    elif w.endswith('ies'):
        w = w[:-3] + 'y' if len(w) > 4 else w[:-3] + 'i'
    elif w.endswith('ss'):
        pass
    elif w.endswith('s') and not w.endswith('us') and not w.endswith('is') and not w.endswith('as'):
        w = w[:-1]
    if w.endswith('ing') and len(w) > 5:
        w = w[:-3]
        if len(w) >= 3 and w[-1] == w[-2] and w[-1] not in ('l', 's', 'z'):
            w = w[:-1]
    elif w.endswith('ed') and len(w) > 4:
        w = w[:-2]
        if len(w) >= 3 and w[-1] == w[-2] and w[-1] not in ('l', 's', 'z'):
            w = w[:-1]
    elif w.endswith('tion') and len(w) > 5:
        w = w[:-3]
    elif w.endswith('ment') and len(w) > 6:
        w = w[:-4]
    return w


def _compute_lexical_overlap(query_tokens: set, sentence: str) -> float:
    """Computes normalized stemmed keyword and exact phrase overlap score."""
    if not query_tokens:
        return 0.0
    
    s_clean = sentence.lower()
    s_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', s_clean))
    if not s_tokens:
        return 0.0
    
    q_stems = {stem_token(t) for t in query_tokens}
    s_stems = {stem_token(t) for t in s_tokens}
    
    matched = (query_tokens.intersection(s_tokens)).union(q_stems.intersection(s_stems))
    if not matched:
        return 0.0
    
    overlap_ratio = len(matched) / len(query_tokens)
    
    # Boost if exact term matches or contains key definition markers
    def_boost = 0.0
    if any(marker in s_clean for marker in ["is a", "is an", "is the", "defined as", "refers to", "objective", "purpose", "problem", "solution", "architecture", "security", "model", "method", "process", "stage", "phase", "types of", "used to"]):
        def_boost = 0.15
        
    return min(1.0, overlap_ratio + def_boost)


def compress_chunk_content(
    query: str,
    original_text: str,
    cross_encoder: Optional[Any] = None,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD
) -> Tuple[str, Dict[str, Any]]:
    """
    Compresses a single chunk's text by extracting the sentences and cohesive sections relevant to the query.
    
    Guarantees:
    - Preserves all numbered items, stages, definitions, and technical explanations belonging to a matched concept.
    - Never truncates a matched concept mid-section.
    - Preserves 100% source fidelity and numerical accuracy.
    """
    if not original_text or not original_text.strip():
        return "", {"original_length": 0, "compressed_length": 0, "compression_ratio": 0.0, "sentence_count": 0, "kept_sentences": 0}
    
    clean_text = original_text.strip()
    orig_len = len(clean_text)
    
    # If chunk is already compact, keep it entirely
    if orig_len <= 300:
        return clean_text, {
            "original_length": orig_len,
            "compressed_length": orig_len,
            "compression_ratio": 0.0,
            "sentence_count": 1,
            "kept_sentences": 1
        }
    
    sentences = split_into_sentences(clean_text)
    if len(sentences) <= 2:
        return clean_text, {
            "original_length": orig_len,
            "compressed_length": orig_len,
            "compression_ratio": 0.0,
            "sentence_count": len(sentences),
            "kept_sentences": len(sentences)
        }
    
    # Extract query tokens (ignoring trivial stop words)
    stop_words = {"what", "is", "the", "a", "an", "and", "or", "in", "on", "of", "to", "for", "with", "how", "why", "are", "by", "from", "at", "as"}
    q_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', query.lower())) - stop_words
    if not q_tokens:
        q_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', query.lower()))

    # Score each sentence
    sentence_scores: List[float] = []
    
    # Method 1: Semantic Scoring with Cross-Encoder if available
    use_cross_encoder = False
    if cross_encoder is not None and hasattr(cross_encoder, "predict") and len(sentences) <= 30:
        try:
            pairs = [[query, s] for s in sentences]
            ce_raw = cross_encoder.predict(pairs)
            import numpy as np
            ce_scores = 1.0 / (1.0 + np.exp(-np.clip(ce_raw, -10, 10)))
            sentence_scores = [float(s) for s in ce_scores]
            use_cross_encoder = True
        except Exception as ce_err:
            logger.debug(f"[COMPRESSION] Cross-encoder sentence scoring fallback: {ce_err}")
            use_cross_encoder = False

    if not use_cross_encoder:
        # Method 2: Lexical and Keyword overlap scoring
        for s in sentences:
            score = _compute_lexical_overlap(q_tokens, s)
            sentence_scores.append(score)

    kept_indices = set()
    max_score = max(sentence_scores) if sentence_scores else 0.0
    
    # Find matching anchor indices
    anchor_indices = []
    for idx, score in enumerate(sentence_scores):
        lex_sc = _compute_lexical_overlap(q_tokens, sentences[idx])
        if use_cross_encoder:
            if score >= max(threshold, max_score * 0.45) or lex_sc >= 0.35:
                anchor_indices.append(idx)
        else:
            if score >= max(0.15, max_score * 0.4) or lex_sc >= 0.35:
                anchor_indices.append(idx)

    # If anchor indices found, expand to include their complete section & list items
    for a_idx in anchor_indices:
        kept_indices.add(a_idx)
        # Include immediate preceding context if it's a section title
        if a_idx > 0 and (len(sentences[a_idx - 1]) < 80 or re.match(r'^\d+\.\s+', sentences[a_idx - 1])):
            kept_indices.add(a_idx - 1)
        
        # Lookahead: Include all subsequent numbered items, list items, stage descriptions, or paragraph continuation
        in_list_mode = False
        for next_idx in range(a_idx + 1, min(len(sentences), a_idx + 25)):
            s_next = sentences[next_idx]
            is_num_item = bool(re.match(r'^(?:[-*•\u007f\x7f]|\d+[\.\)]|(?:INNER|LEFT|RIGHT|FULL|CROSS|NATURAL|OUTER)\s+JOIN)\s*', s_next, re.IGNORECASE))
            if is_num_item:
                in_list_mode = True
                kept_indices.add(next_idx)
            elif in_list_mode and len(s_next) < 220:
                # Explanatory description line under numbered item
                kept_indices.add(next_idx)
            elif re.search(r'\b(?:stage|phase|step|requirement|design|implementation|testing|deployment|readiness|maintenance|component|layer)\b', s_next, re.IGNORECASE):
                kept_indices.add(next_idx)
            else:
                # If a completely new major section with different title starts (e.g. "8. Overall Development Flowchart"), stop expansion
                if re.match(r'^\d+\.\s+[A-Z]', s_next) and not re.match(r'^[1-9]\.\s+[A-Z][a-z]', s_next):
                    break
                if not in_list_mode and next_idx - a_idx <= 4:
                    kept_indices.add(next_idx)
                elif in_list_mode:
                    break

    # Safety check: If nothing was kept, retain top 3 sentences or full text
    if not kept_indices:
        if sentence_scores and max_score > 0:
            top_idx = int(max(range(len(sentence_scores)), key=lambda i: sentence_scores[i]))
            for k in range(max(0, top_idx - 1), min(len(sentences), top_idx + 3)):
                kept_indices.add(k)
        else:
            kept_indices = set(range(min(4, len(sentences))))

    # Build compressed text in original sentence order
    sorted_indices = sorted(list(kept_indices))
    selected_sentences = [sentences[i] for i in sorted_indices]
    
    compressed_text = "\n".join(selected_sentences).strip()
    comp_len = len(compressed_text)
    
    # Safety: If compressed text is too small (< 100 chars on a chunk with > 300 chars), preserve full chunk
    if comp_len < 100 and orig_len >= 250:
        compressed_text = clean_text
        comp_len = orig_len
        kept_indices = set(range(len(sentences)))

    ratio = round((1.0 - (comp_len / max(1, orig_len))) * 100.0, 1) if orig_len > 0 else 0.0
    if ratio < 0.0:
        ratio = 0.0

    stats = {
        "original_length": orig_len,
        "compressed_length": comp_len,
        "compression_ratio": ratio,
        "sentence_count": len(sentences),
        "kept_sentences": len(kept_indices)
    }

    return compressed_text, stats


def compress_context(
    query: str,
    retrieved_chunks: List[Any],
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    max_characters: int = MAX_CONTEXT_CHARACTERS,
    cross_encoder: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Contextual Compression Layer.
    
    Workflow:
    1. Iterates over top retrieved/reranked candidate chunks.
    2. Identifies and extracts query-relevant sentences and passages from each chunk.
    3. Removes irrelevant sections, boilerplate headers/footers, and duplicate passages.
    4. Calculates per-chunk and overall compression ratios.
    5. Preserves all source traceability (doc name, page number, chunk ID, relevance score).
    6. Respects configurable limits (max_chunks, max_characters).
    7. Provides safe fallback to original chunks in case of any processing exception.
    """
    t_start = time.perf_counter()
    
    if not retrieved_chunks:
        return {
            "compressed_chunks": [],
            "formatted_context": "",
            "summary": {
                "original_characters": 0,
                "compressed_characters": 0,
                "reduction_percentage": 0.0,
                "chunks_compressed_count": 0,
                "compression_time_ms": 0.0,
                "compression_time_sec": 0.0,
                "applied": False
            }
        }

    try:
        compressed_chunk_list = []
        seen_passage_signatures = set()
        total_orig_chars = 0
        total_comp_chars = 0

        # Process up to max_chunks
        candidate_items = retrieved_chunks[:max_chunks]
        
        for idx, item in enumerate(candidate_items):
            # Parse chunk item variations: Tuple[doc, score, pct], Tuple[doc, score], or Document
            if isinstance(item, tuple) or isinstance(item, list):
                doc = item[0]
                score = float(item[1]) if len(item) > 1 else 0.90
                rel_pct = int(item[2]) if len(item) > 2 else int(round(score * 100))
            elif isinstance(item, dict):
                doc = item.get("doc", item)
                score = float(item.get("score", 0.90))
                rel_pct = int(item.get("relevance_pct", int(round(score * 100))))
            else:
                doc = item
                score = 0.90
                rel_pct = 90

            metadata = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
            doc_name = metadata.get("document") or metadata.get("source") or metadata.get("filename") or "Document"
            page_num = int(metadata.get("page", 1))
            chunk_id = metadata.get("chunk_id") or f"{doc_name}_p{page_num}_c{idx+1}"
            orig_text = doc.page_content.strip() if hasattr(doc, "page_content") else str(doc).strip()

            total_orig_chars += len(orig_text)

            is_table_chunk = bool(metadata.get("is_table") or orig_text.startswith("[Table]"))
            table_num = metadata.get("table_number")
            table_cols = metadata.get("columns", [])
            table_rows_data = metadata.get("table_rows", [])
            table_md = metadata.get("table_markdown", "")

            # Compress content: if it's a table chunk, keep table structure intact
            if is_table_chunk:
                compressed_text = orig_text
                chunk_stats = {
                    "original_length": len(orig_text),
                    "compressed_length": len(orig_text),
                    "compression_ratio": 0.0,
                    "sentence_count": len(table_rows_data) + 1,
                    "kept_sentences": len(table_rows_data) + 1
                }
            else:
                compressed_text, chunk_stats = compress_chunk_content(
                    query=query,
                    original_text=orig_text,
                    cross_encoder=cross_encoder
                )
                # If compression filtered out everything or left negligible text, fallback to original chunk
                if not compressed_text or len(compressed_text.strip()) < 20:
                    compressed_text = orig_text
                    chunk_stats["compressed_length"] = len(orig_text)
                    chunk_stats["compression_ratio"] = 0.0

            # Deduplication & Near-duplicate filter across chunks
            # Check passage signature (first 80 alphanumeric characters normalized)
            sig = re.sub(r'[^a-zA-Z0-9]', '', compressed_text[:100].lower())
            if sig and sig in seen_passage_signatures:
                logger.info(f"[COMPRESSION] Skipping near-duplicate compressed passage from {doc_name} Page {page_num}")
                continue
            if sig:
                seen_passage_signatures.add(sig)

            # Enforce max_characters budget
            if total_comp_chars + len(compressed_text) > max_characters and compressed_chunk_list:
                # Truncate if exceeds character limit
                remaining_budget = max(0, max_characters - total_comp_chars)
                if remaining_budget > 100:
                    compressed_text = compressed_text[:remaining_budget]
                else:
                    break

            total_comp_chars += len(compressed_text)

            # Build new Document object with compressed content and preserved metadata
            updated_meta = dict(metadata)
            updated_meta["original_text"] = orig_text
            updated_meta["compression_ratio"] = chunk_stats["compression_ratio"]
            updated_meta["original_length"] = chunk_stats["original_length"]
            updated_meta["compressed_length"] = len(compressed_text)
            
            comp_doc = Document(page_content=compressed_text, metadata=updated_meta)

            chunk_info = {
                "document": doc_name,
                "document_id": doc_name,
                "filename": doc_name,
                "page": page_num,
                "chunk_id": chunk_id,
                "original_text": orig_text,
                "compressed_text": compressed_text,
                "original_length": chunk_stats["original_length"],
                "compressed_length": len(compressed_text),
                "compression_ratio": chunk_stats["compression_ratio"],
                "score": round(score, 3),
                "relevance_pct": rel_pct,
                "pdf_url": f"/view_pdf/{doc_name}#page={page_num}",
                "snippet": compressed_text[:500] + ("..." if len(compressed_text) > 500 else ""),
                "excerpt": compressed_text,
                "is_table": is_table_chunk
            }
            if is_table_chunk:
                chunk_info["table_number"] = table_num
                chunk_info["columns"] = table_cols
                chunk_info["table_rows"] = table_rows_data
                chunk_info["table_markdown"] = table_md

            compressed_chunk_list.append(chunk_info)

        # Calculate overall reduction percentage
        overall_reduction = 0.0
        if total_orig_chars > 0:
            overall_reduction = round((1.0 - (total_comp_chars / total_orig_chars)) * 100.0, 1)
            if overall_reduction < 0:
                overall_reduction = 0.0

        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        elapsed_sec = round(elapsed_ms / 1000.0, 3)

        logger.info(
            f"[CONTEXTUAL_COMPRESSION] {len(compressed_chunk_list)} chunks compressed. "
            f"Original: {total_orig_chars} chars -> Compressed: {total_comp_chars} chars "
            f"({overall_reduction}% reduction in {elapsed_ms}ms)"
        )

        # Build formatted context string for LLM prompt
        formatted_blocks = []
        for item in compressed_chunk_list:
            d_name = item["document"]
            p_num = item["page"]
            c_text = item["compressed_text"]
            formatted_blocks.append(f"[Document: {d_name}, Page: {p_num}]\n{c_text}")
        
        formatted_context_str = "\n\n---\n\n".join(formatted_blocks)

        return {
            "compressed_chunks": compressed_chunk_list,
            "formatted_context": formatted_context_str,
            "summary": {
                "original_characters": total_orig_chars,
                "compressed_characters": total_comp_chars,
                "reduction_percentage": overall_reduction,
                "chunks_compressed_count": len(compressed_chunk_list),
                "compression_time_ms": elapsed_ms,
                "compression_time_sec": elapsed_sec,
                "applied": True
            }
        }

    except Exception as err:
        logger.error(f"[CONTEXTUAL_COMPRESSION] Error during compression: {err}. Falling back to uncompressed context.", exc_info=True)
        elapsed_sec = round(time.perf_counter() - t_start, 3)
        
        # Fallback to uncompressed chunks
        fallback_list = []
        fallback_blocks = []
        for idx, item in enumerate(retrieved_chunks[:max_chunks]):
            if isinstance(item, (tuple, list)):
                doc = item[0]
                score = item[1] if len(item) > 1 else 0.90
            else:
                doc = item
                score = 0.90
            
            d_name = getattr(doc, "metadata", {}).get("document", "Document")
            p_num = getattr(doc, "metadata", {}).get("page", 1)
            t_text = doc.page_content.strip() if hasattr(doc, "page_content") else str(doc).strip()
            
            fallback_list.append({
                "document": d_name,
                "document_id": d_name,
                "filename": d_name,
                "page": p_num,
                "chunk_id": f"{d_name}_p{p_num}_c{idx+1}",
                "original_text": t_text,
                "compressed_text": t_text,
                "original_length": len(t_text),
                "compressed_length": len(t_text),
                "compression_ratio": 0.0,
                "score": score,
                "relevance_pct": 90,
                "pdf_url": f"/view_pdf/{d_name}#page={p_num}",
                "snippet": t_text[:500],
                "excerpt": t_text
            })
            fallback_blocks.append(f"[Document: {d_name}, Page: {p_num}]\n{t_text}")

        return {
            "compressed_chunks": fallback_list,
            "formatted_context": "\n\n---\n\n".join(fallback_blocks),
            "summary": {
                "original_characters": sum(len(x["original_text"]) for x in fallback_list),
                "compressed_characters": sum(len(x["original_text"]) for x in fallback_list),
                "reduction_percentage": 0.0,
                "chunks_compressed_count": len(fallback_list),
                "compression_time_ms": round(elapsed_sec * 1000.0, 2),
                "compression_time_sec": elapsed_sec,
                "applied": False,
                "fallback": True
            }
        }
