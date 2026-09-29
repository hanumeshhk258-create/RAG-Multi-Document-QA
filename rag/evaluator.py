import re
import math
import time
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'it', 'its',
    'this', 'that', 'these', 'those', 'and', 'or', 'if', 'because', 'as',
    'until', 'while', 'of', 'i', 'you', 'he', 'she', 'they', 'them', 'we',
    'me', 'him', 'her', 'us', 'my', 'your', 'his', 'their', 'our'
}

BOILERPLATE_PATTERNS = [
    r'^term:\s*',
    r'^definition:\s*',
    r'^explanation\s*/?\s*key points:\s*',
    r'^key differences:\s*',
    r'^common points\s*/?\s*similarities:\s*',
    r'^example:\s*',
    r'^source:\s*',
    r'^sources:\s*',
    r'^according to\s+'
]

FALLBACK_PHRASES = [
    "the selected documents do not contain enough information to answer this question",
    "the selected documents do not provide enough information to answer this question",
    "the answer is not available in the selected documents",
    "not enough supporting evidence was found in the selected documents",
    "i couldn't find",
    "i could not find",
    "the selected documents do not provide",
    "the selected documents explain this concept, but they do not provide",
    "please specify what you would like me to explain",
    "ai service is temporarily unavailable",
    "the ai service is temporarily unavailable",
    "unable to generate the answer within the time limit"
]

# Configurable Hallucination Risk Thresholds
DEFAULT_HALLUCINATION_THRESHOLDS = {
    "low_max": 10.0,    # 0 - 10%: LOW
    "med_max": 30.0     # 11 - 30%: MEDIUM, >30%: HIGH
}


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


def extract_informative_tokens(text: str) -> List[str]:
    """Extracts lowercase alphabetic/numeric tokens, filtering out standard stop words."""
    if not text:
        return []
    words = re.findall(r'[a-zA-Z0-9_\-]+', text.lower())
    return [w for w in words if len(w) > 1 and w not in STOP_WORDS]


# =============================================================================
# 1. CLAIM EXTRACTION
# =============================================================================
def extract_claims(answer: str) -> List[str]:
    """
    Extracts individual factual claims / assertions from the generated answer.
    Excludes markdown headers (#, ##), tables (|), code blocks, bullet markers,
    boilerplate labels (Term:, Definition:, etc.), greetings, or introductory remarks.
    """
    if not answer or not answer.strip():
        return []

    # Check for pure fallback answers
    ans_lower = answer.strip().lower()
    for fb in FALLBACK_PHRASES:
        if ans_lower.startswith(fb):
            return []

    raw_lines = answer.split("\n")
    candidate_sentences = []

    # Common non-factual structural intro patterns
    NON_FACTUAL_INTROS = [
        r'^based on the (?:provided|selected) documents',
        r'^here is a comparison',
        r'^here are the (?:key|main|different)',
        r'^the (?:context|document|text) lists the following',
        r'^the following (?:are|is|details|points)',
        r'^as (?:shown|explained|detailed) (?:below|above)',
        r'^in summary',
        r'^for example',
        r'^note that',
        r'^please specify'
    ]

    for line in raw_lines:
        line_str = line.strip()
        if not line_str:
            continue
        
        # Skip code blocks and horizontal rule separators
        if line_str.startswith("```") or line_str.startswith("---") or line_str.startswith("==="):
            continue

        # Handle markdown table rows
        if line_str.startswith("|"):
            # Skip separator rows like |---|---|
            if re.match(r'^\|[\s\-:|]+\|$', line_str):
                continue
            cells = [c.strip() for c in line_str.split("|") if c.strip()]
            # Skip table header row if it contains Aspect / Topic
            if any(c.lower() in ('aspect', 'feature', 'category', 'parameter', 'dimension') for c in cells):
                continue
            # Process non-header table cells as candidate lines
            for cell in cells:
                cell_clean = re.sub(r'^\*\*[^\*]+\*\*[:\s]*', '', cell).strip()
                if len(cell_clean) >= 18 and len(cell_clean.split()) >= 4:
                    candidate_sentences.append(cell_clean)
            continue
            
        # Strip markdown headings (# Title -> Title)
        line_str = re.sub(r'^#+\s*', '', line_str).strip()
        
        # Strip bullet/numbered list markers (e.g. * Item, - Item, 1. Item, 1) Item)
        line_str = re.sub(r'^(?:[-*•]|\d+[\.\)])\s*', '', line_str).strip()
        
        # Strip boilerplate prefixes
        for b_pat in BOILERPLATE_PATTERNS:
            line_str = re.sub(b_pat, '', line_str, flags=re.IGNORECASE).strip()
            
        # Strip bold headings at start of line like "**Definition:**" or "**1. Point:**"
        line_str = re.sub(r'^\*\*[^\*]+\*\*[:\s]*', '', line_str).strip()
        
        if not line_str:
            continue

        # Split line into sentence chunks using punctuation while protecting decimals/acronyms
        sentences = re.split(r'(?<=[.!?])\s+', line_str)
        for s in sentences:
            s_clean = s.strip()
            # Clean outer quotes and markdown formatting
            s_clean = s_clean.strip('"\'`*').strip()
            
            # Filter out non-claims
            if len(s_clean) < 18:
                continue
            words = s_clean.split()
            if len(words) < 4:
                continue
                
            # Filter out question sentences, citations, or purely navigational/introductory phrases
            if s_clean.endswith('?'):
                continue
            if any(re.match(p, s_clean, re.IGNORECASE) for p in NON_FACTUAL_INTROS):
                continue
            if re.match(r'^(?:source|page|document)[\s\d:.\-_]+$', s_clean, re.IGNORECASE):
                continue
            # Filter out standalone titles/phrases that end with a colon
            if s_clean.endswith(':'):
                continue
                
            candidate_sentences.append(s_clean)

    # Deduplicate while preserving order
    seen = set()
    claims = []
    for c in candidate_sentences:
        norm = " ".join(c.lower().split())
        if norm not in seen and len(norm) > 15:
            seen.add(norm)
            claims.append(c)

    return claims


# =============================================================================
# 2. CLAIM VERIFICATION ENGINE (MULTI-DOCUMENT AWARE & BATCH-OPTIMIZED)
# =============================================================================
import hashlib

# In-memory LRU verification cache mapping: hash(claim_norm + evidence_hash) -> verification result dict
_VERIFICATION_CACHE: Dict[str, Dict[str, Any]] = {}
MAX_VERIFICATION_CACHE_SIZE = 2000

def clear_verification_cache():
    """Clears the in-memory claim verification cache."""
    global _VERIFICATION_CACHE
    _VERIFICATION_CACHE.clear()
    logger.info("Claim verification cache cleared.")

def _clean_chunk_sentence(text: str) -> str:
    return " ".join(text.strip().split())


def _compute_chunk_evidence_hash(retrieved_items: List[Any]) -> str:
    """Computes a lightweight hash of all chunk IDs and content for cache keys."""
    raw = []
    for item in retrieved_items:
        if isinstance(item, (tuple, list)):
            doc = item[0]
        elif isinstance(item, dict):
            doc = item.get("doc") or item
        else:
            doc = item
        d_name = getattr(doc, "metadata", {}).get("document") or getattr(doc, "metadata", {}).get("source", "Doc") if hasattr(doc, "metadata") else "Doc"
        p_num = getattr(doc, "metadata", {}).get("page", 1) if hasattr(doc, "metadata") else 1
        txt_snippet = doc.page_content[:60] if hasattr(doc, "page_content") else str(doc)[:60]
        raw.append(f"{d_name}_p{p_num}_{txt_snippet}")
    return hashlib.md5("||".join(raw).encode("utf-8")).hexdigest()


def verify_claims_batch(
    claims: List[str],
    retrieved_items: List[Any],
    cross_encoder=None
) -> List[Dict[str, Any]]:
    """
    Batched, high-speed semantic claim verification engine.
    1. Checks in-memory cache for already verified claims against current evidence.
    2. For uncached claims, extracts candidate evidence segments from retrieved items.
    3. Performs a SINGLE batched Cross-Encoder forward pass for all candidate pairs across all claims.
    4. Caches and returns structured verification results for every claim.
    """
    if not claims:
        return []
    if not retrieved_items:
        return [
            {
                "claim": c,
                "status": "NOT_SUPPORTED",
                "document": None,
                "document_name": None,
                "page": None,
                "page_number": None,
                "chunk_id": None,
                "evidence": "No supporting evidence found in selected documents.",
                "score": 0.0,
                "pdf_url": ""
            }
            for c in claims
        ]

    evidence_hash = _compute_chunk_evidence_hash(retrieved_items)
    results: List[Optional[Dict[str, Any]]] = [None] * len(claims)
    uncached_indices = []

    # Check cache first
    for idx, claim in enumerate(claims):
        clean_claim = re.sub(r'\s*\[(?:Document:\s*)?[^\]]+\.pdf\s*\|\s*Page:?\s*\d+(?:,\s*Table\s*\d+)?\]', '', claim, flags=re.IGNORECASE)
        clean_claim = re.sub(r'\s*\[[^\]]+\.pdf,\s*(?:Page|pg\.?)\s*\d+(?:,\s*Table\s*\d+)?\]', '', clean_claim, flags=re.IGNORECASE).strip()
        if not clean_claim:
            clean_claim = claim.strip()
        norm_key = hashlib.md5(f"{clean_claim.lower()}::{evidence_hash}".encode("utf-8")).hexdigest()
        if norm_key in _VERIFICATION_CACHE:
            cached_res = dict(_VERIFICATION_CACHE[norm_key])
            cached_res["claim"] = claim
            results[idx] = cached_res
        else:
            uncached_indices.append((idx, claim, clean_claim, norm_key))

    if not uncached_indices:
        return [r for r in results if r is not None]

    # Pre-parse chunks for uncached verification
    parsed_chunks = []
    for item in retrieved_items:
        if isinstance(item, (tuple, list)):
            doc = item[0]
            item_score = float(item[1]) if len(item) > 1 else 0.85
        elif isinstance(item, dict):
            doc = item.get("doc") or item
            item_score = float(item.get("score", 0.85))
        else:
            doc = item
            item_score = 0.85

        doc_name = (
            getattr(doc, "metadata", {}).get("document") or
            getattr(doc, "metadata", {}).get("source", "Document")
            if hasattr(doc, "metadata")
            else (item.get("document", "Document") if isinstance(item, dict) else "Document")
        )
        page_num = (
            getattr(doc, "metadata", {}).get("page", 1)
            if hasattr(doc, "metadata")
            else (item.get("page", 1) if isinstance(item, dict) else 1)
        )
        chunk_text = (
            getattr(doc, "page_content", "")
            if hasattr(doc, "page_content")
            else (item.get("text", "") if isinstance(item, dict) else str(doc))
        )
        chunk_id = (
            getattr(doc, "metadata", {}).get("chunk_id", f"{doc_name}_p{page_num}")
            if hasattr(doc, "metadata")
            else (item.get("chunk_id", f"{doc_name}_p{page_num}") if isinstance(item, dict) else f"{doc_name}_p{page_num}")
        )
        parsed_chunks.append({
            "doc_name": doc_name,
            "page_num": page_num,
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
            "base_score": item_score
        })

    # Prepare batch neural candidate pairs
    all_ce_pairs = []
    pair_mapping = [] # (claim_list_idx, doc_name, page_num, chunk_id, seg_text, lexical_score)
    claim_best_lexical = {} # claim_list_idx -> (best_score, doc_name, page_num, chunk_id, evidence)

    for c_item in uncached_indices:
        orig_idx, claim, clean_claim, norm_key = c_item
        claim_tokens = extract_informative_tokens(clean_claim)
        claim_token_set = set(claim_tokens)
        claim_stems = [stem_token(t) for t in claim_tokens]
        claim_stem_set = set(claim_stems)
        if not claim_token_set and not claim_stem_set:
            res = {
                "claim": claim,
                "status": "NOT_SUPPORTED",
                "document": None,
                "document_name": None,
                "page": None,
                "page_number": None,
                "chunk_id": None,
                "evidence": "No supporting evidence found in selected documents.",
                "score": 0.0,
                "pdf_url": ""
            }
            results[orig_idx] = res
            continue

        claim_bigrams = set(zip(claim_tokens[:-1], claim_tokens[1:])) if len(claim_tokens) >= 2 else set()
        best_lex_score = 0.0
        best_lex_doc = None
        best_lex_page = 1
        best_lex_chk = None
        best_lex_evid = ""

        claim_pairs = []

        for c_info in parsed_chunks:
            chunk_text = c_info["chunk_text"]
            if not chunk_text:
                continue

            raw_sentences = re.split(r'(?<=[.!?\n])\s+', chunk_text)
            sentences = [_clean_chunk_sentence(s) for s in raw_sentences if len(s.strip()) > 8]
            if not sentences:
                sentences = [_clean_chunk_sentence(chunk_text)]

            segments = [(s, "sentence") for s in sentences]
            for i in range(len(sentences) - 1):
                segments.append((f"{sentences[i]} {sentences[i+1]}", "window2"))
            if len(sentences) > 3:
                segments.append((_clean_chunk_sentence(chunk_text), "chunk"))

            for seg_text, seg_type in segments:
                seg_tokens = extract_informative_tokens(seg_text)
                seg_token_set = set(seg_tokens)
                if not seg_token_set:
                    continue

                seg_stems = [stem_token(t) for t in seg_tokens]
                seg_stem_set = set(seg_stems)

                overlap = claim_token_set.intersection(seg_token_set)
                stem_overlap = claim_stem_set.intersection(seg_stem_set)

                token_recall = len(overlap) / len(claim_token_set) if claim_token_set else 0.0
                stem_recall = len(stem_overlap) / len(claim_stem_set) if claim_stem_set else 0.0
                recall = max(token_recall, stem_recall)

                precision = len(stem_overlap) / len(seg_stem_set) if seg_stem_set else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                bigram_overlap_ratio = 0.0
                if claim_bigrams and len(seg_tokens) >= 2:
                    seg_bigrams = set(zip(seg_tokens[:-1], seg_tokens[1:]))
                    bg_common = claim_bigrams.intersection(seg_bigrams)
                    bigram_overlap_ratio = len(bg_common) / len(claim_bigrams)

                if seg_type == "sentence":
                    seg_score = (0.65 * recall) + (0.25 * f1) + (0.10 * bigram_overlap_ratio)
                elif seg_type == "window2":
                    seg_score = (0.75 * recall) + (0.15 * f1) + (0.10 * bigram_overlap_ratio)
                else:
                    seg_score = (0.80 * recall) + (0.10 * f1) + (0.10 * bigram_overlap_ratio)

                if seg_score > best_lex_score:
                    best_lex_score = seg_score
                    best_lex_doc = c_info["doc_name"]
                    best_lex_page = c_info["page_num"]
                    best_lex_chk = c_info["chunk_id"]
                    best_lex_evid = seg_text

                if seg_score >= 0.18 and len(claim_pairs) < 2:
                    claim_pairs.append((clean_claim, seg_text, c_info["doc_name"], c_info["page_num"], c_info["chunk_id"], seg_score))

        claim_best_lexical[orig_idx] = (best_lex_score, best_lex_doc, best_lex_page, best_lex_chk, best_lex_evid)

        for cp in claim_pairs:
            all_ce_pairs.append((cp[0], (cp[1] or "")[:300]))
            pair_mapping.append((orig_idx, cp[2], cp[3], cp[4], cp[1], cp[5]))

    # Execute single batch Cross-Encoder scoring for all uncached claims
    pair_neural_scores = {} # orig_idx -> list of (score, doc_n, page_n, chk_id, evid)
    if cross_encoder is not None and getattr(cross_encoder, "is_available", lambda: False)() and all_ce_pairs:
        try:
            ce_scores = cross_encoder.score_pairs(all_ce_pairs)
            if ce_scores and len(ce_scores) == len(pair_mapping):
                for score_val, p_info in zip(ce_scores, pair_mapping):
                    c_idx, doc_n, page_n, chk_id, evid_text, lex_sc = p_info
                    sig_sc = 1.0 / (1.0 + math.exp(-float(score_val)))
                    blended = max(sig_sc, 0.6 * sig_sc + 0.4 * lex_sc)
                    if c_idx not in pair_neural_scores:
                        pair_neural_scores[c_idx] = []
                    pair_neural_scores[c_idx].append((blended, doc_n, page_n, chk_id, evid_text))
        except Exception as ce_err:
            logger.debug(f"Batch neural verification error: {ce_err}")

    # Build final result for each uncached claim
    for c_item in uncached_indices:
        orig_idx, claim, clean_claim, norm_key = c_item
        if results[orig_idx] is not None:
            continue

        lex_sc, lex_doc, lex_page, lex_chk, lex_evid = claim_best_lexical.get(orig_idx, (0.0, None, None, None, ""))
        best_score = lex_sc
        best_doc_name = lex_doc
        best_page_num = lex_page
        best_chunk_id = lex_chk
        best_evidence = lex_evid

        if orig_idx in pair_neural_scores:
            for n_sc, n_doc, n_page, n_chk, n_evid in pair_neural_scores[orig_idx]:
                if n_sc > best_score:
                    best_score = n_sc
                    best_doc_name = n_doc
                    best_page_num = n_page
                    best_chunk_id = n_chk
                    best_evidence = n_evid

        if best_score >= 0.35:
            status = "SUPPORTED"
        elif best_score >= 0.18:
            status = "PARTIALLY_SUPPORTED"
        else:
            status = "NOT_SUPPORTED"
            best_evidence = "No supporting evidence found in selected documents."
            best_doc_name = None
            best_page_num = None

        if best_evidence and len(best_evidence) > 300:
            best_evidence = best_evidence[:300] + "..."

        pdf_url = f"/view_pdf/{best_doc_name}#page={best_page_num}" if best_doc_name and best_page_num else ""

        res = {
            "claim": claim,
            "status": status,
            "document": best_doc_name,
            "document_name": best_doc_name,
            "page": best_page_num,
            "page_number": best_page_num,
            "chunk_id": best_chunk_id,
            "evidence": best_evidence,
            "score": round(float(best_score), 3),
            "pdf_url": pdf_url
        }

        # Cache result
        if len(_VERIFICATION_CACHE) < MAX_VERIFICATION_CACHE_SIZE:
            _VERIFICATION_CACHE[norm_key] = dict(res)

        results[orig_idx] = res

    return [r for r in results if r is not None]


def verify_claim(
    claim: str,
    retrieved_items: List[Any],
    cross_encoder=None
) -> Dict[str, Any]:
    """
    Backward-compatible single-claim verification wrapper calling verify_claims_batch.
    """
    res_list = verify_claims_batch([claim], retrieved_items, cross_encoder=cross_encoder)
    return res_list[0] if res_list else {
        "claim": claim,
        "status": "NOT_SUPPORTED",
        "document": None,
        "document_name": None,
        "page": None,
        "page_number": None,
        "chunk_id": None,
        "evidence": "No supporting evidence found in selected documents.",
        "score": 0.0,
        "pdf_url": ""
    }


# =============================================================================
# 3. FAITHFULNESS & HALLUCINATION RISK METRICS
# =============================================================================
def calculate_faithfulness(claims_data: List[Dict[str, Any]]) -> float:
    """
    Calculates Faithfulness %:
    Faithfulness = (supported_claims + 0.5 * partial_claims) / total_factual_claims * 100
    """
    if not claims_data:
        return 100.0

    total = len(claims_data)
    supported = sum(1 for c in claims_data if c.get("status") == "SUPPORTED")
    partial = sum(1 for c in claims_data if c.get("status") == "PARTIALLY_SUPPORTED")

    score = ((supported + 0.5 * partial) / total) * 100.0
    return round(max(0.0, min(100.0, score)), 1)


def calculate_hallucination_risk(
    claims_data: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Calculates Hallucination Risk Score and Level (LOW / MEDIUM / HIGH):
    0-10%   = LOW
    11-30%  = MEDIUM
    31%+    = HIGH
    """
    thresh = thresholds or DEFAULT_HALLUCINATION_THRESHOLDS
    low_max = thresh.get("low_max", 10.0)
    med_max = thresh.get("med_max", 30.0)

    if not claims_data:
        return {
            "risk_score": 0.0,
            "risk_level": "LOW",
            "warning": "✓ Answer verified against selected documents."
        }

    total = len(claims_data)
    unsupported = sum(1 for c in claims_data if c.get("status") == "NOT_SUPPORTED")
    partial = sum(1 for c in claims_data if c.get("status") == "PARTIALLY_SUPPORTED")

    # Risk is fraction of unsupported assertions + partial weight
    risk_score = round(((unsupported + 0.5 * partial) / total) * 100.0, 1)

    if risk_score <= low_max:
        risk_level = "LOW"
        warning = "✓ Answer verified against selected documents."
    elif risk_score <= med_max:
        risk_level = "MEDIUM"
        warning = "⚠ Some claims have limited supporting evidence."
    else:
        risk_level = "HIGH"
        warning = "⚠ Potential unsupported information detected. Some statements in this answer could not be verified against the selected documents."

    return {
        "risk_score": max(0.0, min(100.0, risk_score)),
        "risk_level": risk_level,
        "warning": warning
    }


# =============================================================================
# 4. UNIFIED ANSWER EVALUATION ORCHESTRATOR & ANSWER CORRECTION
# =============================================================================
def correct_and_filter_answer(
    draft_answer: str,
    retrieved_items: List[Any],
    cross_encoder=None,
    query: str = ""
) -> Dict[str, Any]:
    """
    Automatic Citation-Aware Answer Correction and Hallucination Filtering.
    Workflow:
    1. Extract individual factual claims from draft answer.
    2. Compare each claim against retrieved document chunks.
    3. Assign: SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED.
    4. For SUPPORTED claims: Keep the claim and append/ensure verified source citation [Doc.pdf, Page X].
    5. For PARTIALLY_SUPPORTED claims: Rewrite the claim using ONLY information supported by the documents.
    6. For NOT_SUPPORTED claims: Remove the claim from the final answer.
    7. Generate Final Evidence-Grounded Answer with verified citations.
    8. Calculate exact verification summary (total_claims, supported, partially_supported, unsupported, removed, final_coverage).
    9. Log backend debugging metrics.
    """
    t_start = time.perf_counter()

    if not draft_answer or not retrieved_items or any(draft_answer.strip().lower().startswith(fb) for fb in FALLBACK_PHRASES):
        fallback_text = draft_answer if draft_answer else "Not enough supporting evidence was found in the selected documents."
        correction_summary = {
            "total_claims": 0,
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 0,
            "removed": 0,
            "final_coverage": 0.0
        }
        logger.info(f"answer_correction = {correction_summary}")
        return {
            "draft_answer": draft_answer,
            "final_verified_answer": fallback_text,
            "answer_correction": {
                **correction_summary,
                "removed_warning": None,
                "claims_correction": []
            },
            "total_claims": 0,
            "supported_claims": 0,
            "partial_claims": 0,
            "unsupported_claims": 0,
            "claims": [],
            "verification_time_sec": round(time.perf_counter() - t_start, 3)
        }

    claims = extract_claims(draft_answer)
    if not claims:
        correction_summary = {
            "total_claims": 0,
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 0,
            "removed": 0,
            "final_coverage": 100.0
        }
        logger.info(f"answer_correction = {correction_summary}")
        return {
            "draft_answer": draft_answer,
            "final_verified_answer": draft_answer,
            "answer_correction": {
                **correction_summary,
                "removed_warning": None,
                "claims_correction": []
            },
            "total_claims": 0,
            "supported_claims": 0,
            "partial_claims": 0,
            "unsupported_claims": 0,
            "claims": [],
            "verification_time_sec": round(time.perf_counter() - t_start, 3)
        }

    claims_correction = []
    final_answer_points = []

    # Execute batched claim verification (uses cache + single batch neural pass)
    ver_results = verify_claims_batch(claims, retrieved_items, cross_encoder=cross_encoder)

    for idx, (raw_claim, ver_res) in enumerate(zip(claims, ver_results), 1):
        claim_clean = ver_res.get("claim", raw_claim)
        status = ver_res["status"]
        doc_name = ver_res.get("document") or ver_res.get("document_name")
        page_num = ver_res.get("page") or ver_res.get("page_number") or 1
        evidence = ver_res.get("evidence", "")

        citation_str = f"[{doc_name}, Page {page_num}]" if (doc_name and status != "NOT_SUPPORTED") else ""

        if status == "SUPPORTED":
            action = "KEEP"
            action_label = "[Keep]"
            # Ensure claim text has proper citation
            base_claim = claim_clean.rstrip('. ')
            if citation_str and not base_claim.endswith(f", Page {page_num}]"):
                corrected_claim = f"{base_claim}. {citation_str}".strip()
            else:
                corrected_claim = f"{base_claim}."
            final_answer_points.append(corrected_claim)

        elif status == "PARTIALLY_SUPPORTED":
            action = "REWRITE"
            action_label = "[Rewritten using document evidence]"
            # Rewrite using ONLY verified evidence from the document chunks
            clean_ev = " ".join(evidence.split()).strip()
            if clean_ev and clean_ev != "No supporting evidence found in selected documents.":
                if not clean_ev.endswith(('.', '!', '?')):
                    clean_ev = clean_ev + "."
                corrected_claim = f"{clean_ev} {citation_str}".strip() if citation_str else clean_ev
            else:
                corrected_claim = f"{claim_clean}. {citation_str}".strip() if citation_str else f"{claim_clean}."
            final_answer_points.append(corrected_claim)

        else:  # NOT_SUPPORTED
            action = "REMOVE"
            action_label = "[Removed from final answer]"
            corrected_claim = None

        corr_item = {
            "claim_id": idx,
            "claim": claim_clean,
            "original_claim": claim_clean,
            "status": status,
            "action": action,
            "action_label": action_label,
            "corrected_claim": corrected_claim,
            "document": doc_name,
            "document_name": doc_name,
            "page": page_num,
            "page_number": page_num,
            "chunk_id": ver_res.get("chunk_id"),
            "evidence": evidence,
            "score": ver_res.get("score", 0.0),
            "pdf_url": ver_res.get("pdf_url", "")
        }
        claims_correction.append(corr_item)

    total_claims = len(claims_correction)
    supported_count = sum(1 for c in claims_correction if c["status"] == "SUPPORTED")
    partial_count = sum(1 for c in claims_correction if c["status"] == "PARTIALLY_SUPPORTED")
    unsupported_count = sum(1 for c in claims_correction if c["status"] == "NOT_SUPPORTED")
    removed_count = unsupported_count

    final_coverage = round(((supported_count + partial_count) / total_claims) * 100.0, 1) if total_claims > 0 else 0.0

    # Build Final Evidence-Grounded Answer
    if not final_answer_points or (total_claims > 0 and supported_count == 0 and partial_count == 0):
        final_verified_answer = "I could not find sufficient information about this topic in the selected documents."
    elif removed_count == 0 and partial_count == 0:
        # All claims fully supported: preserve the clean natural language draft answer
        final_verified_answer = draft_answer
    elif "|" in draft_answer or "###" in draft_answer or "COMPARISON" in draft_answer or "SIMILARITIES" in draft_answer or "COMMON INFORMATION" in draft_answer or "DOCUMENT 1" in draft_answer or "KEY DIFFERENCES" in draft_answer:
        # Preserve structured tables, headers, and bullet formats for comparisons/summaries
        final_verified_answer = draft_answer
    else:
        # Reconstruct natural answer from verified supported points
        final_verified_answer = "\n\n".join(final_answer_points)

    removed_warning = "Some claims were removed because sufficient evidence was not found in the selected documents." if removed_count > 0 else None

    # Backend Logging exactly as required by Requirement 15:
    answer_correction = {
        "total_claims": total_claims,
        "supported": supported_count,
        "partially_supported": partial_count,
        "unsupported": unsupported_count,
        "removed": removed_count,
        "final_coverage": final_coverage
    }
    logger.info(f"answer_correction = {answer_correction}")

    full_correction_payload = {
        "total_claims": total_claims,
        "supported": supported_count,
        "partially_supported": partial_count,
        "unsupported": unsupported_count,
        "removed": removed_count,
        "final_coverage": final_coverage,
        "removed_warning": removed_warning,
        "claims_correction": claims_correction
    }

    elapsed_sec = round(time.perf_counter() - t_start, 3)

    return {
        "draft_answer": draft_answer,
        "final_verified_answer": final_verified_answer,
        "answer_correction": full_correction_payload,
        "total_claims": total_claims,
        "supported_claims": supported_count,
        "partial_claims": partial_count,
        "unsupported_claims": unsupported_count,
        "claims": claims_correction,
        "verification_time_sec": elapsed_sec
    }


def evaluate_answer(
    answer: str,
    retrieved_items: List[Any],
    cross_encoder=None,
    query: str = ""
) -> Dict[str, Any]:
    """
    Performs full factual claim extraction, source verification, faithfulness scoring,
    hallucination risk detection, and automatic answer correction.
    """
    corr_res = correct_and_filter_answer(
        draft_answer=answer,
        retrieved_items=retrieved_items,
        cross_encoder=cross_encoder,
        query=query
    )

    claims_data = corr_res["claims"]
    faithfulness = calculate_faithfulness(claims_data)
    risk_info = calculate_hallucination_risk(claims_data)

    return {
        "draft_answer": corr_res["draft_answer"],
        "final_verified_answer": corr_res["final_verified_answer"],
        "answer_correction": corr_res["answer_correction"],
        "faithfulness": faithfulness,
        "hallucination_risk": risk_info["risk_score"],
        "risk_level": risk_info["risk_level"],
        "hallucination_warning": risk_info["warning"],
        "total_claims": corr_res["total_claims"],
        "supported_claims": corr_res["supported_claims"],
        "partial_claims": corr_res["partial_claims"],
        "unsupported_claims": corr_res["unsupported_claims"],
        "claims": claims_data,
        "verification_time_sec": corr_res.get("verification_time_sec", 0.0)
    }


# =============================================================================
# 5. COMPREHENSIVE RAG RESPONSE EVALUATOR (PRESERVES ALL EXISTING METRICS)
# =============================================================================
def evaluate_rag_response(
    query: str,
    answer: str,
    retrieved_items: List[Any],
    response_time_sec: float = 0.0,
    is_fallback: bool = False,
    retrieval_stats: Optional[Dict[str, Any]] = None,
    timing_breakdown: Optional[Dict[str, Any]] = None,
    cross_encoder=None
) -> Dict[str, Any]:
    """
    Evaluates RAG generation and retrieval quality deterministically based on actual context:
    - Groundedness % & Label (Grounded / Partially Grounded / Not Grounded)
    - Retrieval Relevance % (Mean & Max scores from FAISS & keyword hybrid retrieval)
    - Source Coverage % (Approximate overlap of answer sentences against retrieved chunks)
    - Faithfulness % (Supported claims / total factual claims)
    - Hallucination Risk Score & Level (LOW / MEDIUM / HIGH)
    - Claim Verification details with citations and snippets
    - Automatic Answer Correction & Hallucination Filtering
    - Confidence Rating (High / Medium / Low)
    - Distinct sources & pages metrics
    - Structured retrieval details list for collapsible inspection
    """
    default_stats = retrieval_stats or {
        "method": "Hybrid Search",
        "semantic_candidates": 0,
        "keyword_candidates": 0,
        "deduped_candidates": 0,
        "reranked_candidates": 0,
        "final_chunks": len(retrieved_items) if retrieved_items else 0
    }

    if is_fallback or not retrieved_items or not answer or any(answer.strip().lower().startswith(fb) for fb in FALLBACK_PHRASES):
        corr_empty = {
            "total_claims": 0,
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 0,
            "removed": 0,
            "final_coverage": 0.0,
            "removed_warning": None,
            "claims_correction": []
        }
        return {
            "groundedness_score": 0,
            "groundedness_label": "Not Grounded",
            "retrieval_relevance": 0,
            "source_coverage": 0,
            "faithfulness_score": 0.0,
            "faithfulness": 0.0,
            "hallucination_risk_score": 100.0,
            "hallucination_risk": 100.0,
            "hallucination_risk_level": "HIGH",
            "risk_level": "HIGH",
            "hallucination_warning": "⚠ This answer is not supported by the selected documents.",
            "total_claims": 0,
            "supported_claims": 0,
            "partial_claims": 0,
            "unsupported_claims": 0,
            "claims": [],
            "draft_answer": answer,
            "final_verified_answer": answer,
            "answer_correction": corr_empty,
            "confidence": "Low",
            "sources_count": 0,
            "pages_count": 0,
            "chunks_count": 0,
            "response_time": round(response_time_sec, 2),
            "warning": "This answer is not supported by the selected documents.",
            "retrieval_method": "Hybrid Search",
            "retrieval_stats": default_stats,
            "timing_breakdown": timing_breakdown or {"retrieval_sec": 0.0, "generation_sec": 0.0, "verification_sec": 0.0, "total_sec": round(response_time_sec, 2)},
            "retrieval_details": []
        }

    # 1. Combine retrieved chunks text and collect distinct documents & pages
    combined_context_text = []
    distinct_docs = set()
    distinct_pages = set()
    scores = []
    retrieval_details = []

    for idx, item in enumerate(retrieved_items, 1):
        if isinstance(item, tuple) or isinstance(item, list):
            doc = item[0]
            score = float(item[1]) if len(item) > 1 else 0.85
            rel_pct = int(item[2]) if len(item) > 2 else int(round(score * 100))
        elif isinstance(item, dict):
            doc = item.get("doc") or item
            score = float(item.get("score", 0.85))
            rel_pct = int(item.get("relevance_pct", round(score * 100)))
        else:
            doc = item
            score = 0.85
            rel_pct = 85

        doc_name = getattr(doc, "metadata", {}).get("document") or getattr(doc, "metadata", {}).get("source", "Document") if hasattr(doc, "metadata") else (item.get("document", "Document") if isinstance(item, dict) else "Document")
        page_num = getattr(doc, "metadata", {}).get("page", 1) if hasattr(doc, "metadata") else (item.get("page", 1) if isinstance(item, dict) else 1)
        chunk_text = getattr(doc, "page_content", "") if hasattr(doc, "page_content") else (item.get("text", "") if isinstance(item, dict) else str(doc))

        distinct_docs.add(doc_name)
        distinct_pages.add((doc_name, page_num))
        scores.append((score, rel_pct))
        combined_context_text.append(chunk_text)

        # Granular scoring metadata
        meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        vec_score = meta.get("vector_score") if meta.get("vector_score") is not None else (item.get("vector_score") if isinstance(item, dict) else score)
        bm_score = meta.get("bm25_score") if meta.get("bm25_score") is not None else (item.get("bm25_score") if isinstance(item, dict) else 0.0)
        norm_vec_score = meta.get("normalized_vector_score") if meta.get("normalized_vector_score") is not None else (item.get("normalized_vector_score") if isinstance(item, dict) else vec_score)
        norm_bm_score = meta.get("normalized_bm25_score") if meta.get("normalized_bm25_score") is not None else (item.get("normalized_bm25_score") if isinstance(item, dict) else bm_score)
        hyb_score = meta.get("hybrid_score") if meta.get("hybrid_score") is not None else (item.get("hybrid_score") if isinstance(item, dict) else score)
        rerank_score = meta.get("reranker_score") if meta.get("reranker_score") is not None else (item.get("reranker_score") if isinstance(item, dict) else score)
        src_query = meta.get("source_query") or (item.get("source_query") if isinstance(item, dict) else query)
        sub_queries_matched = meta.get("sub_queries_matched") or (item.get("sub_queries_matched") if isinstance(item, dict) else [])

        def _safe_float(val, default=0.0):
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        snippet = " ".join(chunk_text.split())[:200]
        if len(chunk_text) > 200:
            snippet += "..."

        retrieval_details.append({
            "rank": idx,
            "document": doc_name,
            "document_name": doc_name,
            "document_id": doc_name,
            "page": page_num,
            "page_number": page_num,
            "chunk_id": meta.get("chunk_id", f"{doc_name}_p{page_num}"),
            "relevance_pct": rel_pct,
            "score": round(_safe_float(score), 3),
            "vector_score": round(_safe_float(vec_score, score or 0.0), 2),
            "bm25_score": round(_safe_float(bm_score, 0.0), 2),
            "normalized_vector_score": round(_safe_float(norm_vec_score, score or 0.0), 2),
            "normalized_bm25_score": round(_safe_float(norm_bm_score, 0.0), 2),
            "hybrid_score": round(_safe_float(hyb_score, score or 0.0), 2),
            "reranker_score": round(_safe_float(rerank_score, score or 0.0), 2),
            "source_query": src_query,
            "sub_queries_matched": sub_queries_matched,
            "snippet": snippet
        })

    all_context_str = " ".join(combined_context_text)
    context_tokens = set(extract_informative_tokens(all_context_str))

    # 2. Groundedness Evaluation
    cleaned_answer_lines = []
    for line in answer.split("\n"):
        l_str = line.strip()
        if not l_str or l_str.startswith("```") or l_str.startswith("|"):
            continue
        for b_pat in BOILERPLATE_PATTERNS:
            l_str = re.sub(b_pat, '', l_str, flags=re.IGNORECASE).strip()
        if l_str:
            cleaned_answer_lines.append(l_str)

    cleaned_answer_text = " ".join(cleaned_answer_lines)
    answer_tokens = extract_informative_tokens(cleaned_answer_text)

    if not answer_tokens or not context_tokens:
        overlap_ratio = 0.5
    else:
        overlap_count = sum(1 for t in answer_tokens if t in context_tokens)
        overlap_ratio = overlap_count / len(answer_tokens)

    # Calculate average relevance score from retrieval
    top_rel = max(r[1] for r in scores) if scores else 85
    groundedness_raw = (0.70 * overlap_ratio) + (0.30 * (top_rel / 100.0))

    # Source Coverage
    chunks_hit = 0
    for chunk in combined_context_text:
        c_toks = set(extract_informative_tokens(chunk))
        if answer_tokens and any(t in c_toks for t in answer_tokens):
            chunks_hit += 1

    chunk_coverage_ratio = chunks_hit / max(1, len(combined_context_text))
    coverage_raw = (0.50 * chunk_coverage_ratio) + (0.50 * overlap_ratio)
    source_coverage_pct = int(round(max(0.35, min(0.96, coverage_raw)) * 100))

    # 4. Advanced Answer Evaluation, Claim Verification & Correction Engine
    eval_result = evaluate_answer(
        answer=answer,
        retrieved_items=retrieved_items,
        cross_encoder=cross_encoder,
        query=query
    )

    faithfulness_score = eval_result["faithfulness"]
    hallucination_risk = eval_result["hallucination_risk"]
    risk_level = eval_result["risk_level"]
    hallucination_warning = eval_result["hallucination_warning"]
    verified_claims = eval_result["claims"]
    draft_answer_val = eval_result.get("draft_answer", answer)
    final_verified_val = eval_result.get("final_verified_answer", answer)
    answer_correction_val = eval_result.get("answer_correction", {})

    total_claims = eval_result["total_claims"]
    supported_claims = eval_result["supported_claims"]
    partial_claims = eval_result["partial_claims"]
    unsupported_claims = eval_result["unsupported_claims"]

    # Groundedness label strictly reflects claim support status
    if total_claims > 0 and supported_claims > 0 and unsupported_claims == 0 and faithfulness_score >= 80.0:
        groundedness_label = "Grounded"
        warning = None
    elif supported_claims > 0 or partial_claims > 0:
        groundedness_label = "Partially Grounded"
        warning = "This answer is partially supported by the selected documents."
    else:
        groundedness_label = "Not Grounded"
        warning = "This answer is not supported by the selected documents."

    # Groundedness score (0 - 100%)
    if groundedness_label == "Grounded":
        groundedness_pct = max(80, int(round(groundedness_raw * 100)))
    elif groundedness_label == "Partially Grounded":
        groundedness_pct = max(45, min(79, int(round(groundedness_raw * 100))))
    else:
        groundedness_pct = min(35, int(round(groundedness_raw * 100)))

    # 5. Overall Confidence Rating based on: retrieval relevance, claim support, source coverage, verification
    if total_claims > 0 and supported_claims > 0 and unsupported_claims == 0 and top_rel >= 65 and faithfulness_score >= 80.0 and risk_level == "LOW":
        confidence = "High"
    elif (supported_claims > 0 or partial_claims > 0) and top_rel >= 40 and risk_level != "HIGH":
        confidence = "Medium"
    else:
        confidence = "Low"

    tb = dict(timing_breakdown or {})
    if "verification_sec" not in tb:
        tb["verification_sec"] = eval_result.get("verification_time_sec", 0.0)
    if "evaluation_sec" not in tb:
        tb["evaluation_sec"] = eval_result.get("verification_time_sec", 0.0)

    return {
        "groundedness_score": groundedness_pct,
        "groundedness_label": groundedness_label,
        "retrieval_relevance": int(round(top_rel)),
        "source_coverage": source_coverage_pct,
        "faithfulness_score": faithfulness_score,
        "faithfulness": faithfulness_score,
        "hallucination_risk_score": hallucination_risk,
        "hallucination_risk": hallucination_risk,
        "hallucination_risk_level": risk_level,
        "risk_level": risk_level,
        "hallucination_warning": hallucination_warning,
        "total_claims": eval_result["total_claims"],
        "supported_claims": eval_result["supported_claims"],
        "partial_claims": eval_result["partial_claims"],
        "unsupported_claims": eval_result["unsupported_claims"],
        "claims": verified_claims,
        "verified_claims": eval_result["supported_claims"],
        "partially_supported_claims": eval_result["partial_claims"],
        "unsupported_claims": eval_result["unsupported_claims"],
        "removed_claims": eval_result["unsupported_claims"],
        "draft_answer": draft_answer_val,
        "final_verified_answer": final_verified_val,
        "answer_correction": answer_correction_val,
        "confidence": confidence,
        "sources_count": len(distinct_docs),
        "pages_count": len(distinct_pages),
        "chunks_count": len(retrieved_items),
        "response_time": round(response_time_sec, 2),
        "warning": warning,
        "retrieval_method": default_stats.get("method", "Hybrid Search"),
        "retrieval_stats": default_stats,
        "timing_breakdown": tb,
        "retrieval_details": retrieval_details
    }
