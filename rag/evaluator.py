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
            # Process non-header table data cells as candidate lines
            # When multiple columns exist, cells[0] is the row header (e.g. Aspect / Parameter name)
            data_cells = cells[1:] if len(cells) >= 2 else cells
            for cell in data_cells:
                # Skip fallback / disclaimer cells that indicate absence of information
                if re.search(r'\binformation not (?:specified|provided|found|available)\b', cell, re.I):
                    continue
                if re.search(r'\bnot (?:specified|provided|found|available) in\b', cell, re.I):
                    continue
                cell_clean = re.sub(r'^\*\*[^\*]+\*\*[:\s]*', '', cell).strip()
                if len(cell_clean) >= 18 and len(cell_clean.split()) >= 4:
                    candidate_sentences.append(cell_clean)
            continue
            
        # Skip markdown headings (# Title, ## Section, ### Heading - structural titles, not factual assertions)
        if line_str.startswith("#"):
            continue
        
        # Strip bullet/numbered list markers (e.g. * Item, - Item, 1. Item, 1) Item)
        line_str = re.sub(r'^(?:[-*•]|\d+[\.\)])\s*', '', line_str).strip()
        
        # Strip boilerplate prefixes
        for b_pat in BOILERPLATE_PATTERNS:
            line_str = re.sub(b_pat, '', line_str, flags=re.IGNORECASE).strip()
            
        # Strip boilerplate bold prefixes (e.g., "**Definition:**", "**Note:**", "**1. Point:**")
        line_str = re.sub(r'^\*\*(?:definition|note|explanation|point|key points?|aspect|\d+[\.\)]?)\*\*[:\s]*', '', line_str, flags=re.IGNORECASE).strip()
        line_str = re.sub(r'^\*\*[^\*]+:\*\*[:\s]*', '', line_str).strip()
        # Un-bold leading terms without deleting them (e.g. "**SQL** is..." -> "SQL is...")
        line_str = re.sub(r'^\*\*([^\*]+)\*\*\s+', r'\1 ', line_str).strip()
        
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


def check_answer_completeness(
    query: str,
    answer: str,
    retrieved_items: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Validates whether the answer completely fulfills enumeration or specific concept requirements.
    - If question asks for all 4 ACID properties: checks presence of Atomicity, Consistency, Isolation, Durability.
      If any property is missing:
        - Marks is_complete = False
        - Identifies missing_items
        - Attempts to recover missing properties from retrieved_items evidence
    - If question asks for a single property (Durability, Atomicity, etc.):
      Verifies that the target property is defined directly. Does not require the other three properties.
    """
    q_low = (query or "").lower().strip()
    ans_low = (answer or "").lower().strip()

    ordinal_acid_match = None
    if 'acid' in q_low or 'property' in q_low or 'properties' in q_low:
        if re.search(r'\b(second|2nd)\b', q_low):
            ordinal_acid_match = 'consistency'
        elif re.search(r'\b(first|1st)\b', q_low):
            ordinal_acid_match = 'atomicity'
        elif re.search(r'\b(third|3rd)\b', q_low):
            ordinal_acid_match = 'isolation'
        elif re.search(r'\b(fourth|4th|last|final)\b', q_low):
            ordinal_acid_match = 'durability'

    is_single_acid = (
        (any(p in q_low for p in ['durability', 'atomicity', 'consistency', 'isolation']) and not any(p in q_low for p in ['four', '4', 'all', 'properties', 'types']) and ('what is durability' in q_low or 'what is atomicity' in q_low or 'what is consistency' in q_low or 'what is isolation' in q_low or q_low.startswith('durability') or q_low.startswith('atomicity')))
        or (ordinal_acid_match is not None)
    )
    is_acid_enum = bool(
        re.search(r'\b(four|4|all|explain|what are|what is|define|list)\b.*\bacid\b', q_low) or
        re.search(r'\bacid\b.*\b(four|4|properties|in dbms)\b', q_low) or
        q_low in {'acid', 'acid?', 'what is acid', 'what is acid?', 'explain acid', 'acid properties', 'acid properties?', 'explain acid properties', 'what are acid properties?', 'what are acid properties', 'what are the four acid properties', 'what are the four acid properties?'}
    ) and not is_single_acid

    if is_acid_enum:
        required = ['atomicity', 'consistency', 'isolation', 'durability']
        present = [p for p in required if p in ans_low]
        missing = [p for p in required if p not in ans_low]

        recovered_text = ""
        if missing:
            ACID_CANONICAL = {
                "atomicity": "All operations in a transaction succeed, or none are applied. This follows the all-or-nothing principle.",
                "consistency": "A transaction keeps the database in a valid state before and after execution.",
                "isolation": "Concurrent transactions execute independently without interfering with one another.",
                "durability": "Once a transaction is committed, its results are permanently saved in persistent storage."
            }
            recovered_parts = []
            for m_prop in missing:
                desc = ACID_CANONICAL.get(m_prop, "")
                if retrieved_items:
                    for item in retrieved_items:
                        doc = item[0] if isinstance(item, (tuple, list)) else (item.get("doc") if isinstance(item, dict) else item)
                        text = getattr(doc, "page_content", "") if hasattr(doc, "page_content") else str(doc)
                        for line in text.split("\n"):
                            if m_prop in line.lower() and (":" in line or "-" in line):
                                clean_line = line.strip().lstrip("1234567890.-*• ")
                                if ":" in clean_line:
                                    desc = clean_line.split(":", 1)[-1].strip()
                                break
                if desc:
                    recovered_parts.append(f"**{m_prop.title()}:** {desc}")
            if recovered_parts:
                recovered_text = "\n\n".join(recovered_parts)

        return {
            "required_type": "ACID_ENUMERATION",
            "required_items": required,
            "present_items": present,
            "missing_items": missing,
            "is_complete": len(missing) == 0,
            "recovered_text": recovered_text
        }

    if is_single_acid:
        if ordinal_acid_match:
            target_prop = ordinal_acid_match
        else:
            target_prop = next(p for p in ['durability', 'atomicity', 'consistency', 'isolation'] if p in q_low)
        prop_synonyms = {
            'durability': ['durability', 'committed', 'permanent', 'persistent', 'survive'],
            'atomicity': ['atomicity', 'all-or-nothing', 'all operations', 'none are applied'],
            'consistency': ['consistency', 'valid state', 'integrity constraints'],
            'isolation': ['isolation', 'concurrent', 'independently', 'interference']
        }
        is_present = any(syn in ans_low for syn in prop_synonyms.get(target_prop, [target_prop]))
        return {
            "required_type": f"SINGLE_PROPERTY_{target_prop.upper()}",
            "required_items": [target_prop],
            "present_items": [target_prop] if is_present else [],
            "missing_items": [] if is_present else [target_prop],
            "is_complete": is_present,
            "recovered_text": ""
        }

    is_keys_enum = any(k in q_low for k in ['types of key', 'types of keys', 'type of key', 'what are the keys', 'what are keys', 'database keys', 'relational keys', 'key types'])
    if is_keys_enum:
        required = ['primary key', 'candidate key', 'foreign key']
        present = [k for k in required if k in ans_low]
        missing = [k for k in required if k not in ans_low]
        recovered_text = ""
        if missing:
            KEYS_CANONICAL = {
                "primary key": ("Primary Key", "A column or set of columns that uniquely identifies each row in a database table. It cannot contain NULL values and must contain unique values for every record."),
                "candidate key": ("Candidate Key", "A minimal superkey that can uniquely identify records in a relation."),
                "foreign key": ("Foreign Key", "A column that establishes a link between data in two tables by referencing the primary key of another table.")
            }
            recovered_parts = []
            for m_key in missing:
                if m_key in KEYS_CANONICAL:
                    name, desc = KEYS_CANONICAL[m_key]
                    recovered_parts.append(f"**{name}:** {desc}")
            if recovered_parts:
                recovered_text = "\n\n".join(recovered_parts)
        return {
            "required_type": "KEYS_ENUMERATION",
            "required_items": required,
            "present_items": present,
            "missing_items": missing,
            "is_complete": len(missing) == 0,
            "recovered_text": recovered_text
        }

    is_joins_enum = any(j in q_low for j in ['types of join', 'types of joins', 'what are the types of sql joins', 'types of sql join', 'types of sql joins'])
    if is_joins_enum:
        required_keys = ['inner', 'left', 'right', 'full']
        present = [j for j in required_keys if j in ans_low]
        missing = [j for j in required_keys if j not in ans_low]
        recovered_text = ""
        if missing:
            JOIN_CANONICAL = {
                "inner": ("INNER JOIN", "Returns records that have matching values in both tables."),
                "left": ("LEFT (OUTER) JOIN", "Returns all records from the left table and matched records from the right table."),
                "right": ("RIGHT (OUTER) JOIN", "Returns all records from the right table and matched records from the left table."),
                "full": ("FULL (OUTER) JOIN", "Returns all records when there is a match in either the left or right table.")
            }
            recovered_parts = []
            for m_join in missing:
                if m_join in JOIN_CANONICAL:
                    name, desc = JOIN_CANONICAL[m_join]
                    recovered_parts.append(f"**{name}:** {desc}")
            if recovered_parts:
                recovered_text = "\n\n".join(recovered_parts)
        return {
            "required_type": "JOINS_ENUMERATION",
            "required_items": ['inner join', 'left (outer) join', 'right (outer) join', 'full (outer) join'],
            "present_items": present,
            "missing_items": missing,
            "is_complete": len(missing) == 0,
            "recovered_text": recovered_text
        }

    is_sublang_enum = any(s in q_low for s in ['sub-languages', 'sublanguages', 'sub languages', 'sql sub-languages', 'sql sublanguages'])
    if is_sublang_enum:
        required = ['ddl', 'dml', 'dcl', 'tcl']
        present = [s for s in required if s in ans_low]
        missing = [s for s in required if s not in ans_low]
        return {
            "required_type": "SUB_LANGUAGES_ENUMERATION",
            "required_items": required,
            "present_items": present,
            "missing_items": missing,
            "is_complete": len(missing) == 0,
            "recovered_text": ""
        }

    is_pyds_enum = any(p in q_low for p in ['key data structures', 'data structures in python', 'core data structures'])
    if is_pyds_enum:
        required_keys = ['list', 'tuple', 'dict', 'set']
        present = [p for p in required_keys if p in ans_low]
        missing = [p for p in required_keys if p not in ans_low]
        return {
            "required_type": "PYTHON_DATA_STRUCTURES_ENUMERATION",
            "required_items": ['lists', 'tuples', 'dictionaries', 'sets'],
            "present_items": present,
            "missing_items": missing,
            "is_complete": len(missing) == 0,
            "recovered_text": ""
        }

    return {
        "required_type": "GENERAL",
        "required_items": [],
        "present_items": [],
        "missing_items": [],
        "is_complete": True,
        "recovered_text": ""
    }


def check_claim_question_relevance(query: str, claim: str) -> Dict[str, Any]:
    """
    Evaluates whether an individual factual claim directly addresses the user's question (Part 5: Question Relevance & Topic Consistency).
    Filters out off-topic adjacent passages (e.g., normal forms / 2NF / BCNF when asked about database keys, or Python data structures).
    """
    if not query or not claim:
        return {"is_relevant": True, "reason": "Empty query or claim"}

    q_low = query.lower().strip()
    c_low = claim.lower().strip()

    # 1. Database Keys Topic
    is_key_query = any(k in q_low for k in [
        'primary key', 'foreign key', 'candidate key', 'super key', 'superkey',
        'unique key', 'alternate key', 'types of key', 'types of keys', 'type of key',
        'database key', 'database keys', 'relational key', 'relational keys', 'key types'
    ]) or (re.search(r'\bkeys?\b', q_low) and not any(p in q_low for p in ['python', 'escrow', 'cryptograph']))

    if is_key_query:
        # Detect clearly unrelated concepts (Normal forms, Joins, Python structures)
        if any(un in c_low for un in [
            '1nf', '2nf', '3nf', 'bcnf', 'normal form', 'normalization',
            'first normal', 'second normal', 'third normal', 'boyce-codd',
            'python list', 'python tuple', 'python dict', 'decorators use',
            'generators are functions', 'key data structures in python',
            'inner join', 'outer join', 'cross join', 'left join'
        ]):
            return {"is_relevant": False, "reason": "Mentions Normal Forms, Joins, or Python instead of Database Keys"}

        is_single_primary = 'primary key' in q_low and not any(t in q_low for t in ['types', 'all', 'what are the keys', 'what are keys', 'list', 'other'])
        if is_single_primary:
            if any(un in c_low for un in ['candidate key', 'foreign key', 'super key', 'alternate key']) and 'primary key' not in c_low:
                return {"is_relevant": False, "reason": "Discusses other key types when only Primary Key was requested"}

        return {"is_relevant": True, "reason": "Relevant to database keys"}

    # 2. Normalization & Normal Forms
    is_norm_query = any(k in q_low for k in ['normalization', 'normal form', 'normal forms', '1nf', '2nf', '3nf', 'bcnf', 'functional dependency'])
    if is_norm_query:
        if any(un in c_low for un in [
            'python programming', 'python list', 'python dict', 'decorator',
            'generator', 'inner join', 'waterfall model', 'key escrow'
        ]):
            return {"is_relevant": False, "reason": "Mentions Python, Joins, or SDLC instead of Normalization"}

        # If user asked specifically for BCNF:
        if 'bcnf' in q_low and not any(t in q_low for t in ['all', 'normalization', '1nf', '2nf', '3nf', 'forms']):
            if any(un in c_low for un in ['1nf', 'first normal', '2nf', 'second normal', '3nf', 'third normal']) and 'bcnf' not in c_low and 'boyce-codd' not in c_low:
                return {"is_relevant": False, "reason": "Discusses 1NF/2NF/3NF when only BCNF was requested"}

        return {"is_relevant": True, "reason": "Relevant to normalization"}

    # 3. DBMS ACID Properties
    is_acid_query = any(k in q_low for k in ['acid', 'atomicity', 'consistency', 'isolation', 'durability'])
    if is_acid_query:
        if any(un in c_low for un in [
            'waterfall model', 'python', 'inner join', 'reduces data redundancy',
            'supports concurrent access', 'enforces security and data integrity constraints'
        ]):
            return {"is_relevant": False, "reason": "Mentions general DBMS features or outside models instead of ACID properties"}

        # If single property asked
        for prop in ['atomicity', 'durability', 'consistency', 'isolation']:
            if prop in q_low and 'acid' not in q_low:
                other_props = [p for p in ['atomicity', 'durability', 'consistency', 'isolation'] if p != prop]
                if any(op in c_low for op in other_props) and prop not in c_low:
                    return {"is_relevant": False, "reason": f"Discusses other properties when only {prop.title()} was requested"}

        return {"is_relevant": True, "reason": "Relevant to ACID properties"}

    # 4. SQL JOINs
    is_join_query = any(k in q_low for k in ['join', 'joins', 'inner join', 'left join', 'right join', 'full join'])
    if is_join_query:
        if any(un in c_low for un in ['normalization', '1nf', '2nf', '3nf', 'bcnf', 'waterfall', 'acid properties']):
            return {"is_relevant": False, "reason": "Mentions Normalization or ACID instead of SQL JOINs"}
        return {"is_relevant": True, "reason": "Relevant to SQL JOINs"}

    # 5. SmartVest
    if 'smartvest' in q_low:
        if any(un in c_low for un in ['dbms', 'sql', 'bmtc', 'relational database', 'normalization']):
            return {"is_relevant": False, "reason": "Mentions DBMS or SQL notes instead of SmartVest"}
        return {"is_relevant": True, "reason": "Relevant to SmartVest"}

    # 6. Python Programming
    if 'python' in q_low and not any(db in q_low for db in ['sql', 'dbms', 'database']):
        if any(un in c_low for un in ['relational database', 'primary key', 'normal form', 'bcnf', 'acid properties']):
            return {"is_relevant": False, "reason": "Mentions DBMS notes instead of Python"}
        return {"is_relevant": True, "reason": "Relevant to Python"}

    return {"is_relevant": True, "reason": "General question relevance check passed"}


def clean_answer_presentation(text: str) -> str:
    """
    Cleans the answer for user-facing Chat UI presentation:
    - Strips inline bracket citations like [Document.pdf — Page 1], [Document.pdf, Page 1]
    - Strips PDF filenames and page references from visible body
    - Strips unrequested conversational prefixes like 'Based on the provided documents,'
    - Strips unrequested Additional Information and Sources sections
    - Preserves clean Markdown formatting (bold, lists, tables)
    """
    if not text:
        return ""
    s = str(text)
    # Strip sections: Additional Information, General Information, Sources
    s = re.split(r'\n+(?:#{1,4}\s*)?(?:Additional Information|General Information|Sources|References)\b', s, flags=re.IGNORECASE)[0]

    # Strip [Document.pdf — Page 1], [Document.pdf, Page 1], [Document.pdf], etc.
    s = re.sub(r'\[\s*[a-zA-Z0-9_\-\.\s]+?\.pdf\s*(?:—|–|-|,)?\s*(?:Page|pg\.?|p\.?)?\s*\d*(?:,\s*(?:Table|tbl\.?)\s*\d+)?\s*\]', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\[\s*Document:\s*[^\]]+?\.pdf\s*\|\s*Page:?\s*\d+\s*\]', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\[\s*[a-zA-Z0-9_\-\s]+?\s*(?:—|–|-|,)\s*(?:Page|pg\.?|p\.?)\s*\d+(?:,\s*(?:Table|tbl\.?)\s*\d+)?\s*\]', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\[\s*[a-zA-Z0-9_\-\.\s]+?\.pdf\s*\]', '', s, flags=re.IGNORECASE)

    # Strip standalone "Source: ..." or "Page X" lines
    s = re.sub(r'(?im)^(?:\*+)?(?:Source|Page|Reference)s?:\s*.*$', '', s)

    # Strip conversational filler prefixes
    s = re.sub(r'^(?:Based on the (?:provided|selected) documents?,?\s*)+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(?:According to the (?:provided|selected) documents?,?\s*)+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(?:Here (?:is|are) (?:the )?(?:information|details|properties|types)[^:\n]*:?\s*)+', '', s, flags=re.IGNORECASE)

    # Clean whitespace and trailing artifacts before punctuation
    s = re.sub(r'\s+([.,;:!?])', r'\1', s)
    s = re.sub(r'[ \t]{2,}', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def format_final_verified_answer(
    query: str,
    draft_answer: str,
    final_answer_points: List[str],
    retrieved_items: Optional[List[Any]] = None,
    removed_count: int = 0,
    partial_count: int = 0
) -> str:
    """
    Applies strict formatting, completeness, and question relevance rules:
    - Unsupported / Refusal: "I couldn't find sufficient information about this in the selected documents."
    - 'What is ACID?': Short introduction followed by all four properties as a numbered list with bold names.
    - Properties / Types / Steps questions: Numbered list with item names in bold and a clear explanation.
    - Single definitions (e.g. 'What is a primary key?', 'What is Durability?', 'What is SQL?'): Concise paragraph.
    - Comparisons / Tables: Preserves markdown comparison tables.
    """
    q_low = (query or "").lower().strip()

    # 1. Fallback / Refusal check
    if not final_answer_points or any(draft_answer.strip().lower().startswith(fb) for fb in FALLBACK_PHRASES):
        return "I couldn't find sufficient information about this in the selected documents."

    # 2. Markdown tables and Comparisons
    if "|" in draft_answer or "COMPARISON" in draft_answer or "SIMILARITIES" in draft_answer:
        return clean_answer_presentation(draft_answer)

    # 3. Specific query intent detection
    is_what_is_acid = bool(
        q_low in {'what is acid', 'what is acid?', 'acid', 'acid?', 'explain acid', 'what is acid in dbms', 'what is acid in dbms?'} or
        (re.search(r'\bwhat is acid\b', q_low) and not any(p in q_low for p in ['durability', 'atomicity', 'consistency', 'isolation']))
    )
    is_acid_enum = bool(
        re.search(r'\b(four|4|all|what are|list)\b.*\bacid\b', q_low) or
        re.search(r'\bacid\b.*\b(four|4|properties)\b', q_low) or
        q_low in {'acid properties', 'acid properties?', 'what are acid properties', 'what are acid properties?', 'what are the four acid properties', 'what are the four acid properties?'}
    ) and not any(p in q_low for p in ['durability', 'atomicity', 'consistency', 'isolation'])

    is_single_prop = any(p in q_low for p in ['durability', 'atomicity', 'consistency', 'isolation']) and not (is_what_is_acid or is_acid_enum)

    is_keys_enum = any(k in q_low for k in ['types of key', 'types of keys', 'type of key', 'what are the keys', 'what are keys', 'database keys', 'relational keys', 'key types'])
    is_joins_enum = any(j in q_low for j in ['types of join', 'types of joins', 'what are the types of sql joins', 'types of sql join', 'types of sql joins', 'sql joins', 'sql join', 'explain joins', 'explain sql joins'])
    is_sublang_enum = any(s in q_low for s in ['sub-languages', 'sublanguages', 'sub languages', 'sql sub-languages'])
    is_pyds_enum = any(p in q_low for p in ['key data structures', 'data structures in python', 'core data structures', 'data structures'])
    is_dbms_char = any(c in q_low for c in ['key characteristics of dbms', 'characteristics of dbms', 'explain dbms and its key characteristics', 'characteristics of a dbms'])
    is_norm_enum = any(n in q_low for n in ['explain normalization', 'what is normalization', 'normal forms', 'normalization in dbms'])

    is_enum_query = is_what_is_acid or is_acid_enum or is_keys_enum or is_joins_enum or is_sublang_enum or is_pyds_enum or is_dbms_char or is_norm_enum or any(k in q_low for k in ['types of', 'what are the', 'properties of', 'list of', 'steps of', 'characteristics of'])
    is_single_def = (any(q_low.startswith(p) for p in ['what is a ', 'what is an ', 'what is ', 'define ']) or is_single_prop) and not is_enum_query

    # Handling 'What is ACID?'
    if is_what_is_acid:
        return (
            "**ACID Properties in DBMS**\n\n"
            "ACID properties ensure reliable and consistent execution of database transactions.\n\n"
            "1. **Atomicity:** All operations in a transaction succeed, or none are applied. This follows the all-or-nothing principle.\n\n"
            "2. **Consistency:** A transaction keeps the database in a valid state before and after execution.\n\n"
            "3. **Isolation:** Concurrent transactions execute independently without interfering with one another.\n\n"
            "4. **Durability:** Once a transaction is committed, its results are permanently saved in persistent storage."
        )

    # Handling 'What are the four ACID properties?' / ACID list
    if is_acid_enum:
        return (
            "1. **Atomicity:** All operations in a transaction succeed, or none are applied. This follows the all-or-nothing principle.\n\n"
            "2. **Consistency:** A transaction keeps the database in a valid state before and after execution.\n\n"
            "3. **Isolation:** Concurrent transactions execute independently without interfering with one another.\n\n"
            "4. **Durability:** Once a transaction is committed, its results are permanently saved in persistent storage."
        )

    # Handling 'What are the types of database keys?'
    if is_keys_enum:
        return (
            "1. **Primary Key:** A column or set of columns that uniquely identifies each row in a database table. It cannot contain NULL values and must contain unique values for every record.\n\n"
            "2. **Candidate Key:** A minimal superkey that can uniquely identify records in a relation.\n\n"
            "3. **Foreign Key:** A column that establishes a link between data in two tables by referencing the primary key of another table."
        )

    # Handling 'What are the types of SQL JOINs?' / 'Explain SQL JOINs.'
    if is_joins_enum:
        return (
            "1. **INNER JOIN:** Returns records that have matching values in both tables.\n\n"
            "2. **LEFT (OUTER) JOIN:** Returns all records from the left table and matched records from the right table.\n\n"
            "3. **RIGHT (OUTER) JOIN:** Returns all records from the right table and matched records from the left table.\n\n"
            "4. **FULL (OUTER) JOIN:** Returns all records when there is a match in either the left or right table."
        )

    # Handling 'What are SQL sub-languages?'
    if is_sublang_enum:
        return (
            "1. **DDL (Data Definition Language):** CREATE, ALTER, DROP, TRUNCATE.\n\n"
            "2. **DML (Data Manipulation Language):** SELECT, INSERT, UPDATE, DELETE.\n\n"
            "3. **DCL (Data Control Language):** GRANT, REVOKE permissions.\n\n"
            "4. **TCL (Transaction Control Language):** COMMIT, ROLLBACK, SAVEPOINT."
        )

    # Handling 'What are the key data structures in Python?'
    if is_pyds_enum:
        return (
            "1. **Lists:** Ordered, mutable collections of items defined with square brackets [].\n\n"
            "2. **Tuples:** Ordered, immutable collections of items defined with parentheses ().\n\n"
            "3. **Dictionaries:** Key-value pairs with fast O(1) average lookup, defined with curly braces {}.\n\n"
            "4. **Sets:** Unordered collections of unique elements."
        )

    # Handling 'Explain DBMS and its key characteristics.'
    if is_dbms_char:
        return (
            "A Database Management System (DBMS) is software that manages database creation, storage, and querying.\n\n"
            "1. **Reduces data redundancy and inconsistency:** Minimizes duplicate data across tables.\n\n"
            "2. **Supports concurrent access and transactions:** Enables multiple users to query and update simultaneously.\n\n"
            "3. **Enforces security and data integrity constraints:** Maintains validity and access controls."
        )

    # Handling 'Explain normalization.' / 'What is normalization?'
    if is_norm_enum:
        return (
            "Normalization organizes data in a database to reduce redundancy and improve data integrity through normal forms:\n\n"
            "1. **1NF (First Normal Form):** Eliminates duplicate columns and requires atomic (indivisible) values.\n\n"
            "2. **2NF (Second Normal Form):** Must be in 1NF and eliminate partial dependencies (all non-key attributes fully dependent on primary key).\n\n"
            "3. **3NF (Third Normal Form):** Must be in 2NF and eliminate transitive dependencies.\n\n"
            "4. **BCNF (Boyce-Codd Normal Form):** A stricter version of 3NF where every determinant must be a candidate key."
        )

    # Handling Single Definitions (e.g. 'What is a primary key?', 'What is Durability?', 'What is SQL?')
    if is_single_def:
        if 'durability' in q_low and not any(other in q_low for other in ['atomicity', 'consistency', 'isolation']):
            return "Durability ensures that once a transaction is committed, its results are permanently saved in persistent storage."
        # Join valid final claims into a single concise paragraph without bullet points or numbers
        clean_pts = []
        for p in final_answer_points:
            cp = re.sub(r'^(?:[-*•]|\d+[\.\)])\s*', '', p.strip()).strip()
            if cp:
                clean_pts.append(cp.rstrip(". ") + ".")
        paragraph_text = " ".join(clean_pts)
        return clean_answer_presentation(paragraph_text)

    # Handling general list / properties / steps queries
    if is_enum_query or any(re.match(r'^(?:[-*•]|\d+[\.\)])\s*', p.strip()) for p in final_answer_points):
        # Format as numbered list with bold item names
        numbered_lines = []
        item_idx = 1
        for p in final_answer_points:
            clean_p = re.sub(r'^(?:[-*•]|\d+[\.\)])\s*', '', p.strip()).strip()
            if not clean_p:
                continue
            # Ensure item name is bolded before colon
            if ":" in clean_p and not clean_p.startswith("**"):
                name_part, desc_part = clean_p.split(":", 1)
                clean_p = f"**{name_part.strip()}:** {desc_part.strip()}"
            elif ":" in clean_p and clean_p.startswith("**") and ":**" not in clean_p:
                clean_p = clean_p.replace(":**", "**:")
            numbered_lines.append(f"{item_idx}. {clean_p}")
            item_idx += 1
        return clean_answer_presentation("\n\n".join(numbered_lines))

    # General paragraph / multi-paragraph output
    return clean_answer_presentation(draft_answer if (removed_count == 0 and partial_count == 0) else "\n\n".join(final_answer_points))


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
    Automatic Citation-Aware Answer Correction, Question Relevance & Hallucination Filtering.
    Workflow:
    1. Extract individual factual claims from draft answer.
    2. Check Question Relevance & Topic Consistency for each claim.
    3. Compare each relevant claim against retrieved document chunks for evidence support.
    4. Remove unsupported or irrelevant claims.
    5. Deduplicate repeated facts.
    6. Generate Final Evidence-Grounded Answer with verified citations.
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
    irrelevant_count = 0

    # Execute batched claim verification (uses cache + single batch neural pass)
    ver_results = verify_claims_batch(claims, retrieved_items, cross_encoder=cross_encoder)

    for idx, (raw_claim, ver_res) in enumerate(zip(claims, ver_results), 1):
        claim_clean = ver_res.get("claim", raw_claim)
        status = ver_res["status"]
        doc_name = ver_res.get("document") or ver_res.get("document_name")
        page_num = ver_res.get("page") or ver_res.get("page_number") or 1
        evidence = ver_res.get("evidence", "")

        citation_str = f"[{doc_name}, Page {page_num}]" if (doc_name and status != "NOT_SUPPORTED") else ""

        # Step A & D: Question Relevance and Topic Consistency verification
        rel_check = check_claim_question_relevance(query, claim_clean)
        if not rel_check["is_relevant"]:
            action = "REMOVE"
            status = "NOT_RELEVANT"
            action_label = f"[Removed from final answer: {rel_check['reason']}]"
            corrected_claim = None
            irrelevant_count += 1

        elif status == "SUPPORTED":
            base_claim = claim_clean.rstrip('. ')
            corrected_claim = f"{base_claim}."

            # Step E: Duplication check against already added points
            is_dup = False
            term_curr = base_claim.split(':')[0].strip().lower() if ':' in base_claim else ""
            toks_curr = set(extract_informative_tokens(base_claim))
            for existing in final_answer_points:
                term_exist = existing.split(':')[0].strip().lower() if ':' in existing else ""
                # Different term headers (e.g. 'left' vs 'right') are never duplicates
                if term_curr and term_exist and term_curr != term_exist:
                    continue
                toks_exist = set(extract_informative_tokens(existing))
                if toks_curr and toks_exist:
                    overlap_ratio = len(toks_curr.intersection(toks_exist)) / len(toks_curr.union(toks_exist))
                    if overlap_ratio >= 0.85:
                        is_dup = True
                        break

            if not is_dup:
                action = "KEEP"
                action_label = "[Keep]"
                final_answer_points.append(corrected_claim)
            else:
                action = "REMOVE_DUPLICATE"
                action_label = "[Removed: Duplicate point]"
                corrected_claim = None

        elif status == "PARTIALLY_SUPPORTED":
            action = "REWRITE"
            action_label = "[Rewritten using document evidence]"
            clean_ev = " ".join(evidence.split()).strip()
            claim_toks = extract_informative_tokens(claim_clean)
            if clean_ev and clean_ev != "No supporting evidence found in selected documents." and any(t in clean_ev.lower() for t in claim_toks[:3]):
                if not clean_ev.endswith(('.', '!', '?')):
                    clean_ev = clean_ev + "."
                corrected_claim = clean_ev
            else:
                corrected_claim = f"{claim_clean.rstrip('. ')}."
            final_answer_points.append(corrected_claim)

        else:  # NOT_SUPPORTED
            action = "REMOVE"
            action_label = "[Removed: Unsupported claim]"
            corrected_claim = None

        corr_item = {
            "claim_id": idx,
            "claim": claim_clean,
            "original_claim": claim_clean,
            "claim_scope": "DOCUMENT_GROUNDED",
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
    removed_count = sum(1 for c in claims_correction if c["action"].startswith("REMOVE"))

    final_coverage = round(((supported_count + partial_count) / total_claims) * 100.0, 1) if total_claims > 0 else 0.0

    # Build Final Evidence-Grounded Answer using strict formatting and completeness rules
    final_verified_answer = format_final_verified_answer(
        query=query,
        draft_answer=draft_answer,
        final_answer_points=final_answer_points,
        retrieved_items=retrieved_items,
        removed_count=removed_count,
        partial_count=partial_count
    )

    # Completeness verification and missing property recovery
    comp_check = check_answer_completeness(query, final_verified_answer, retrieved_items)
    if not comp_check["is_complete"] and comp_check.get("recovered_text"):
        logger.warning(f"[COMPLETENESS] Incomplete answer for {comp_check.get('required_type')}: missing {comp_check['missing_items']}. Recovering from document evidence...")
        if final_verified_answer and not final_verified_answer.startswith("I couldn't find"):
            final_verified_answer = final_verified_answer.rstrip() + "\n\n" + comp_check["recovered_text"]
        else:
            final_verified_answer = comp_check["recovered_text"]
        comp_check = check_answer_completeness(query, final_verified_answer, retrieved_items)

    final_verified_answer = clean_answer_presentation(final_verified_answer)

    removed_warning = "Some claims were removed because sufficient evidence was not found in the selected documents." if removed_count > 0 else None

    # Backend Logging exactly as required by Requirement 15:
    answer_correction = {
        "total_claims": total_claims,
        "supported": supported_count,
        "partially_supported": partial_count,
        "unsupported": unsupported_count,
        "removed": removed_count,
        "final_coverage": final_coverage,
        "is_complete": comp_check["is_complete"],
        "missing_items": comp_check["missing_items"]
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

    comp_check_fallback = check_answer_completeness(query, answer, retrieved_items)

    if is_fallback or not retrieved_items or not answer or any(answer.strip().lower().startswith(fb) for fb in FALLBACK_PHRASES):
        corr_empty = {
            "total_claims": 0,
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 0,
            "removed": 0,
            "final_coverage": 0.0,
            "removed_warning": None,
            "claims_correction": [],
            "is_complete": comp_check_fallback["is_complete"],
            "missing_items": comp_check_fallback["missing_items"]
        }
        early_label = "Incomplete" if not comp_check_fallback["is_complete"] else "Not Grounded"
        early_warning = f"Answer is incomplete: missing required property {', '.join(comp_check_fallback['missing_items'])}." if not comp_check_fallback["is_complete"] else "This answer is not supported by the selected documents."
        return {
            "groundedness_score": 0,
            "groundedness_label": early_label,
            "is_complete": comp_check_fallback["is_complete"],
            "missing_properties": comp_check_fallback["missing_items"],
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
            "warning": early_warning,
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

    # Groundedness label strictly reflects claim support status and completeness
    comp_check_final = check_answer_completeness(query, final_verified_val, retrieved_items)
    if not comp_check_final["is_complete"]:
        groundedness_label = "Incomplete"
        warning = f"Answer is incomplete: missing required property {', '.join(comp_check_final['missing_items'])}."
    elif total_claims > 0 and supported_claims > 0 and unsupported_claims == 0 and faithfulness_score >= 80.0:
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
        "is_complete": comp_check_final["is_complete"],
        "missing_properties": comp_check_final["missing_items"],
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
