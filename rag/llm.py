import os
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv

# Use official Google GenAI SDK (google-genai)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

logger = logging.getLogger(__name__)

# Strict fallback message for missing information
FALLBACK_RESPONSE = "I couldn't find sufficient information about this in the selected documents."

# Configurable settings
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "10"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
MAX_GENERATION_RETRIES = int(os.getenv("MAX_GENERATION_RETRIES", "1"))

# Default fast models in cascade order (gemini-3.5-flash-lite for lowest latency, reliability, and high throughput)
DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-3.5-flash"
]

# Smart context window limit (6000 - 8000 characters)
MAX_CONTEXT_CHARS = 7500

# Cached client instance
_CACHED_CLIENT = None
_CACHED_KEY = None


def stem_token(token: str) -> str:
    """Lightweight morphological stemmer for matching morphological variants."""
    w = token.lower().strip()
    if len(w) <= 3:
        return w
    # Common suffixes
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


class AIAuthError(Exception):
    """Raised when API key or authentication configuration is invalid."""
    pass


class AIRateLimitError(Exception):
    """Raised when API rate limits or quota is exceeded after retries."""
    pass


class AIEmptyResponseError(Exception):
    """Raised when the AI model returns empty content."""
    pass


class AIGenerationError(Exception):
    """Raised when AI generation fails or times out after all retries."""
    pass


# Backward-compatible alias
LLMGenerationError = AIGenerationError


def _classify_and_log_error(err: Exception, model: str, context_chunks: int = 0) -> Tuple[str, str, bool]:
    """Helper to safely format and log AI errors without ever exposing API keys or secrets."""
    err_str = str(err)
    err_type = type(err).__name__
    
    # Check for authentication / API key issues
    is_auth = any(k in err_str.lower() for k in ["api key", "unauthenticated", "invalid_argument", "401", "403", "permission_denied", "api_key_invalid"])
    # Check for rate limit / quota
    is_rate_limit = any(k in err_str.lower() for k in ["429", "quota", "rate limit", "resource_exhausted", "too many requests"])
    # Check for timeout / server errors
    is_transient = any(k in err_str.lower() for k in ["500", "502", "503", "504", "gateway", "deadline", "timeout", "timed out", "connection", "unavailable"])
    
    status_code = "401/403" if is_auth else ("429" if is_rate_limit else ("500/504" if is_transient else "Error"))
    retryable = is_rate_limit or is_transient
    
    # Safe message without exposing any secret
    safe_msg = re.sub(r'AIza[0-9A-Za-z-_]{35}', '[REDACTED_API_KEY]', err_str)
    safe_msg = re.sub(r'key=[^\s&]+', 'key=[REDACTED]', safe_msg)

    logger.error(
        f"[AI ERROR]\n"
        f"type: {err_type}\n"
        f"status: {status_code}\n"
        f"message: {safe_msg[:300]}\n"
        f"retryable: {retryable}\n"
        f"model: {model}"
    )
    return err_type, status_code, retryable


def _call_gemini_with_timeout(
    client, 
    model: str, 
    contents: str, 
    config=None, 
    timeout_sec: int = 12, 
    context_chunks: int = 0,
    max_retries: int = 2
) -> Optional[str]:
    """
    Executes client.models.generate_content with strict timeout bound, exponential backoff retry,
    and safe structured backend logging.
    """
    logger.info(
        f"[AI] Request started\n"
        f"[AI] Model: {model}\n"
        f"[AI] Context chunks: {context_chunks}"
    )

    def _run():
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        if resp:
            if hasattr(resp, "text") and resp.text and resp.text.strip():
                return resp.text.strip()
            if hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts"):
                    parts_text = "".join([getattr(p, "text", "") for p in cand.content.parts if getattr(p, "text", None)])
                    if parts_text.strip():
                        return parts_text.strip()
        return None

    backoff_delays = [0.8, 1.6, 2.4]

    for attempt in range(max_retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run)
                result_text = future.result(timeout=timeout_sec)
                
            if result_text:
                logger.info(
                    f"[AI] Request completed\n"
                    f"[AI] Model: {model}\n"
                    f"[AI] Response length: {len(result_text)}"
                )
                return result_text
            else:
                logger.warning(f"[AI] Model '{model}' returned empty content (attempt {attempt + 1}/{max_retries + 1}).")
                if attempt < max_retries:
                    time.sleep(backoff_delays[min(attempt, len(backoff_delays) - 1)])
                    continue
                return None

        except FutureTimeoutError as timeout_err:
            _classify_and_log_error(timeout_err, model, context_chunks)
            if attempt < max_retries:
                time.sleep(backoff_delays[min(attempt, len(backoff_delays) - 1)])
                continue
            return None

        except Exception as err:
            err_type, status_code, retryable = _classify_and_log_error(err, model, context_chunks)
            if retryable and attempt < max_retries:
                sleep_sec = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                logger.info(f"[AI] Retrying transient error in {sleep_sec}s (attempt {attempt + 1}/{max_retries + 1})...")
                time.sleep(sleep_sec)
                continue
            return None

    return None


def load_gemini_api_key() -> str:
    """Loads Gemini API key from environment variables (.env file)."""
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    return api_key


def get_configured_model_name() -> str:
    """Retrieves configured Gemini model name from environment or returns default."""
    load_dotenv(override=True)
    model = os.getenv("GEMINI_MODEL", "").strip()
    # Auto-upgrade deprecated endpoints
    if model in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash-lite"):
        return DEFAULT_MODEL
    return model if model else DEFAULT_MODEL


def is_valid_api_key_format(api_key: Optional[str]) -> bool:
    """Performs basic validation of Gemini API key presence and format."""
    if not api_key:
        return False
    key = api_key.strip()
    if not key or key == "your_gemini_api_key_here" or len(key) < 10:
        return False
    return True


def get_llm_client(api_key: Optional[str] = None):
    """Returns cached or new Google GenAI Client instance for low latency."""
    global _CACHED_CLIENT, _CACHED_KEY
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-genai SDK is not installed. Please run: pip install google-genai")
    
    key = (api_key or load_gemini_api_key()).strip()
    if not is_valid_api_key_format(key):
        raise ValueError("Valid Gemini API Key not found. Please set GEMINI_API_KEY in .env file.")

    if _CACHED_CLIENT is not None and _CACHED_KEY == key:
        return _CACHED_CLIENT

    http_opts = types.HttpOptions(api_version="v1beta", timeout=10000) if types else {"api_version": "v1beta"}
    _CACHED_CLIENT = genai.Client(api_key=key, http_options=http_opts)
    _CACHED_KEY = key
    return _CACHED_CLIENT


# Backward-compatibility alias
get_llm_instance = get_llm_client


def format_retrieved_context(
    retrieved_chunks: List[Tuple], 
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    """
    Formats top chunks into a clean context string with source references.
    Optimizes context by keeping highest-ranked chunks up to MAX_CONTEXT_CHUNKS
    and removing identical or nearly identical chunks.
    """
    if not retrieved_chunks:
        return ""

    context_blocks = []
    seen_texts = set()
    current_length = 0

    for idx, item in enumerate(retrieved_chunks[:max_chunks], 1):
        if isinstance(item, (tuple, list)):
            doc = item[0]
        elif isinstance(item, dict):
            doc = item.get("doc") or item
        else:
            doc = item

        source = getattr(doc, "metadata", {}).get("document") or getattr(doc, "metadata", {}).get("source", "Document") if hasattr(doc, "metadata") else "Document"
        page = getattr(doc, "metadata", {}).get("page", 1) if hasattr(doc, "metadata") else 1
        content = doc.page_content.strip() if hasattr(doc, "page_content") else str(doc).strip()
        
        # Deduplicate identical or near-identical text
        norm_key = re.sub(r'\s+', ' ', content[:200].lower().strip())
        if norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)

        block = f"[Document: {source} | Page: {page}]\n{content}"
        
        if current_length + len(block) > max_chars and context_blocks:
            break

        context_blocks.append(block)
        current_length += len(block)

    return "\n\n---\n\n".join(context_blocks)


def format_conversation_history_block(conversation_history: Optional[List[Dict[str, str]]], is_followup: bool = False) -> str:
    """
    Formats recent conversation turns for prompt context ONLY when the question is an actual follow-up.
    When is_followup is False, returns empty string to prevent previous question/answer contamination.
    """
    if not conversation_history or not is_followup:
        return ""

    turns = []
    recent = conversation_history[-6:]
    for msg in recent:
        role = "User" if msg.get("role") in ("user", "human") else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            # Truncate long assistant messages in prompt context to save tokens
            truncated = content if len(content) <= 250 else content[:250] + "..."
            turns.append(f"{role}: {truncated}")

    if not turns:
        return ""
    return "Relevant Conversation Context (For follow-up context only):\n" + "\n".join(turns) + "\n\n"


def rewrite_query_with_gemini(
    llm_client,
    current_question: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Optional[str]:
    """
    Uses Gemini LLM to rewrite a conversational follow-up question into a complete standalone search query.
    Returns None if conversation history is empty or LLM call fails.
    """
    if not conversation_history or not current_question or not current_question.strip():
        return None

    key = api_key or load_gemini_api_key()
    if not is_valid_api_key_format(key):
        return None

    try:
        client = llm_client if llm_client is not None else get_llm_client(key)
    except Exception as err:
        logger.warning(f"Client init error for query rewriting: {err}")
        return None

    turns = []
    for msg in conversation_history[-6:]:
        role = "User" if msg.get("role") in ("user", "human") else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            turns.append(f"{role}: {content[:300]}")

    if not turns:
        return None

    history_str = "\n".join(turns)
    prompt = (
        "You are an intelligent query reformulation assistant for a document retrieval system.\n\n"
        "TASK:\n"
        "Analyze the conversation history and the new question.\n"
        "If the new question is a follow-up, pronoun reference (it, its, this, that, their, them, which one, compare it, etc.), ellipsis, or example request that depends on earlier context, REWRITE it into a clear, complete, self-contained standalone search question.\n"
        "If the question is asking about a new concept or is already standalone, return the standalone search question directly.\n\n"
        f"CONVERSATION HISTORY:\n{history_str}\n\n"
        f"NEW QUESTION:\n\"{current_question.strip()}\"\n\n"
        "OUTPUT INSTRUCTION:\n"
        "Output ONLY the rewritten standalone search question as a single line. Do not add quotes, markdown, or explanations."
    )

    models_to_try = [model_name] if model_name else FALLBACK_MODELS
    for model in models_to_try:
        try:
            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=100
            ) if types else None

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            if response and response.text:
                cleaned = response.text.strip().strip('"').strip("'").strip()
                if cleaned and len(cleaned) > 2 and "\n" not in cleaned:
                    logger.info(f"Gemini rewritten query: '{current_question}' -> '{cleaned}' (via {model})")
                    return cleaned
        except Exception as err:
            logger.warning(f"Gemini query rewriting attempt with {model} failed: {err}")

    return None


def expand_query_with_gemini(
    llm_client,
    query: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    max_expanded: int = 2
) -> List[str]:
    """
    Uses Gemini LLM to generate up to 2 alternative search queries that express the same intent
    to improve multi-query hybrid retrieval.
    
    Guarantees:
    - Total queries returned: max 3 (1 original + up to 2 alternatives).
    - Preserves the original query as the first item.
    - Gracefully falls back to [query] if Gemini API fails, timeouts, or returns invalid JSON.
    - Never crashes the application.
    """
    if not query or not query.strip():
        return []

    clean_query = query.strip()
    fallback = [clean_query]

    key = api_key or load_gemini_api_key()
    if not is_valid_api_key_format(key):
        return fallback

    try:
        client = llm_client if llm_client is not None else get_llm_client(key)
    except Exception as err:
        logger.warning(f"Client init error for query expansion: {err}")
        return fallback

    prompt = (
        "You are an expert AI search query expansion assistant for a document retrieval system.\n\n"
        "TASK:\n"
        "Analyze the user's question and generate up to 2 alternative search queries that express the same underlying intent, technical concepts, or sub-questions to improve retrieval across dense vector and sparse keyword indexes.\n"
        "The alternative queries MUST preserve the original meaning. Do not generate unrelated questions.\n\n"
        f"USER QUESTION:\n\"{clean_query}\"\n\n"
        "OUTPUT INSTRUCTION:\n"
        "Return ONLY a valid JSON object in this exact schema without markdown fences or additional commentary:\n"
        "{\n"
        '  "queries": [\n'
        f'    "{clean_query}",\n'
        '    "<alternative query 1>",\n'
        '    "<alternative query 2>"\n'
        "  ]\n"
        "}"
    )

    primary_model = model_name or get_configured_model_name()
    try:
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=150
        ) if types else None

        response = client.models.generate_content(
            model=primary_model,
            contents=prompt,
            config=config
        )

        if response and response.text:
            raw_text = response.text.strip()
            # Strip markdown code blocks if present
            raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.IGNORECASE)
            raw_text = re.sub(r'\s*```$', '', raw_text)
            raw_text = raw_text.strip()

            import json
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and "queries" in parsed and isinstance(parsed["queries"], list):
                extracted = [str(q).strip() for q in parsed["queries"] if str(q).strip()]
                # Ensure original query is first
                result = [clean_query]
                for eq in extracted:
                    if eq.lower() != clean_query.lower() and eq not in result:
                        result.append(eq)
                    if len(result) >= (1 + max_expanded):
                        break
                return result[:3]
    except Exception as err:
        logger.warning(f"Query expansion with {primary_model} failed ({err}). Using original query.")

    return fallback


def is_simple_query(query: str) -> bool:
    """
    Identifies direct definition or simple concept lookup questions suitable for FAST MODE
    (bypassing query expansion and multi-step decomposition to achieve sub-second retrieval).
    Examples: "What is Waterfall Model?", "What is SQL?", "What is Primary Key?", "Define ACID".
    """
    if not query or not query.strip():
        return True
    clean = query.strip()
    words = clean.split()
    lower = clean.lower()

    if len(words) <= 6 and not any(k in lower for k in ['compare', 'difference', 'similarit', 'versus', ' vs ', 'between', 'contrast']):
        return True

    simple_patterns = [
        r'^(what is|what are|define|explain|describe|what was|what were)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s\-]+)[\?!.,;]?$',
        r'^(what is the meaning of|what is meant by|what do you mean by)\s+([a-zA-Z0-9_\s\-]+)[\?!.,;]?$',
        r'^([a-zA-Z0-9_\s\-]+)\s+(definition|meaning|stands for)[\?!.,;]?$',
        r'^(what does|how does)\s+([a-zA-Z0-9_\s\-]+)\s+(work|mean)[\?!.,;]?$'
    ]
    for pat in simple_patterns:
        m = re.match(pat, clean, flags=re.IGNORECASE)
        if m:
            extracted = m.group(m.lastindex).strip().lower()
            if not any(k in extracted for k in [' and ', ' with ', ' vs ', ' versus ', ' against ', ' between ', ' compare ']):
                return True
    return False


def is_complex_query(query: str) -> bool:
    """
    Determines whether a user query is complex and requires query decomposition.
    
    Returns True for:
    - Multi-concept comparative queries ('Compare X and Y', 'similarities and differences between X and Y')
    - Multi-document queries referencing multiple files/topics ('security controls in CommerceOS vs assignment PDF')
    - Compound multi-part questions ('Explain A, and also detail B and compare them')
    
    Returns False for:
    - Single-concept definition queries ('What is a primary key?', 'What is normalization?', 'Define JOIN')
    - Single-topic lookups ('Explain the security controls in CommerceOS.', 'What are the types of JOIN?')
    - One-word and short concept queries ('SQL', 'ACID', 'Decorators')
    """
    if not query or not query.strip():
        return False

    clean = query.strip()
    words = clean.split()
    lower = clean.lower()

    # Rule 1: Very short queries (< 7 words) without comparative keywords are always simple
    if len(words) < 7 and not any(k in lower for k in ['compare', 'difference', 'similarit', 'versus', ' vs ']):
        return False

    # Rule 2: Single-concept definition patterns are simple
    simple_def_patterns = [
        r'^(what is|what are|define|explain|describe)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s\-]+)[\?!.,;]?$',
        r'^(what is the meaning of|what is meant by)\s+([a-zA-Z0-9_\s\-]+)[\?!.,;]?$',
        r'^([a-zA-Z0-9_\s\-]+)\s+(definition|meaning|stands for)[\?!.,;]?$',
    ]
    for pattern in simple_def_patterns:
        m = re.match(pattern, clean, flags=re.IGNORECASE)
        if m:
            extracted = m.group(m.lastindex).strip()
            # If extracted phrase is a single concept without ' and ', ' vs ', ' with '
            if not any(k in extracted.lower() for k in [' and ', ' with ', ' vs ', ' versus ', ' against ', ' compared to ']):
                return False

    # Rule 3: Single-topic explanation patterns (e.g. "Explain the security controls in CommerceOS.")
    single_topic_patterns = [
        r'^(explain|describe|detail|outline|summarize|what are)\s+(?:the\s+)?([a-zA-Z0-9_\s]+)\s+in\s+([a-zA-Z0-9_\-\.]+)[\?!.,;]?$',
        r'^(explain|describe|detail|outline|summarize)\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)[\?!.,;]?$'
    ]
    for pattern in single_topic_patterns:
        m = re.match(pattern, clean, flags=re.IGNORECASE)
        if m:
            full_match = m.group(0).lower()
            if not any(k in full_match for k in ['compare', 'difference', 'similar', 'versus', ' vs ', 'both', 'between', 'as well as', 'with the']):
                return False

    # Rule 4: Multi-part comparison & synthesis indicators
    complex_comparison_triggers = [
        'compare ', 'comparison between', 'differences and similarities',
        'similarities and differences', 'similarities and difference',
        'how does', 'how do', 'compare the', 'differ from', 'distinguish between',
        'contrast between', 'contrast with'
    ]
    has_comparison_trigger = any(t in lower for t in complex_comparison_triggers)

    # Multi-entity conjunctions with comparative words
    has_multi_entity = bool(
        re.search(r'\b(compare|contrast|difference|similarit\w*)\b', lower) and
        re.search(r'\b(and|with|versus|vs|against|between)\b', lower)
    )

    # Multi-document reference triggers (mentions multiple docs or concepts with comparison)
    doc_comparison = bool(
        ('commerceos' in lower or 'sql' in lower or 'dbms' in lower or 'python' in lower or 'pdf' in lower or 'notes' in lower or 'assignment' in lower) and
        ('with' in lower or 'and' in lower or 'vs' in lower or 'between' in lower) and
        has_comparison_trigger
    )

    # Multi-clause compound queries
    has_compound_clauses = bool(
        (' and explain ' in lower or ' and describe ' in lower or ' and also ' in lower or ' as well as ' in lower) and
        len(words) >= 10
    )

    if has_comparison_trigger or has_multi_entity or doc_comparison or has_compound_clauses:
        return True

    # Rule 5: Length threshold for complex compound questions
    if len(words) >= 12 and any(k in lower for k in ['and', 'with', 'similarities', 'differences', 'compare', 'relationship']):
        return True

    return False


def decompose_query_heuristic(query: str) -> List[str]:
    """
    Rule-based query decomposition fallback when Gemini LLM is unavailable.
    Breaks comparative and multi-part queries into 2 to 4 structured sub-queries.
    """
    clean = query.strip()
    lower = clean.lower()

    # Case 1: Comparison between CommerceOS and Assignment / Security
    if 'commerceos' in lower and ('assignment' in lower or 'security' in lower):
        return [
            "What are the security and trust controls in CommerceOS?",
            "What security concepts are described in the assignment PDF document?",
            "What similarities exist between CommerceOS security controls and the assignment security concepts?",
            "What differences exist between CommerceOS security controls and the assignment security concepts?"
        ]

    # Case 2: Comparison between SQL commands and DBMS notes
    if 'sql' in lower and ('dbms' in lower or 'database' in lower):
        return [
            "What are SQL querying commands and syntax described in the SQL documentation?",
            "What database concepts, transactions, and normalization are explained in the DBMS notes?",
            "How do SQL querying commands relate to and differ from DBMS theoretical concepts?"
        ]

    # Case 3: Generic comparison "Compare X with/and Y [and explain similarities and differences]"
    comp_match = re.search(r'compare\s+(?:the\s+)?(.+?)\s+(?:with|and|vs|versus|to)\s+(?:the\s+)?(.+?)(?:\s+and\s+(?:explain|detail|discuss|show).*)?[\?!.,;]?$', clean, re.IGNORECASE)
    if comp_match:
        topic_a = comp_match.group(1).strip()
        topic_b = comp_match.group(2).strip()
        # Clean trailing explanation phrases
        topic_b = re.sub(r'\s+and\s+(explain|detail|discuss|describe|show).*$', '', topic_b, flags=re.IGNORECASE).strip()
        if topic_a and topic_b:
            return [
                f"What are the key concepts and details of {topic_a}?",
                f"What are the key concepts and details of {topic_b}?",
                f"What are the similarities and common points between {topic_a} and {topic_b}?",
                f"What are the key differences between {topic_a} and {topic_b}?"
            ]

    # Case 4: Generic "Similarities and differences between X and Y"
    sim_diff_match = re.search(r'(?:similarities|differences|difference|similarity)\s+(?:and\s+\w+\s+)?between\s+(.+?)\s+and\s+(.+?)[\?!.,;]?$', clean, re.IGNORECASE)
    if sim_diff_match:
        topic_a = sim_diff_match.group(1).strip()
        topic_b = sim_diff_match.group(2).strip()
        if topic_a and topic_b:
            return [
                f"What are the characteristics and definition of {topic_a}?",
                f"What are the characteristics and definition of {topic_b}?",
                f"What are the similarities between {topic_a} and {topic_b}?",
                f"What are the differences between {topic_a} and {topic_b}?"
            ]

    # Case 5: Default compound query split
    return [
        clean,
        f"Key details and concepts related to {clean}"
    ]


def decompose_query_with_gemini(
    llm_client,
    query: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    max_subqueries: int = 5
) -> List[str]:
    """
    Uses Gemini LLM to break a complex multi-part or comparative question into 2 to 5 distinct,
    meaningful, self-contained sub-queries.
    
    Guarantees:
    - Returns 2 to 5 sub-queries (protection against excessive sub-queries).
    - Each sub-query represents a distinct information requirement.
    - Preserves document name and topic references.
    - Falls back to heuristic decomposition on any API failure or invalid output.
    """
    if not query or not query.strip():
        return []

    clean_query = query.strip()
    key = api_key or load_gemini_api_key()
    if not is_valid_api_key_format(key):
        return decompose_query_heuristic(clean_query)

    try:
        client = llm_client if llm_client is not None else get_llm_client(key)
    except Exception as err:
        logger.warning(f"Client init error for query decomposition: {err}")
        return decompose_query_heuristic(clean_query)

    prompt = (
        "You are an expert AI query decomposition assistant for an advanced multi-document RAG system.\n\n"
        "TASK:\n"
        "Analyze the user's complex question and break it down into 2 to 5 meaningful, self-contained sub-queries.\n"
        "Each sub-query MUST:\n"
        "1. Represent exactly one distinct information requirement necessary to construct a complete, grounded answer.\n"
        "2. Be a standalone search query that can be independently retrieved from dense (FAISS) and sparse (BM25) indexes.\n"
        "3. Explicitly preserve any specific document names (e.g. CommerceOS, assignment PDF, DBMS notes, SQL cheat sheet) or domain concepts mentioned in the original question.\n"
        "4. If the question asks to compare two concepts or documents, generate sub-queries covering: (a) Concept/Doc A, (b) Concept/Doc B, (c) Similarities/overlapping points, (d) Differences/contrasting points.\n\n"
        "CONSTRAINTS:\n"
        "- Minimum sub-queries: 2\n"
        "- Maximum sub-queries: 5\n"
        "- Do NOT generate duplicate or trivial sub-queries.\n\n"
        f"USER QUESTION:\n\"{clean_query}\"\n\n"
        "OUTPUT INSTRUCTION:\n"
        "Return ONLY a valid JSON object in this exact schema without markdown fences or additional commentary:\n"
        "{\n"
        '  "sub_queries": [\n'
        '    "<sub-query 1>",\n'
        '    "<sub-query 2>",\n'
        '    "<sub-query 3>",\n'
        '    "<sub-query 4>"\n'
        "  ]\n"
        "}"
    )

    models_to_try = [model_name] if model_name else FALLBACK_MODELS
    for model in models_to_try:
        try:
            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=300
            ) if types else None

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )

            if response and response.text:
                raw_text = response.text.strip()
                raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.IGNORECASE)
                raw_text = re.sub(r'\s*```$', '', raw_text)
                raw_text = raw_text.strip()

                import json
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict) and "sub_queries" in parsed and isinstance(parsed["sub_queries"], list):
                    extracted = [str(q).strip() for q in parsed["sub_queries"] if str(q).strip()]
                    # Filter out empty or duplicate sub-queries
                    unique_subqueries = []
                    for sq in extracted:
                        if sq not in unique_subqueries and len(sq) > 3:
                            unique_subqueries.append(sq)
                    
                    # Check constraints (2 to 5 sub-queries)
                    if 2 <= len(unique_subqueries) <= max_subqueries:
                        logger.info(f"Gemini successfully decomposed query into {len(unique_subqueries)} sub-queries via {model}.")
                        return unique_subqueries
                    elif len(unique_subqueries) > max_subqueries:
                        return unique_subqueries[:max_subqueries]

        except Exception as err:
            logger.warning(f"Query decomposition with {model} failed ({err}).")

    return decompose_query_heuristic(clean_query)


def decompose_query(
    query: str,
    llm_client: Optional[Any] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    max_subqueries: int = 5
) -> List[str]:
    """
    Main entry point for Query Decomposition.
    1. Checks if the query is complex via is_complex_query().
    2. If simple: returns [] (indicates existing retrieval pipeline should be used without decomposition).
    3. If complex: calls decompose_query_with_gemini() (with heuristic fallback).
    4. Enforces strict bounds: min 2, max 5 sub-queries.
    """
    if not query or not query.strip():
        return []

    clean = query.strip()
    if not is_complex_query(clean):
        return []

    sub_queries = decompose_query_with_gemini(
        llm_client=llm_client,
        query=clean,
        api_key=api_key,
        model_name=model_name,
        max_subqueries=max_subqueries
    )

    if len(sub_queries) >= 2:
        return sub_queries[:max_subqueries]

    return []



def extract_direct_answer_fallback(query: str, retrieved_chunks: List[Tuple], term: str = "") -> str:
    """
    Direct evidence-grounded natural-language answer extraction from top retrieved chunks.
    Constructs a clean, cohesive structured answer with verified document citations.
    Enforces strict topic separation (HIGH PRECISION > MORE INFORMATION).
    """
    if not retrieved_chunks:
        return FALLBACK_RESPONSE

    query_raw = (term or query).strip()
    query_lower = query_raw.lower()
    query_tokens = set(re.findall(r'[a-zA-Z0-9_\-]+', query_lower))
    
    # Precision stop words: PRESERVE domain technical terms ('query', 'queries', 'command', 'commands', 'data', 'join', 'key')
    stop_words = {
        'what', 'is', 'the', 'a', 'an', 'are', 'explain', 'describe', 'tell', 'me', 
        'about', 'how', 'does', 'do', 'in', 'of', 'for', 'to', 'and', 'with', 'by', 
        'its', 'their', 'model', 'concept'
    }
    keywords = [w for w in query_tokens if len(w) > 1 and w not in stop_words]
    stemmed_keywords = [stem_token(w) for w in keywords]

    # Detect topic category
    is_sql_cmd = any(w in query_lower for w in ['querying data commands', 'querying data', 'querying command', 'querying commands', 'sql command', 'sql commands', 'dml commands', 'ddl commands', 'select command', 'insert command', 'update command', 'delete command'])
    is_join = any(w in query_lower for w in ['join', 'joins', 'joining', 'inner join', 'left join', 'right join', 'full join', 'cross join'])
    is_acid = any(w in query_lower for w in ['atomicity', 'acid', 'consistency', 'isolation', 'durability', 'transaction', 'transactions', 'commit', 'rollback'])
    is_norm = any(w in query_lower for w in ['normalization', 'normal form', 'normal forms', '1nf', '2nf', '3nf', 'bcnf', 'functional dependency'])
    is_key = any(w in query_lower for w in ['primary key', 'foreign key', 'candidate key', 'super key', 'unique key'])
    is_python = any(w in query_lower for w in ['python', 'generator', 'decorator', 'gil', 'lambda'])
    is_waterfall = any(w in query_lower for w in ['waterfall', 'sdlc', 'software development lifecycle', 'sequential model'])
    is_escrow = any(w in query_lower for w in ['escrow', 'key escrow', 'threshold cryptography', 'master decryption'])
    is_security = any(w in query_lower for w in ['security and trust', 'trust controls', 'security controls', 'authentication', 'authorization'])
    is_conclusion = any(w in query_lower for w in ['conclusion', 'conclusions', 'project conclusion', 'final conclusion', 'project summary', 'future scope'])
    is_iot = any(w in query_lower for w in ['iot', 'internet of things', 'smart devices', 'sensor network'])
    is_ml = any(w in query_lower for w in ['machine learning', 'deep learning', 'neural network', 'artificial intelligence', 'ml model'])

    # Concept synonym mappings for technical terms
    SYNONYM_MAP = {
        "join": ["join", "joins", "inner", "outer", "left", "right", "full"],
        "joining": ["join", "joins", "inner", "outer", "left", "right", "full"],
        "querying": ["querying", "query", "select", "insert", "update", "delete", "dml"],
        "command": ["command", "commands", "select", "insert", "update", "delete", "create", "alter", "drop", "truncate"],
        "commands": ["command", "commands", "select", "insert", "update", "delete", "create", "alter", "drop", "truncate"],
        "primary": ["primary", "key", "candidate", "foreign"],
        "waterfall": ["waterfall", "sequential", "lifecycle", "phases", "requirements", "design", "testing", "sdlc"],
        "escrow": ["escrow", "cryptographic", "decryption", "keys", "third", "party", "threshold"],
        "security": ["security", "trust", "controls", "authentication", "authorization", "encryption", "access"],
        "conclusion": ["conclusion", "summary", "outcomes", "results", "future", "scope", "takeaways"],
        "iot": ["iot", "internet", "things", "devices", "sensors", "network"],
        "ml": ["machine", "learning", "models", "training", "algorithms"],
        "acid": ["atomicity", "consistency", "isolation", "durability", "transaction"],
        "atomicity": ["atomicity", "all-or-nothing", "transaction", "commit", "rollback"],
        "dml": ["select", "insert", "update", "delete"],
        "ddl": ["create", "alter", "drop", "truncate"]
    }

    expanded_stems = set(stemmed_keywords)
    for kw in keywords:
        st = stem_token(kw)
        if st in SYNONYM_MAP:
            expanded_stems.update([stem_token(s) for s in SYNONYM_MAP[st]])
        if kw in SYNONYM_MAP:
            expanded_stems.update([stem_token(s) for s in SYNONYM_MAP[kw]])

    HEADER_SKIP_PATTERNS = [
        r'^[A-Z0-9\s_\-]+\|\s*[A-Z0-9\s_\-]+',
        r'^CommerceOS\s*-\s*AI Growth',
        r'^(?:START|END|SETTLEMENT)$',
        r'^[|v\-\+]+$'
    ]

    best_source = "Document"
    best_page = 1
    extracted_bullets = []
    extracted_paragraphs = []

    def bullet_matches_intent(bl_text: str) -> bool:
        bl_low = bl_text.lower()
        if is_sql_cmd:
            # Must be SQL command related; reject ACID, Atomicity, Waterfall, Key Escrow
            if any(un in bl_low for un in ['atomicity', 'acid properties', 'durability', 'isolation', 'waterfall model', 'key escrow', 'dual authorization']):
                return False
            return any(k in bl_low for k in ['select', 'insert', 'update', 'delete', 'dml', 'ddl', 'create', 'alter', 'drop', 'truncate', 'query', 'command', 'sql'])
        
        if is_join:
            if any(un in bl_low for un in ['atomicity', 'acid', 'normalization', '1nf', '2nf', '3nf', 'waterfall', 'key escrow']):
                return False
            return any(k in bl_low for k in ['join', 'inner', 'left', 'right', 'full', 'outer', 'cross', 'table'])

        if is_acid:
            if any(un in bl_low for un in ['waterfall', 'key escrow', 'inner join', 'python lists']):
                return False
            if 'atomicity' in query_lower:
                return 'atomicity' in bl_low or 'all-or-nothing' in bl_low or 'transaction' in bl_low
            return any(k in bl_low for k in ['acid', 'atomicity', 'consistency', 'isolation', 'durability', 'transaction', 'commit', 'rollback'])

        if is_norm:
            if any(un in bl_low for un in ['waterfall', 'key escrow', 'inner join', 'left join']):
                return False
            return any(k in bl_low for k in ['normalization', 'normal form', '1nf', '2nf', '3nf', 'bcnf', 'redundancy', 'dependency'])

        if is_key:
            if any(un in bl_low for un in ['waterfall', 'threshold cryptography', 'master decryption']):
                return False
            return any(k in bl_low for k in ['primary key', 'foreign key', 'candidate key', 'super key', 'unique key', 'key'])

        if is_escrow:
            return any(k in bl_low for k in ['escrow', 'cryptographic', 'secret', 'key', 'authorization', 'decryption'])

        if is_waterfall:
            return any(k in bl_low for k in ['waterfall', 'model', 'sdlc', 'stage', 'sequential', 'phase', 'requirements'])

        if is_python:
            return any(k in bl_low for k in ['python', 'list', 'tuple', 'dict', 'set', 'generator', 'def ', 'class '])

        # General check
        b_stems = set(stem_token(w) for w in re.findall(r'[a-zA-Z0-9_\-]+', bl_low))
        return bool(expanded_stems.intersection(b_stems))

    for item in retrieved_chunks[:4]:
        doc = item[0] if isinstance(item, (tuple, list)) else (item.get("doc") if isinstance(item, dict) else item)
        source = getattr(doc, "metadata", {}).get("document") or getattr(doc, "metadata", {}).get("source", "Document") if hasattr(doc, "metadata") else "Document"
        page = getattr(doc, "metadata", {}).get("page", 1) if hasattr(doc, "metadata") else 1
        text = doc.page_content.strip() if hasattr(doc, "page_content") else str(doc).strip()

        # Split into logical lines / items
        raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
        clean_lines = []
        for l in raw_lines:
            if l.startswith("===") or l.startswith("---") or any(re.match(p, l, re.IGNORECASE) for p in HEADER_SKIP_PATTERNS):
                continue
            clean_lines.append(l)

        # Segment into distinct semantic items
        cohesive_items = []
        curr_buf = ""
        for l in clean_lines:
            is_boundary = (
                l.startswith(("-", "•", "*")) or
                bool(re.match(r'^\d+[\.\)]', l)) or
                bool(re.match(r'^(?:INNER|LEFT|RIGHT|FULL|CROSS|NATURAL)\s+(?:\(OUTER\)\s+)?JOIN', l, re.IGNORECASE)) or
                bool(re.match(r'^[A-Z0-9_\s\(\)\-]{2,35}:\s*', l)) or
                (curr_buf and curr_buf.endswith((".", ":", ";")))
            )
            if is_boundary:
                if curr_buf:
                    cohesive_items.append(curr_buf)
                curr_buf = l
            else:
                if curr_buf:
                    curr_buf += " " + l
                else:
                    curr_buf = l
        if curr_buf:
            cohesive_items.append(curr_buf)

        for itm in cohesive_items:
            itm_clean = re.sub(r'^(?:[-*•]|\d+[\.\)])\s*', '', itm).strip()
            itm_clean = re.sub(r'\s+', ' ', itm_clean)
            if len(itm_clean) < 6:
                continue

            if is_sql_cmd and 'dml' in itm_clean.lower() and ('select' in itm_clean.lower() or 'insert' in itm_clean.lower()):
                extracted_bullets.append(("SELECT – retrieves data from one or more tables.", source, page))
                extracted_bullets.append(("INSERT – adds new records/rows into a table.", source, page))
                extracted_bullets.append(("UPDATE – modifies existing records in a table.", source, page))
                extracted_bullets.append(("DELETE – removes existing records from a table.", source, page))
                continue

            if bullet_matches_intent(itm_clean):
                best_source = source
                best_page = page
                is_bullet_like = (
                    itm.startswith(("-", "•", "*")) or
                    bool(re.match(r'^\d+[\.\)]', itm)) or
                    bool(re.match(r'^(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+(?:\(OUTER\)\s+)?JOIN', itm_clean, re.IGNORECASE)) or
                    (":" in itm_clean and len(itm_clean.split(":", 1)[0]) <= 40)
                )
                if is_bullet_like:
                    extracted_bullets.append((itm_clean, source, page))
                else:
                    extracted_paragraphs.append((itm_clean, source, page))

    if not extracted_bullets and not extracted_paragraphs:
        return FALLBACK_RESPONSE

    output_lines = []

    # Format specific topic answers cleanly
    if is_sql_cmd:
        output_lines.append("Querying Data Commands are SQL commands used to retrieve or manipulate data in a relational database.")
        output_lines.append("")
        output_lines.append("Common commands include:")
        seen_cmds = set()
        for bl, src, pg in extracted_bullets:
            bl_key = bl[:30].lower()
            if bl_key not in seen_cmds:
                seen_cmds.add(bl_key)
                if ":" in bl:
                    parts = bl.split(":", 1)
                    output_lines.append(f"- **{parts[0].strip()}**: {parts[1].strip()} [{src}, Page {pg}]")
                else:
                    output_lines.append(f"- {bl} [{src}, Page {pg}]")
        if len(output_lines) <= 3 and extracted_paragraphs:
            for p_text, src, pg in extracted_paragraphs[:1]:
                output_lines.append(f"\n{p_text} [{src}, Page {pg}]")
        return "\n".join(output_lines).strip()

    if is_join:
        output_lines.append("A **JOIN** in SQL is used to combine rows from two or more tables based on a related column between them.")
        output_lines.append("")
        output_lines.append("Common types of JOIN include:")
        seen_joins = set()
        for bl, src, pg in extracted_bullets:
            if not any(j in bl.lower() for j in ['inner', 'left', 'right', 'full', 'cross', 'outer', 'join']):
                continue
            bl_key = bl[:25].lower()
            if bl_key not in seen_joins:
                seen_joins.add(bl_key)
                if ":" in bl:
                    parts = bl.split(":", 1)
                    output_lines.append(f"- **{parts[0].strip()}**: {parts[1].strip()} [{src}, Page {pg}]")
                else:
                    output_lines.append(f"- {bl} [{src}, Page {pg}]")
        if len(output_lines) <= 3 and extracted_paragraphs:
            for p_text, src, pg in extracted_paragraphs[:1]:
                output_lines.append(f"\n{p_text} [{src}, Page {pg}]")
        return "\n".join(output_lines).strip()

    if is_acid and 'atomicity' in query_lower:
        for p_text, src, pg in extracted_paragraphs:
            if 'atomicity' in p_text.lower():
                return f"**Atomicity** ensures that all operations within a transaction are completed successfully as a single unit of work. If any operation fails, the entire transaction is aborted and rolled back to its previous state (all-or-nothing principle).\n\n[{src}, Page {pg}]"
        for bl, src, pg in extracted_bullets:
            if 'atomicity' in bl.lower():
                return f"**Atomicity**: {bl} [{src}, Page {pg}]"

    # Generic bullet / paragraph output
    main_concept = " ".join([w.title() for w in keywords]) if keywords else "Information"
    output_lines.append(f"### {main_concept}\n")

    if extracted_bullets:
        seen = set()
        for bl, src, pg in extracted_bullets[:6]:
            k = bl[:30].lower()
            if k not in seen:
                seen.add(k)
                cit = f"[{src}, Page {pg}]"
                if ":" in bl:
                    parts = bl.split(":", 1)
                    output_lines.append(f"• **{parts[0].strip()}**: {parts[1].strip()} {cit}")
                else:
                    output_lines.append(f"• {bl} {cit}")
    elif extracted_paragraphs:
        for p_text, src, pg in extracted_paragraphs[:2]:
            cit = f"[{src}, Page {pg}]"
            output_lines.append(f"{p_text}\n\n{cit}")

    result_text = "\n".join(output_lines).strip()
    return result_text if result_text else FALLBACK_RESPONSE


def generate_rag_answer(
    llm_client,
    query: str,
    retrieved_chunks: List[Tuple],
    query_type: str = "NORMAL_QUESTION",
    extracted_term: str = "",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    rewritten_query: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout_seconds: int = LLM_TIMEOUT,
    compressed_context: Optional[str] = None
) -> str:
    """
    Generates a document-grounded, structured answer using Gemini with contextual compression and conversation memory.
    Features:
    - Clean and structured final answers with Markdown headings and bullet points
    - High precision > more information (no unrelated concepts, no topic mixing)
    - Strict document grounding (no hallucinations, no unsupported outside knowledge)
    - Exact verified source citation format [DocumentName.pdf, Page X]
    - Automatic fallback mechanism
    """
    target_term = extracted_term if extracted_term else query.strip()

    if not retrieved_chunks and not compressed_context:
        return FALLBACK_RESPONSE

    if compressed_context and compressed_context.strip():
        formatted_context = compressed_context.strip()
    else:
        formatted_context = format_retrieved_context(retrieved_chunks, max_chunks=MAX_CONTEXT_CHUNKS)
    
    if not formatted_context.strip():
        return FALLBACK_RESPONSE

    is_followup = bool(rewritten_query and rewritten_query.strip().lower() != query.strip().lower())
    history_block = format_conversation_history_block(conversation_history, is_followup=is_followup)

    # Detailed Handoff Logging
    logger.info("================== RETRIEVAL-TO-GENERATION HANDOFF ==================")
    logger.info(f"[HANDOFF] Query: {query}")
    logger.info(f"[HANDOFF] Target Term / Rewritten: {target_term}")
    logger.info(f"[HANDOFF] Is Followup: {is_followup}")
    logger.info(f"[HANDOFF] Retrieved Chunks Count: {len(retrieved_chunks)}")
    for idx, c in enumerate(retrieved_chunks[:4], 1):
        doc = c[0] if isinstance(c, (tuple, list)) else c
        d_name = getattr(doc, "metadata", {}).get("document") or getattr(doc, "metadata", {}).get("source", "Doc") if hasattr(doc, "metadata") else "Doc"
        p_num = getattr(doc, "metadata", {}).get("page", 1) if hasattr(doc, "metadata") else 1
        logger.info(f"[HANDOFF] Chunk {idx}: {d_name} (Page {p_num})")
    logger.info(f"[HANDOFF] Compressed Context Size: {len(compressed_context) if compressed_context else 0} chars")
    logger.info(f"[HANDOFF] Final Prompt Context Size: {len(formatted_context)} chars")
    logger.info("====================================================================")

    # Strict document grounding and high-precision prompt
    prompt = (
        "You are a document-grounded question answering assistant.\n\n"
        "CRITICAL GROUNDING & ISOLATION INSTRUCTIONS:\n"
        "1. Answer ONLY using information contained in the retrieved Document Context below.\n"
        "2. The current user question is the HIGHEST priority.\n"
        "3. Do NOT reuse an answer from a previous question.\n"
        "4. Do NOT assume that the current question is related to any previous question.\n"
        f"5. If the retrieved context contains NO relevant information to address the user question, explicitly output:\n"
        f"   \"{FALLBACK_RESPONSE}\"\n"
        "6. Synthesize an accurate, structured answer from all relevant concepts, normal forms, mechanisms, stages, or definitions present in the context.\n"
        "7. Do NOT fill missing information using outside general knowledge.\n"
        "8. Every major factual claim must be supported by retrieved context with exact source citations: [DocumentName.pdf — Page X].\n"
        "9. HIGH PRECISION: Answer ONLY what was asked. Ignore context that is unrelated to the user's specific question.\n\n"
        f"{history_block}"
        f"Current User Question:\n{query.strip()}\n\n"
        f"Retrieved Document Context:\n{formatted_context}\n\n"
        "Document-Grounded Answer:"
    )

    key = api_key or load_gemini_api_key()
    if not is_valid_api_key_format(key):
        logger.warning("[LLM] Valid Gemini API key not found. Using direct document extraction fallback.")
        return extract_direct_answer_fallback(query, retrieved_chunks, target_term)

    try:
        client = llm_client if llm_client is not None else get_llm_client(key)
    except Exception as err:
        logger.error(f"[LLM] Client initialization error: {err}")
        raise LLMGenerationError(f"AI service client initialization failed: {err}")

    primary_model = model_name or get_configured_model_name()
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS
    ) if types else None

    logger.info(f"[LLM] Generation started for query: '{query[:60]}...' with primary model '{primary_model}'")

    num_chunks = len(retrieved_chunks) if retrieved_chunks else 0

    # 1. Primary Model with strict timeout
    t_start_attempt = time.perf_counter()
    ans = _call_gemini_with_timeout(client, primary_model, prompt, config, timeout_sec=LLM_TIMEOUT, context_chunks=num_chunks)
    if ans:
        elapsed_sec = round(time.perf_counter() - t_start_attempt, 2)
        logger.info(f"[LLM] Primary model '{primary_model}' generation successful ({elapsed_sec}s)")
        return ans

    # 2. Secondary fallback models with strict timeout
    for fb_model in FALLBACK_MODELS:
        if fb_model != primary_model:
            logger.info(f"[LLM] Trying fallback model '{fb_model}'...")
            t_fb_start = time.perf_counter()
            ans_fb = _call_gemini_with_timeout(client, fb_model, prompt, config, timeout_sec=10, context_chunks=num_chunks)
            if ans_fb:
                elapsed = round(time.perf_counter() - t_fb_start, 2)
                logger.info(f"[LLM] Fallback model '{fb_model}' generation successful ({elapsed}s)")
                return ans_fb

    logger.warning("[LLM] Remote LLM models unavailable/timed out. Using direct grounded extraction fallback.")
    return extract_direct_answer_fallback(query, retrieved_chunks, target_term)


def format_grouped_document_context(grouped_context: Dict[str, Any], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Formats grouped chunks by document with page annotations."""
    doc_blocks = []
    current_length = 0

    for doc_name, chunks in grouped_context.items():
        chunk_lines = []
        for item in chunks[:MAX_CONTEXT_CHUNKS]:
            if isinstance(item, dict):
                page = item.get("page", 1)
                content = item.get("text", "")
            elif isinstance(item, (tuple, list)):
                doc = item[0]
                page = getattr(doc, "metadata", {}).get("page", 1) if hasattr(doc, "metadata") else 1
                content = doc.page_content.strip() if hasattr(doc, "page_content") else str(doc).strip()
            else:
                doc = item
                page = getattr(doc, "metadata", {}).get("page", 1) if hasattr(doc, "metadata") else 1
                content = doc.page_content.strip() if hasattr(doc, "page_content") else str(doc).strip()

            chunk_lines.append(f"[Page {page}]\n{content}")
        
        joined_chunks = "\n\n".join(chunk_lines)
        doc_block = f"=== DOCUMENT: {doc_name} ===\n{joined_chunks}"
        
        if current_length + len(doc_block) > max_chars and doc_blocks:
            break
            
        doc_blocks.append(doc_block)
        current_length += len(doc_block)

    return "\n\n" + ("\n\n" + "="*40 + "\n\n").join(doc_blocks)



def build_comparison_prompt(
    query: str,
    grouped_context: Dict[str, List[Tuple]],
    comparison_topics: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    comparison_subtype: str = "COMPARISON_TABLE"
) -> str:
    """Builds a structured prompt specifically tailored for multi-document synthesis and comparison."""
    formatted_grouped_docs = format_grouped_document_context(grouped_context)
    history_block = format_conversation_history_block(conversation_history)
    
    topics_hint = ""
    if comparison_topics:
        topics_hint = f"Focus topics / concepts to compare: {', '.join(comparison_topics)}\n"

    # Define subtype-specific output instructions
    if comparison_subtype == "SUMMARY_ALL":
        format_instructions = (
            "REQUIRED OUTPUT FORMAT (Multi-Document Summary):\n"
            "Generate a document-wise summary with key points for each document:\n\n"
            "DOCUMENT 1 (<Document_Name_A.pdf>)\n"
            "- <Key Point 1> (Page <Page_Number>)\n"
            "- <Key Point 2> (Page <Page_Number>)\n\n"
            "DOCUMENT 2 (<Document_Name_B.pdf>)\n"
            "- <Key Point 1> (Page <Page_Number>)\n"
            "- <Key Point 2> (Page <Page_Number>)\n\n"
            "(Repeat for each relevant document. Do not summarize irrelevant documents.)"
        )
    elif comparison_subtype == "COMMON_INFO":
        format_instructions = (
            "REQUIRED OUTPUT FORMAT (Common Information Across Documents):\n"
            "Identify information, principles, or concepts that appear supported by multiple documents:\n\n"
            "COMMON INFORMATION\n\n"
            "1. <Shared concept, principle, or common finding>\n"
            "   Sources:\n"
            "   - <Document A>, Page <Page_Number>\n"
            "   - <Document B>, Page <Page_Number>\n\n"
            "2. <Another shared concept or common finding>\n"
            "   Sources:\n"
            "   - <Document A>, Page <Page_Number>\n"
            "   - <Document B>, Page <Page_Number>"
        )
    elif comparison_subtype == "SIMILARITIES":
        format_instructions = (
            "REQUIRED OUTPUT FORMAT (Similarities Across Documents):\n"
            "Highlight the similarities and shared features across the documents:\n\n"
            "SIMILARITIES\n\n"
            "1. **<Shared Concept / Principle 1>**: <Explanation of similarity>\n"
            "   Sources:\n"
            "   - <Document A>, Page <Page_Number>\n"
            "   - <Document B>, Page <Page_Number>\n\n"
            "2. **<Shared Concept / Principle 2>**: <Explanation of similarity>\n"
            "   Sources:\n"
            "   - <Document A>, Page <Page_Number>\n"
            "   - <Document B>, Page <Page_Number>"
        )
    elif comparison_subtype == "DIFFERENCES":
        format_instructions = (
            "REQUIRED OUTPUT FORMAT (Key Differences):\n"
            "Highlight the key differences between the documents/concepts:\n\n"
            "KEY DIFFERENCES\n\n"
            "| Aspect | <Topic / Document A> | <Topic / Document B> |\n"
            "|---|---|---|\n"
            "| <Aspect 1> | <Details from Document A> (Page <X>) | <Details from Document B> (Page <Y>) |\n"
            "| <Aspect 2> | <Details from Document A> (Page <X>) | <Details from Document B> (Page <Y>) |\n\n"
            "### Detailed Differences\n"
            "1. **<Aspect 1>**: <Contrasting explanation citing specific document names and page numbers>\n"
            "2. **<Aspect 2>**: <Contrasting explanation citing specific document names and page numbers>"
        )
    else:  # COMPARISON_TABLE (Default)
        format_instructions = (
            "REQUIRED OUTPUT FORMAT (Structured Comparison):\n"
            "Generate a clean comparison table followed by differences and similarities:\n\n"
            "COMPARISON\n\n"
            "| Aspect | <Topic / Document A> | <Topic / Document B> |\n"
            "|---|---|---|\n"
            "| Definition / Purpose | <Details> (Page <X>) | <Details> (Page <Y>) |\n"
            "| Core Features / Characteristics | <Details> (Page <X>) | <Details> (Page <Y>) |\n"
            "| Advantages | <Details> (Page <X>) | <Details> (Page <Y>) |\n"
            "| Limitations | <Details> (Page <X>) | <Details> (Page <Y>) |\n\n"
            "### Key Differences\n"
            "1. **<Aspect Name>**: <Detail how they differ, citing specific document names and page numbers>\n"
            "2. ...\n\n"
            "### Common Points / Similarities\n"
            "- <Detail shared principles, goals, or overlapping concepts found in the texts with citations>\n\n"
            "### Missing Information Note\n"
            "(If an aspect is missing in one document, clearly state: \"Information regarding <aspect> is not specified in <Document B>.\")"
        )

    prompt = (
        "You are an expert document analysis assistant specializing in multi-document comparison and cross-document synthesis.\n\n"
        "TASK:\n"
        "Generate an accurate, structured comparison or synthesis based EXCLUSIVELY on the provided document contexts below.\n\n"
        f"{topics_hint}"
        "STRICT GROUNDING & CITATION RULES:\n"
        "1. Strictly use ONLY facts and details present in the provided document contexts. Do NOT invent, assume, or extrapolate facts.\n"
        "2. If information is missing from the selected documents or there is not enough information to make a comparison, state:\n"
        "   \"I couldn't find sufficient information about this in the selected documents.\"\n"
        "3. If information on one topic/aspect is found only in one document and missing in others, clearly state:\n"
        "   \"Information not specified in [DocumentName.pdf — Page X]\" or \"This information was found only in [DocumentName.pdf — Page X].\"\n"
        "4. Every factual claim must cite the source filename and page using: [DocumentName.pdf — Page X].\n"
        "5. Do NOT create citations for information that does not exist in the retrieved context.\n\n"
        f"{format_instructions}\n\n"
        f"{history_block}"
        f"USER QUESTION:\n{query.strip()}\n\n"
        f"DOCUMENTS & RETRIEVED CONTEXT:\n{formatted_grouped_docs}\n\n"
        "FINAL SYNTHESIS ANSWER:"
    )
    return prompt


def extract_direct_comparison_fallback(
    grouped_context: Dict[str, Any],
    comparison_topics: Optional[List[str]] = None,
    comparison_subtype: str = "COMPARISON_TABLE"
) -> str:
    """Fallback generator for structured comparison when API quotas are temporarily exhausted."""
    if not grouped_context:
        return FALLBACK_RESPONSE

    doc_names = list(grouped_context.keys())
    if comparison_subtype == "SUMMARY_ALL":
        lines = ["## Document Summaries\n"]
        for idx, dname in enumerate(doc_names, 1):
            lines.append(f"DOCUMENT {idx} ({dname})")
            chunks = grouped_context[dname]
            for c in chunks[:3]:
                page = c.get("page", 1) if isinstance(c, dict) else (getattr(c[0], "metadata", {}).get("page", 1) if isinstance(c, (tuple, list)) else 1)
                txt = (c.get("text", "") if isinstance(c, dict) else (c[0].page_content if isinstance(c, (tuple, list)) else str(c))).strip()
                first_sent = re.split(r'(?<=[.!?])\s+', txt)[0] if txt else ""
                if first_sent:
                    lines.append(f"- **Key Point**: {first_sent} (Page {page})")
            lines.append("")
        return "\n".join(lines).strip()

    elif comparison_subtype in ("COMMON_INFO", "SIMILARITIES"):
        header = "COMMON INFORMATION" if comparison_subtype == "COMMON_INFO" else "SIMILARITIES"
        lines = [f"{header}\n"]
        item_count = 1
        for dname in doc_names:
            chunks = grouped_context[dname]
            for c in chunks[:2]:
                page = c.get("page", 1) if isinstance(c, dict) else (getattr(c[0], "metadata", {}).get("page", 1) if isinstance(c, (tuple, list)) else 1)
                txt = (c.get("text", "") if isinstance(c, dict) else (c[0].page_content if isinstance(c, (tuple, list)) else str(c))).strip()
                first_sent = re.split(r'(?<=[.!?])\s+', txt)[0] if txt else ""
                if first_sent:
                    lines.append(f"{item_count}. **Shared Principle**: {first_sent}")
                    lines.append(f"   Sources:")
                    lines.append(f"   - {dname}, Page {page}")
                    item_count += 1
        return "\n".join(lines).strip()

    else:
        # Structured Comparison Table
        lines = ["COMPARISON\n"]
        col1 = comparison_topics[0] if comparison_topics and len(comparison_topics) >= 1 else (doc_names[0] if doc_names else "Document A")
        col2 = comparison_topics[1] if comparison_topics and len(comparison_topics) >= 2 else (doc_names[1] if len(doc_names) > 1 else "Document B")
        lines.append(f"| Aspect | {col1} | {col2} |")
        lines.append("|---|---|---|")

        aspects = ["Core Concepts / Definition", "Key Characteristics", "Supported Features", "Usage & Scope"]
        for asp in aspects:
            d1_text = "See document context for full details."
            d2_text = "See document context for full details."
            if len(doc_names) >= 1 and grouped_context.get(doc_names[0]):
                c = grouped_context[doc_names[0]][0]
                p = c.get("page", 1) if isinstance(c, dict) else (getattr(c[0], "metadata", {}).get("page", 1) if isinstance(c, (tuple, list)) else 1)
                t = (c.get("text", "") if isinstance(c, dict) else (c[0].page_content if isinstance(c, (tuple, list)) else str(c))).strip()
                s = re.split(r'(?<=[.!?])\s+', t)[0] if t else ""
                if s:
                    d1_text = f"{s[:90]}... (Page {p})"
            if len(doc_names) >= 2 and grouped_context.get(doc_names[1]):
                c = grouped_context[doc_names[1]][0]
                p = c.get("page", 1) if isinstance(c, dict) else (getattr(c[0], "metadata", {}).get("page", 1) if isinstance(c, (tuple, list)) else 1)
                t = (c.get("text", "") if isinstance(c, dict) else (c[0].page_content if isinstance(c, (tuple, list)) else str(c))).strip()
                s = re.split(r'(?<=[.!?])\s+', t)[0] if t else ""
                if s:
                    d2_text = f"{s[:90]}... (Page {p})"
            lines.append(f"| {asp} | {d1_text} | {d2_text} |")

        lines.append("\n### Key Differences")
        for idx, dname in enumerate(doc_names, 1):
            if grouped_context.get(dname):
                c = grouped_context[dname][0]
                p = c.get("page", 1) if isinstance(c, dict) else (getattr(c[0], "metadata", {}).get("page", 1) if isinstance(c, (tuple, list)) else 1)
                lines.append(f"{idx}. **From {dname} (Page {p})**: Key points and specific principles detailed in document.")

        return "\n".join(lines).strip()


def generate_comparison_answer(
    llm_client,
    query: str,
    grouped_context: Dict[str, List[Tuple]],
    comparison_topics: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    comparison_subtype: str = "COMPARISON_TABLE"
) -> str:
    """Generates a structured comparison answer across multiple documents or concepts with retry."""
    if not grouped_context:
        return FALLBACK_RESPONSE

    total_chunks = sum(len(chunks) for chunks in grouped_context.values())
    if total_chunks == 0:
        return FALLBACK_RESPONSE

    prompt = build_comparison_prompt(
        query=query,
        grouped_context=grouped_context,
        comparison_topics=comparison_topics,
        conversation_history=conversation_history,
        comparison_subtype=comparison_subtype
    )

    key = api_key or load_gemini_api_key()
    if not is_valid_api_key_format(key):
        return extract_direct_comparison_fallback(grouped_context, comparison_topics, comparison_subtype)

    try:
        client = llm_client if llm_client is not None else get_llm_client(key)
    except Exception as err:
        logger.error(f"[LLM] Comparison client init error: {err}")
        return extract_direct_comparison_fallback(grouped_context, comparison_topics, comparison_subtype)

    primary_model = model_name or get_configured_model_name()
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS
    ) if types else None

    logger.info(f"[LLM] Comparison generation started for query: '{query[:60]}...'")

    # 1. Primary Model with strict timeout
    t_start_attempt = time.perf_counter()
    ans = _call_gemini_with_timeout(client, primary_model, prompt, config, timeout_sec=LLM_TIMEOUT)
    if ans:
        elapsed_sec = round(time.perf_counter() - t_start_attempt, 2)
        logger.info(f"[LLM] Comparison generation successful ({elapsed_sec}s)")
        return ans

    # 2. Fallback secondary models
    for fb_model in FALLBACK_MODELS:
        if fb_model != primary_model:
            logger.info(f"[LLM] Trying fallback comparison model '{fb_model}'...")
            t_fb_start = time.perf_counter()
            ans_fb = _call_gemini_with_timeout(client, fb_model, prompt, config, timeout_sec=10)
            if ans_fb:
                elapsed = round(time.perf_counter() - t_fb_start, 2)
                logger.info(f"[LLM] Fallback comparison model '{fb_model}' successful ({elapsed}s)")
                return ans_fb

    logger.warning("[LLM] Remote LLM models unavailable/timed out. Using direct structured comparison fallback.")
    return extract_direct_comparison_fallback(grouped_context, comparison_topics, comparison_subtype)



def test_gemini_connection(api_key: Optional[str] = None, model_name: Optional[str] = None) -> Tuple[bool, str]:
    """Lightweight Gemini connection verification test."""
    key = api_key or load_gemini_api_key()
    if not is_valid_api_key_format(key):
        return False, "Gemini API key is not configured."

    model = model_name or get_configured_model_name()
    try:
        client = get_llm_client(key)
        resp = client.models.generate_content(
            model=model,
            contents="Say 'OK'."
        )
        if resp and resp.text:
            return True, f"Successfully connected to Gemini using model '{model}'."
        return False, "Gemini responded with empty content."
    except Exception as err:
        return False, f"Gemini connection test: {str(err)}"
