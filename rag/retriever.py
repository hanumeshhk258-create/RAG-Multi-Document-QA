import os
import re
import math
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any, Set
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Core technical keywords registry for high-precision entity extraction
TECHNICAL_KEYWORDS = {
    # DBMS & Database Concepts
    'acid', 'acid properties', 'atomicity', 'consistency', 'isolation', 'durability',
    'transaction', 'transactions', 'concurrency', 'concurrency control',
    'normalization', '1nf', '2nf', '3nf', 'bcnf', 'first normal form', 'second normal form',
    'third normal form', 'boyce codd normal form', 'normal forms', 'functional dependency',
    'primary key', 'foreign key', 'candidate key', 'super key', 'composite key',
    'unique key', 'database', 'dbms', 'rdbms', 'schema', 'table', 'relational database',
    'er model', 'entity relationship', 'indexing', 'b-tree', 'btree', 'hash index',
    'serializability', 'deadlock', 'two phase locking', '2pl', 'recovery', 'log based recovery',
    'view', 'stored procedure', 'trigger', 'cursor',

    # SQL Commands & Operations
    'sql', 'join', 'joins', 'inner join', 'left join', 'left outer join',
    'right join', 'right outer join', 'full join', 'full outer join',
    'cross join', 'self join', 'natural join', 'outer join',
    'dml', 'dml commands', 'ddl', 'ddl commands', 'dcl', 'tcl',
    'select', 'insert', 'update', 'delete', 'create', 'alter', 'drop', 'truncate',
    'querying data commands', 'querying commands', 'querying data', 'sql commands', 'data querying',
    'group by', 'order by', 'having', 'where', 'distinct', 'union', 'intersect',
    'except', 'aggregate functions', 'count', 'sum', 'avg', 'min', 'max',
    'subquery', 'nested query', 'correlated subquery', 'cte', 'grant', 'revoke',
    'commit', 'rollback', 'savepoint',

    # Python & Programming Concepts
    'python', 'list', 'lists', 'tuple', 'tuples', 'dict', 'dicts', 'dictionary',
    'dictionaries', 'set', 'sets', 'generator', 'generators', 'iterator',
    'list comprehension', 'decorator', 'lambda', 'oop', 'class', 'object',
    'inheritance', 'polymorphism', 'encapsulation', 'abstraction',
    'exception handling', 'try except', 'gil', 'global interpreter lock',
    'asyncio', 'multithreading', 'multiprocessing',

    # Data Structures & Algorithms
    'array', 'linked list', 'stack', 'queue', 'binary tree', 'graph',
    'binary search', 'sorting', 'recursion', 'dynamic programming',
    'time complexity', 'space complexity', 'big o'
}

# Domain acronym expansions and component mappings
TERM_SYNONYM_MAP = {
    'acid': ['acid', 'acid properties', 'atomicity', 'consistency', 'isolation', 'durability', 'transaction', 'transactions'],
    'acid properties': ['acid', 'acid properties', 'atomicity', 'consistency', 'isolation', 'durability', 'transaction'],
    'atomicity': ['atomicity', 'all-or-nothing', 'transaction', 'commit', 'rollback'],
    'consistency': ['consistency', 'valid state', 'constraints', 'transaction execution'],
    'isolation': ['isolation', 'concurrent', 'intermediate state', 'independent transactions'],
    'durability': ['durability', 'permanent', 'survives crash', 'power failure', 'committed changes'],
    'dml': ['dml', 'dml commands', 'data manipulation language', 'select', 'insert', 'update', 'delete', 'querying data commands'],
    'dml commands': ['dml', 'dml commands', 'data manipulation language', 'select', 'insert', 'update', 'delete', 'querying data commands'],
    'querying data commands': ['querying data commands', 'querying data', 'select', 'insert', 'update', 'delete', 'dml', 'sql commands', 'queries'],
    'querying data': ['querying data', 'querying data commands', 'select', 'insert', 'update', 'delete', 'dml', 'sql commands'],
    'querying commands': ['querying commands', 'querying data commands', 'select', 'insert', 'update', 'delete', 'dml'],
    'sql commands': ['sql commands', 'querying data commands', 'select', 'insert', 'update', 'delete', 'create', 'alter', 'drop', 'truncate', 'dml', 'ddl'],
    'ddl': ['ddl', 'ddl commands', 'data definition language', 'create', 'alter', 'drop', 'truncate'],
    'ddl commands': ['ddl', 'ddl commands', 'data definition language', 'create', 'alter', 'drop', 'truncate'],
    'tcl': ['tcl', 'transaction control', 'commit', 'rollback', 'savepoint'],
    'dcl': ['dcl', 'data control', 'grant', 'revoke'],
    'join': ['join', 'joins', 'joining commands', 'inner join', 'left join', 'right join', 'full join', 'cross join', 'self join', 'natural join'],
    'joins': ['join', 'joins', 'joining commands', 'inner join', 'left join', 'right join', 'full join', 'cross join', 'self join', 'natural join'],
    'joining commands': ['join', 'joins', 'joining commands', 'inner join', 'left join', 'right join', 'full join', 'cross join'],
    'normalization': ['normalization', 'normal forms', '1nf', '2nf', '3nf', 'bcnf', 'redundancy', 'data integrity'],
    'normal form': ['normalization', 'normal forms', '1nf', '2nf', '3nf', 'bcnf'],
    'sql': ['sql', 'structured query language', 'relational database', 'select', 'table', 'queries'],
    'python': ['python', 'programming language', 'dynamically-typed', 'lists', 'tuples', 'dictionaries', 'sets']
}

# Synonym mapping for abbreviations and plurals
TERM_NORMALIZATION_MAP = {
    'joins': 'join',
    'keys': 'key',
    'primary-key': 'primary key',
    'foreign-key': 'foreign key',
    'databases': 'database',
    'transactions': 'transaction',
    'tables': 'table',
    'lists': 'list',
    'tuples': 'tuple',
    'dictionaries': 'dictionary',
    'dicts': 'dictionary',
    'sets': 'set',
    'k-means': 'kmeans',
    'b-tree': 'btree'
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


try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False

# Configurable Top-K Strategy & Smart Source Selection Constants
VECTOR_TOP_K: int = 15
BM25_TOP_K: int = 15
VECTOR_WEIGHT: float = 0.6
BM25_WEIGHT: float = 0.4
RERANK_CANDIDATES: int = 20
FINAL_TOP_K: int = 5

# Smart Source Selection & Evidence Control Configurable Parameters
MAX_RELEVANT_DOCUMENTS: int = int(os.getenv("MAX_RELEVANT_DOCUMENTS", "3"))
MAX_CHUNKS_PER_DOCUMENT: int = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "4"))
MIN_DOCUMENT_RELEVANCE: float = float(os.getenv("MIN_DOCUMENT_RELEVANCE", "0.40"))
MIN_CHUNK_RELEVANCE: float = float(os.getenv("MIN_CHUNK_RELEVANCE", "0.35"))
MIN_RELEVANCE_THRESHOLD: float = MIN_CHUNK_RELEVANCE

# Aliases for backward compatibility
SEMANTIC_TOP_K: int = VECTOR_TOP_K
KEYWORD_TOP_K: int = BM25_TOP_K
FUSION_TOP_K: int = RERANK_CANDIDATES
SEMANTIC_WEIGHT: float = VECTOR_WEIGHT
KEYWORD_WEIGHT: float = BM25_WEIGHT
TOP_K_RETRIEVAL: int = VECTOR_TOP_K


def calculate_document_relevance(chunks: List[Dict[str, Any]]) -> float:
    """
    Calculates document-level relevance score as a weighted average of its top relevant chunks.
    Formula:
        weights for top chunks: [0.60, 0.25, 0.15]
        document_score = sum(w_i * chunk_score_i) / sum(w_i)
    """
    if not chunks:
        return 0.0
    
    scores = sorted([c.get("reranker_score", c.get("final_score", 0.0)) for c in chunks], reverse=True)
    weights = [0.60, 0.25, 0.15]
    total_w = 0.0
    weighted_sum = 0.0
    for idx, s in enumerate(scores[:len(weights)]):
        w = weights[idx]
        weighted_sum += w * s
        total_w += w
    
    if total_w == 0.0:
        return 0.0
    return round(weighted_sum / total_w, 4)


import hashlib

class CrossEncoderManager:
    """
    Singleton manager for Cross-Encoder reranking model.
    Loads cross-encoder/ms-marco-MiniLM-L-6-v2 once, caches instance,
    predicts (query, chunk_text) relevance, and applies sigmoid normalization.
    Includes in-memory LRU pair score caching for sub-millisecond repeated scoring.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._initialized = False
        self._available = False
        self._score_cache: Dict[str, float] = {}
        self._max_cache_size: int = 5000

    def clear_cache(self):
        """Clears the internal pair score cache."""
        self._score_cache.clear()

    def load_model(self):
        if self._initialized:
            return self._model
        self._initialized = True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._available = True
            logger.info(f"CrossEncoder model '{self.model_name}' initialized successfully.")
        except Exception as err:
            logger.warning(f"Could not load CrossEncoder model '{self.model_name}': {err}. Using high-precision fallback reranker.")
            self._model = None
            self._available = False
        return self._model

    def is_available(self) -> bool:
        if not self._initialized:
            self.load_model()
        return self._available

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> Optional[List[float]]:
        if not pairs:
            return []
        
        # Check cache for every pair
        final_scores: List[Optional[float]] = [None] * len(pairs)
        uncached_indices: List[int] = []
        uncached_pairs: List[Tuple[str, str]] = []
        uncached_keys: List[str] = []

        for idx, (q, t) in enumerate(pairs):
            q_clean = q.strip().lower()
            t_snippet = (t or "")[:350].strip().lower()
            pair_key = hashlib.md5(f"{q_clean}::{t_snippet}".encode("utf-8")).hexdigest()
            if pair_key in self._score_cache:
                final_scores[idx] = self._score_cache[pair_key]
            else:
                uncached_indices.append(idx)
                uncached_pairs.append((q, t_snippet))
                uncached_keys.append(pair_key)

        if not uncached_pairs:
            return [s for s in final_scores if s is not None]

        model = self.load_model()
        if not model:
            return None

        try:
            raw_scores = model.predict(uncached_pairs, batch_size=8, show_progress_bar=False)
            for orig_idx, key, sc in zip(uncached_indices, uncached_keys, raw_scores):
                float_sc = float(sc)
                final_scores[orig_idx] = float_sc
                if len(self._score_cache) < self._max_cache_size:
                    self._score_cache[key] = float_sc
            return [s for s in final_scores if s is not None]
        except Exception as err:
            logger.warning(f"CrossEncoder prediction error: {err}. Using fallback reranker.")
            return None


_CROSS_ENCODER = CrossEncoderManager()


# =============================================================================
# 1. QUERY CLASSIFICATION & TOPIC EXTRACTION
# =============================================================================
def classify_query(query: str) -> Tuple[str, str]:
    """
    Classifies the user query into:
    - 'ONE_WORD': Single term/concept (e.g. 'SQL', 'JOIN', 'ACID', 'Python')
    - 'SHORT_DEFINITION': Direct definition question (e.g. 'What is SQL?', 'Define JOIN', 'Primary Key definition')
    - 'COMPLEX_QUESTION': Multi-part/comparative/detailed question (e.g. 'Explain normalization with examples', 'Compare SQL and Python')
    - 'NORMAL_QUESTION': Standard questions (e.g. 'What are the types of joins?', 'What is normalization?')
    
    Returns:
        (query_type, extracted_term)
    """
    if not query:
        return "NORMAL_QUESTION", ""

    clean = query.strip()
    words = clean.split()
    lower = clean.lower()

    # Complex question patterns
    complex_triggers = [
        'with example', 'with examples', 'explain in detail', 'compare', 'difference between',
        'how does', 'step by step', 'advantages and disadvantages', 'pros and cons',
        'working of', 'architecture of', 'types and examples'
    ]
    if any(trigger in lower for trigger in complex_triggers) or (len(words) >= 9):
        return "COMPLEX_QUESTION", clean

    # Definition patterns: "What is X?", "Define X", "X definition", "Explain X", "What is meant by X"
    def_patterns = [
        (r'^(what is the meaning of|what is meant by|what do you mean by)\s+(.+?)[\?!.,;]?$', 2),
        (r'^(what is|what are|what was|what were)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s\-]+)[\?!.,;]?$', 2),
        (r'^(define|explain|describe)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s\-]+)[\?!.,;]?$', 2),
        (r'^([a-zA-Z0-9_\s\-]+)\s+(definition|meaning|stands for)[\?!.,;]?$', 1),
    ]

    for pattern, group_idx in def_patterns:
        m = re.match(pattern, clean, flags=re.IGNORECASE)
        if m:
            term = m.group(group_idx).strip()
            term_words = term.split()
            if len(term_words) <= 3:
                return "SHORT_DEFINITION", term

    question_starters = {'why', 'how', 'when', 'where', 'who', 'which', 'can', 'could', 'should', 'would', 'is', 'are'}

    # Check for single word
    if len(words) == 1:
        cleaned_term = re.sub(r'[?!.,;:]+$', '', clean).strip()
        first_word = words[0].lower().rstrip('?!.,;:')
        if first_word not in question_starters:
            return "ONE_WORD", cleaned_term

    # Check for 2-3 word multi-term queries (e.g. "SQL JOIN", "PRIMARY KEY", "DBMS ACID")
    if 2 <= len(words) <= 3:
        first_word = words[0].lower().rstrip('?!.,;:')
        if first_word not in question_starters:
            cleaned_term = re.sub(r'[?!.,;:]+$', '', clean).strip()
            return "SHORT_DEFINITION", cleaned_term

    # Check normalized term length
    norm_term = normalize_query(clean)
    norm_words = norm_term.split()
    if len(norm_words) == 1 and norm_words[0].lower() not in question_starters:
        return "SHORT_DEFINITION", norm_term
    elif 2 <= len(norm_words) <= 3 and norm_words[0].lower() not in question_starters:
        return "SHORT_DEFINITION", norm_term

    return "NORMAL_QUESTION", clean


def normalize_retrieval_query(query: str) -> str:
    """
    Creates a clean, enriched retrieval query containing the original question,
    vital technical keywords, and topic terms without generic query replacement.
    Examples:
    - 'What is Waterfall Model?' -> 'Waterfall Model software development lifecycle SDLC phases sequential model'
    - 'What is Key Security and Trust Controls?' -> 'Key Security and Trust Controls security authentication authorization trust mechanisms'
    - 'What are SQL JOINs?' -> 'SQL JOINs inner join left join right join full join cross join relational tables'
    - 'What is normalization in DBMS?' -> 'normalization in DBMS normal forms 1NF 2NF 3NF BCNF functional dependency'
    - 'What is Atomicity?' -> 'Atomicity ACID properties database transactions all-or-nothing commit rollback'
    - 'What is IoT?' -> 'IoT Internet of Things connected smart devices sensors network'
    - 'What is machine learning?' -> 'machine learning ML models training supervised unsupervised algorithms'
    - 'What is conclusion?' -> 'conclusion summary outcomes project takeaways'
    """
    if not query:
        return ""

    clean = query.strip()
    lower = clean.lower()

    expansion_map = {
        'waterfall': 'Waterfall Model software development lifecycle SDLC sequential phases requirements design implementation testing',
        'sdlc': 'software development lifecycle SDLC phases sequential iterative requirements design',
        'security': 'security trust controls authentication authorization encryption access control integrity',
        'trust controls': 'security trust controls authentication authorization key management access control',
        'trust control': 'security trust controls authentication authorization key management access control',
        'escrow': 'key escrow cryptographic keys threshold cryptography master decryption key recovery',
        'join': 'SQL JOIN inner join left join right join full join cross join table combining',
        'joins': 'SQL JOIN inner join left join right join full join cross join table combining',
        'joining': 'SQL JOIN inner join left join right join full join cross join table combining',
        'joining commands': 'SQL JOIN inner join left join right join full join cross join table combining',
        'querying data commands': 'querying data commands SQL SELECT INSERT UPDATE DELETE DML DDL relational database',
        'sql commands': 'SQL commands SELECT INSERT UPDATE DELETE CREATE ALTER DROP DML DDL',
        'dml': 'DML Data Manipulation Language SELECT INSERT UPDATE DELETE querying data',
        'ddl': 'DDL Data Definition Language CREATE ALTER DROP TRUNCATE schema structure',
        'atomicity': 'Atomicity ACID properties database transactions all-or-nothing rollback commit',
        'acid': 'ACID properties Atomicity Consistency Isolation Durability transactions database',
        'normalization': 'normalization normal forms 1NF 2NF 3NF BCNF relational redundancy functional dependency',
        'primary key': 'primary key unique identifier candidate key foreign key relational table constraint',
        'foreign key': 'foreign key referential integrity primary key relational table reference constraint',
        'iot': 'IoT Internet of Things connected smart devices sensors embedded network protocols',
        'machine learning': 'machine learning ML algorithms training supervised unsupervised neural models',
        'conclusion': 'conclusion project summary final remarks outcomes future scope takeaways',
        'python': 'Python programming data structures lists tuples dictionaries decorators generators'
    }

    matched_expansions = []
    for trigger, expansion in expansion_map.items():
        if re.search(rf'\b{re.escape(trigger)}\b', lower):
            matched_expansions.append(expansion)

    if matched_expansions:
        # Keep original query at the front, append domain keywords
        unique_tokens = []
        for exp in matched_expansions:
            for token in exp.split():
                if token.lower() not in clean.lower() and token.lower() not in [t.lower() for t in unique_tokens]:
                    unique_tokens.append(token)
        if unique_tokens:
            return f"{clean} {' '.join(unique_tokens[:8])}"

    return clean


def classify_query_intent_and_topic(query: str) -> Dict[str, Any]:
    """
    Classifies the user query into a clean, human-readable topic and category.
    Examples:
    - 'What is Querying Data Commands?' -> 'SQL / Data Querying / SQL Commands' (Category: SQL_COMMANDS)
    - 'What is Atomicity?' -> 'DBMS / ACID / Atomicity' (Category: DBMS_ACID)
    - 'What are types of JOIN?' -> 'SQL / JOIN' (Category: SQL_JOIN)
    - 'What is normalization?' -> 'DBMS / Normalization' (Category: DBMS_NORMALIZATION)
    - 'What is Waterfall Model?' -> 'Software Engineering / Waterfall' (Category: SOFTWARE_ENGINEERING)
    - 'What is Key Security and Trust Controls?' -> 'Security / Trust & Security Controls' (Category: SECURITY_CONTROLS)
    - 'What is Conclusion?' -> 'Project Summary / Conclusion' (Category: CONCLUSION)
    - 'What is IoT?' -> 'IoT / Internet of Things' (Category: IOT)
    - 'What is machine learning?' -> 'AI / Machine Learning' (Category: MACHINE_LEARNING)
    - 'What is Python?' -> 'Python / Programming' (Category: PYTHON_PROGRAMMING)
    - 'Compare SQL and DBMS.' -> 'Comparison / SQL vs DBMS' (Category: COMPARISON)
    """
    if not query:
        return {
            "topic": "General / Document Retrieval",
            "category": "GENERAL",
            "keywords": [],
            "is_comparison": False
        }

    clean = query.strip()
    lower = clean.lower()

    # 1. Comparison detection
    comp_res = detect_comparison_question(clean)
    if comp_res.is_comparison:
        topics_str = " vs ".join([t.title() for t in comp_res.topics]) if comp_res.topics else "Cross-Document"
        return {
            "topic": f"Comparison / {topics_str}",
            "category": "COMPARISON",
            "keywords": comp_res.topics,
            "is_comparison": True,
            "subtype": getattr(comp_res, "subtype", "COMPARISON_TABLE")
        }

    # 2. Software Engineering / Waterfall
    wf_triggers = ['waterfall', 'waterfall model', 'sdlc', 'software development lifecycle', 'sequential model']
    if any(re.search(rf'\b{re.escape(tr)}\b', lower) for tr in wf_triggers):
        return {
            "topic": "Software Engineering / Waterfall",
            "category": "SOFTWARE_ENGINEERING",
            "keywords": ['waterfall', 'model', 'sdlc', 'sequential', 'phases', 'software development'],
            "is_comparison": False
        }

    # 3. Security & Trust Controls / Key Escrow
    sec_triggers = ['key security and trust controls', 'security and trust controls', 'trust controls', 'security controls', 'key escrow', 'threshold cryptography', 'master decryption', 'dual authorization']
    if any(re.search(rf'\b{re.escape(tr)}\b', lower) for tr in sec_triggers) or (
        ('security' in lower or 'trust' in lower) and ('control' in lower or 'controls' in lower or 'escrow' in lower)
    ):
        return {
            "topic": "Security / Trust & Security Controls",
            "category": "SECURITY_CONTROLS",
            "keywords": ['security', 'trust controls', 'authentication', 'authorization', 'escrow', 'encryption'],
            "is_comparison": False
        }

    # 4. Project Summary / Conclusion
    conclusion_triggers = ['conclusion', 'conclusions', 'project conclusion', 'final conclusion', 'summary of project', 'future scope', 'project outcomes']
    if any(re.search(rf'\b{re.escape(tr)}\b', lower) for tr in conclusion_triggers):
        return {
            "topic": "Project Summary / Conclusion",
            "category": "CONCLUSION",
            "keywords": ['conclusion', 'summary', 'outcomes', 'results', 'future scope'],
            "is_comparison": False
        }

    # 5. IoT / Internet of Things
    iot_triggers = ['iot', 'internet of things', 'smart devices', 'sensor network', 'embedded systems']
    if any(re.search(rf'\b{re.escape(tr)}\b', lower) for tr in iot_triggers):
        return {
            "topic": "IoT / Internet of Things",
            "category": "IOT",
            "keywords": ['iot', 'internet of things', 'sensors', 'smart devices'],
            "is_comparison": False
        }

    # 6. Machine Learning / AI
    ml_triggers = ['machine learning', 'deep learning', 'neural network', 'neural networks', 'artificial intelligence', 'ml model', 'supervised learning', 'unsupervised learning']
    if any(re.search(rf'\b{re.escape(tr)}\b', lower) for tr in ml_triggers):
        return {
            "topic": "AI / Machine Learning",
            "category": "MACHINE_LEARNING",
            "keywords": ['machine learning', 'ai', 'model', 'training', 'algorithms'],
            "is_comparison": False
        }

    # 7. SQL Querying & DML / DDL Commands
    sql_cmd_triggers = [
        'querying data commands', 'querying data command', 'querying data', 'querying commands',
        'query commands', 'sql commands', 'sql command', 'data querying', 'dml commands',
        'dml', 'ddl commands', 'ddl', 'data manipulation language', 'data definition language',
        'select command', 'insert command', 'update command', 'delete command',
        'create command', 'alter command', 'drop command', 'truncate command'
    ]
    if any(re.search(rf'\b{re.escape(tr)}\b', lower) for tr in sql_cmd_triggers) or (
        ('sql' in lower) and any(cmd in lower for cmd in ['select', 'insert', 'update', 'delete', 'dml', 'ddl', 'command', 'commands'])
    ):
        return {
            "topic": "SQL / Data Querying / SQL Commands",
            "category": "SQL_COMMANDS",
            "keywords": ['select', 'insert', 'update', 'delete', 'dml', 'ddl', 'querying', 'sql'],
            "is_comparison": False
        }

    # 8. SQL JOIN
    join_triggers = [
        'join', 'joins', 'joining commands', 'joining command', 'joining', 'inner join',
        'left join', 'right join', 'full join', 'cross join', 'outer join', 'self join', 'natural join'
    ]
    if any(re.search(rf'\b{re.escape(tr)}\b', lower) for tr in join_triggers):
        return {
            "topic": "SQL / JOIN",
            "category": "SQL_JOIN",
            "keywords": ['join', 'joins', 'inner join', 'left join', 'right join', 'full join', 'cross join'],
            "is_comparison": False
        }

    # 9. DBMS ACID & Atomicity / Transactions
    acid_triggers = [
        'acid', 'acid properties', 'atomicity', 'consistency', 'isolation', 'durability',
        'transaction', 'transactions', 'commit', 'rollback', 'concurrency control', 'serializability'
    ]
    if any(re.search(rf'\b{re.escape(tr)}\b', lower) for tr in acid_triggers):
        topic_title = "DBMS / ACID / Atomicity" if 'atomicity' in lower else "DBMS / ACID & Transactions"
        return {
            "topic": topic_title,
            "category": "DBMS_ACID",
            "keywords": ['acid', 'atomicity', 'consistency', 'isolation', 'durability', 'transaction'],
            "is_comparison": False
        }

    # 10. DBMS Normalization
    norm_triggers = [
        'normalization', 'normal form', 'normal forms', '1nf', '2nf', '3nf', 'bcnf',
        'boyce codd', 'functional dependency', 'redundancy'
    ]
    if any(re.search(rf'\b{re.escape(tr)}\b', lower) for tr in norm_triggers):
        return {
            "topic": "DBMS / Normalization",
            "category": "DBMS_NORMALIZATION",
            "keywords": ['normalization', 'normal forms', '1nf', '2nf', '3nf', 'bcnf', 'redundancy'],
            "is_comparison": False
        }

    # 11. DBMS Keys
    keys_triggers = [
        'primary key', 'foreign key', 'candidate key', 'super key', 'unique key', 'composite key'
    ]
    if any(re.search(rf'\b{re.escape(tr)}\b', lower) for tr in keys_triggers):
        return {
            "topic": "DBMS / Keys & Constraints",
            "category": "DBMS_KEYS",
            "keywords": ['primary key', 'foreign key', 'candidate key', 'unique key'],
            "is_comparison": False
        }

    # 12. Python Programming
    py_triggers = [
        'python', 'list comprehension', 'decorator', 'gil', 'lambda function', 'generator', 'iterator'
    ]
    if any(re.search(rf'\b{re.escape(tr)}\b', lower) for tr in py_triggers) or (
        re.search(r'\b(list|tuple|dict|dictionary|set)\b', lower) and not any(db in lower for db in ['sql', 'table', 'dbms', 'key', 'security'])
    ):
        return {
            "topic": "Python / Programming",
            "category": "PYTHON_PROGRAMMING",
            "keywords": ['python', 'list', 'tuple', 'dict', 'set', 'generator'],
            "is_comparison": False
        }

    # 13. General entity extraction
    _, extracted_term = classify_query(clean)
    topic_label = f"General / {extracted_term.title()}" if extracted_term else "General / Document Retrieval"
    return {
        "topic": topic_label,
        "category": "GENERAL",
        "keywords": [extracted_term] if extracted_term else [],
        "is_comparison": False
    }


def check_chunk_topic_conflict(query_category: str, chunk_text: str) -> bool:
    """
    Returns True if the chunk text is clearly about an unrelated/conflicting concept
    and contains no relevant mentions for the query category.
    """
    if not query_category or query_category in ("GENERAL", "COMPARISON"):
        return False

    txt_lower = chunk_text.lower()

    if query_category == "SQL_COMMANDS":
        has_sql_cmd = any(w in txt_lower for w in ['select', 'insert', 'update', 'delete', 'dml', 'ddl', 'create', 'alter', 'drop', 'truncate', 'querying', 'query', 'sql commands'])
        has_only_unrelated = (
            any(w in txt_lower for w in ['acid', 'atomicity', 'durability', 'isolation', 'concurrency control', 'waterfall', 'key escrow', 'threshold cryptography'])
            and not has_sql_cmd
        )
        return has_only_unrelated

    if query_category == "SQL_JOIN":
        has_join = any(w in txt_lower for w in ['join', 'joins', 'inner join', 'left join', 'right join', 'full join', 'cross join'])
        has_only_unrelated = (
            any(w in txt_lower for w in ['normalization', '1nf', '2nf', '3nf', 'waterfall', 'key escrow', 'python lists'])
            and not has_join
        )
        return has_only_unrelated

    if query_category == "DBMS_ACID":
        has_acid = any(w in txt_lower for w in ['acid', 'atomicity', 'consistency', 'isolation', 'durability', 'transaction', 'commit', 'rollback'])
        has_only_unrelated = (
            any(w in txt_lower for w in ['waterfall', 'key escrow', 'python lists', 'tuples', 'joins'])
            and not has_acid
        )
        return has_only_unrelated

    if query_category == "DBMS_NORMALIZATION":
        has_norm = any(w in txt_lower for w in ['normalization', 'normal form', '1nf', '2nf', '3nf', 'bcnf', 'redundancy', 'functional dependency'])
        has_only_unrelated = (
            any(w in txt_lower for w in ['waterfall', 'key escrow', 'python lists', 'inner join'])
            and not has_norm
        )
        return has_only_unrelated

    if query_category == "PYTHON_PROGRAMMING":
        has_py = any(w in txt_lower for w in ['python', 'list', 'tuple', 'dict', 'dictionary', 'set', 'generator', 'def ', 'class '])
        has_only_unrelated = (
            any(w in txt_lower for w in ['sql commands', 'normalization', 'waterfall model', 'key escrow'])
            and not has_py
        )
        return has_only_unrelated

    return False


class ComparisonDetectionResult(tuple):
    """
    Subclass of tuple (is_comparison, topics) allowing backward-compatible 2-tuple unpacking
    while also exposing `subtype`, `is_comparison`, and `topics` attributes.
    """
    def __new__(cls, is_comparison: bool, topics: List[str], subtype: str = "COMPARISON_TABLE"):
        return super(ComparisonDetectionResult, cls).__new__(cls, (is_comparison, topics))

    def __init__(self, is_comparison: bool, topics: List[str], subtype: str = "COMPARISON_TABLE"):
        self.is_comparison = is_comparison
        self.topics = topics
        self.subtype = subtype


def detect_comparison_question(query: str) -> ComparisonDetectionResult:
    """
    Detects if a user query asks for comparison, similarities, differences, common info,
    document-wise summary, or cross-document synthesis.
    
    Returns:
        ComparisonDetectionResult(is_comparison, topics, subtype)
        Subtypes:
        - 'SUMMARY_ALL': Document-by-document summary of all/selected documents
        - 'COMMON_INFO': Information or concepts common across documents
        - 'SIMILARITIES': Similarities and shared properties
        - 'DIFFERENCES': Key differences and contrasting points
        - 'COMPARISON_TABLE': Comprehensive structured comparison table
    """
    if not query:
        return ComparisonDetectionResult(False, [], "")

    clean = query.strip()
    lower = clean.lower()

    # 1. Multi-document Summary requests
    summary_all_patterns = [
        r'^(?:can you\s+)?(?:please\s+)?summarize\s+(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+|each\s+)?(?:documents|pdfs|files)(?:\s+in\s+detail)?[\?!.,;]?$',
        r'^(?:can you\s+)?(?:please\s+)?summarize\s+(?:the\s+)?key\s+points\s+(?:from|across|in)\s+(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+)?(?:documents|pdfs|files)[\?!.,;]?$',
        r'^(?:give\s+me\s+a\s+|provide\s+a\s+|generate\s+a\s+)?summary\s+(?:of|from|across)\s+(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+)?(?:documents|pdfs|files)[\?!.,;]?$',
        r'^summarize\s+all(?:\s+documents)?[\?!.,;]?$'
    ]
    for pat in summary_all_patterns:
        if re.match(pat, clean, re.IGNORECASE):
            return ComparisonDetectionResult(True, [], "SUMMARY_ALL")

    # 2. Common Information across documents
    common_info_patterns = [
        r'^(?:what\s+(?:information\s+)?is|which\s+concepts\s+are|what\s+concepts\s+are|what\s+is)\s+common\s+(?:across|in|between|among)\s+(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+)?(?:documents|pdfs|files)[\?!.,;]?$',
        r'^(?:what\s+information\s+is\s+)?common\s+across\s+(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+)?(?:documents|pdfs|files)[\?!.,;]?$',
        r'^(?:what\s+are\s+the\s+)?common\s+(?:points|concepts|themes|features|elements|information)\s+(?:across|in|between)\s+(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+)?(?:documents|pdfs|files)[\?!.,;]?$'
    ]
    for pat in common_info_patterns:
        if re.match(pat, clean, re.IGNORECASE):
            return ComparisonDetectionResult(True, [], "COMMON_INFO")

    # 3. Similarities between documents
    similarity_patterns = [
        r'^(?:what\s+are\s+the\s+|explain\s+the\s+|show\s+the\s+|tell\s+me\s+the\s+)?similarities\s+(?:between|across|in|among)\s+(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+)?(?:documents|pdfs|files)[\?!.,;]?$',
        r'^(?:what\s+is\s+|how\s+are\s+they\s+)?similar\s+(?:between|across|in|among)\s+(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+)?(?:documents|pdfs|files)[\?!.,;]?$',
        r'^(?:how\s+are\s+)(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+)?(?:documents|pdfs|files)\s+similar[\?!.,;]?$'
    ]
    for pat in similarity_patterns:
        if re.match(pat, clean, re.IGNORECASE):
            return ComparisonDetectionResult(True, [], "SIMILARITIES")

    # 4. Differences between documents
    difference_patterns = [
        r'^(?:what\s+are\s+the\s+|explain\s+the\s+|show\s+the\s+|tell\s+me\s+the\s+)?(?:key\s+)?differences\s+(?:between|across|in|among)\s+(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+)?(?:documents|pdfs|files)[\?!.,;]?$',
        r'^(?:how\s+do\s+)(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+)?(?:documents|pdfs|files)\s+differ[\?!.,;]?$',
        r'^(?:what\s+is\s+the\s+)?contrast\s+(?:between|across|in)\s+(?:all\s+(?:the\s+)?(?:selected\s+)?|the\s+selected\s+|these\s+|the\s+)?(?:documents|pdfs|files)[\?!.,;]?$'
    ]
    for pat in difference_patterns:
        if re.match(pat, clean, re.IGNORECASE):
            return ComparisonDetectionResult(True, [], "DIFFERENCES")

    # 5. Multi-term explicit comparison patterns (e.g. "Compare X and Y", "Difference between X and Y", "X vs Y")
    explicit_patterns = [
        # "Compare X and Y", "Compare X with Y", "Compare X vs Y", "Compare X to Y"
        (r'^(?:can you\s+)?(?:please\s+)?(?:compare|contrast|comparison of|compare between)\s+(.+?)\s+(?:and|with|versus|vs\.?|to)\s+(.+?)[\?!.,;]?$', "COMPARISON_TABLE"),
        # "Difference between X and Y", "Differences between X and Y", "What is the difference between X and Y"
        (r'^(?:what is the|what are the|explain the|show the|tell me the)?\s*(?:difference|differences|difference between|different between)\s+(?:between\s+)?(.+?)\s+(?:and|versus|vs\.?|with)\s+(.+?)[\?!.,;]?$', "DIFFERENCES"),
        # "Similarities between X and Y"
        (r'^(?:what are the|what is the|explain the|show the)?\s*(?:similarities|similarity)\s+(?:between|of)\s+(.+?)\s+(?:and|with)\s+(.+?)[\?!.,;]?$', "SIMILARITIES"),
        # "X vs Y", "X versus Y"
        (r'^([a-zA-Z0-9_\s\-]+)\s+(?:vs\.?|versus)\s+([a-zA-Z0-9_\s\-]+)[\?!.,;]?$', "COMPARISON_TABLE"),
        # "Compare concepts in X and Y", "Compare topics between X and Y"
        (r'^(?:compare|contrast)\s+(?:the\s+)?(?:concepts|topics|details|features)?\s*(?:in|between|of)\s+(.+?)\s+(?:and|with)\s+(.+?)[\?!.,;]?$', "COMPARISON_TABLE"),
        # "Advantages and disadvantages of X and Y"
        (r'^(?:what are the\s+)?(?:advantages and disadvantages|pros and cons|tradeoffs)\s+(?:of|between)\s+(.+?)\s+(?:and|vs\.?|with)\s+(.+?)[\?!.,;]?$', "COMPARISON_TABLE")
    ]

    for pat, stype in explicit_patterns:
        m = re.match(pat, clean, re.IGNORECASE)
        if m:
            topic1 = m.group(1).strip()
            topic2 = m.group(2).strip()
            topic1 = re.sub(r'^(the|a|an)\s+', '', topic1, flags=re.IGNORECASE).strip()
            topic2 = re.sub(r'^(the|a|an)\s+', '', topic2, flags=re.IGNORECASE).strip()
            topic1 = re.sub(r'[?!.,;:]+$', '', topic1).strip()
            topic2 = re.sub(r'[?!.,;:]+$', '', topic2).strip()
            if topic1 and topic2:
                return ComparisonDetectionResult(True, [topic1, topic2], stype)

    # 6. Check for "Which document explains/contains X?"
    which_doc_match = re.match(
        r'^(?:which|what)\s+document\s+(?:explains|discusses|contains|has|describes|details|covers)\s+(.+?)[\?!.,;]?$',
        clean,
        re.IGNORECASE
    )
    if which_doc_match:
        target_concept = which_doc_match.group(1).strip()
        target_concept = re.sub(r'^(the|a|an)\s+', '', target_concept, flags=re.IGNORECASE).strip()
        target_concept = re.sub(r'[?!.,;:]+$', '', target_concept).strip()
        return ComparisonDetectionResult(True, [target_concept] if target_concept else [], "COMPARISON_TABLE")

    # 7. Check for general multi-document comparison triggers (e.g. "Compare the security approaches in these documents")
    general_comparison_triggers = [
        ('compare the concepts', "COMPARISON_TABLE"),
        ('compare concepts', "COMPARISON_TABLE"),
        ('compare these documents', "COMPARISON_TABLE"),
        ('compare all documents', "COMPARISON_TABLE"),
        ('compare the documents', "COMPARISON_TABLE"),
        ('compare all selected documents', "COMPARISON_TABLE"),
        ('compare selected documents', "COMPARISON_TABLE"),
        ('common points across', "COMMON_INFO"),
        ('common points in', "COMMON_INFO"),
        ('similarities between these', "SIMILARITIES"),
        ('similarities across', "SIMILARITIES"),
        ('across all selected documents', "COMPARISON_TABLE"),
        ('across all documents', "COMPARISON_TABLE"),
        ('across documents', "COMPARISON_TABLE"),
        ('in all selected documents', "COMPARISON_TABLE"),
        ('in these documents', "COMPARISON_TABLE"),
        ('synthesize these documents', "COMPARISON_TABLE"),
        ('summarize the common points', "COMMON_INFO"),
        ('common themes', "COMMON_INFO"),
        ('contrast between', "DIFFERENCES"),
        ('differences across', "DIFFERENCES"),
        ('similarities between', "SIMILARITIES")
    ]
    for trigger, stype in general_comparison_triggers:
        if trigger in lower:
            # Extract any specific aspect being compared (e.g. "security approaches" from "Compare the security approaches in these documents")
            aspect_match = re.search(r'compare\s+(?:the\s+)?(.+?)\s+(?:in|between|across)\s+(?:these|the|all)', clean, re.IGNORECASE)
            topics = [aspect_match.group(1).strip()] if aspect_match else []
            return ComparisonDetectionResult(True, topics, stype)

    # 8. Check for standalone comparison triggers like "Compare security approaches", "Compare normalization concepts"
    short_compare_match = re.match(r'^(?:compare|contrast|comparison of)\s+(.+?)[\?!.,;]?$', clean, re.IGNORECASE)
    if short_compare_match:
        topic = short_compare_match.group(1).strip()
        topic = re.sub(r'^(the|a|an)\s+', '', topic, flags=re.IGNORECASE).strip()
        topic = re.sub(r'[?!.,;:]+$', '', topic).strip()
        if topic and topic.lower() not in DISALLOWED_TOPICS:
            return ComparisonDetectionResult(True, [topic], "COMPARISON_TABLE")

    # 9. Check if query has comparison words alongside multiple technical entities
    comp_keywords = ['compare', 'comparison', 'versus', 'vs', 'contrast', 'difference', 'differences', 'similarities', 'similarity']
    if any(re.search(rf'\b{re.escape(k)}\b', lower) for k in comp_keywords):
        matched_kws = [kw for kw in TECHNICAL_KEYWORDS if re.search(rf'\b{re.escape(kw)}\b', lower)]
        stype = "DIFFERENCES" if "differen" in lower else ("SIMILARITIES" if "similar" in lower else "COMPARISON_TABLE")
        if len(matched_kws) >= 2:
            return ComparisonDetectionResult(True, matched_kws[:2], stype)
        return ComparisonDetectionResult(True, [], stype)

    return ComparisonDetectionResult(False, [], "")



def normalize_query(query: str) -> str:
    """
    Normalizes user question for retrieval while preserving vital technical terms.
    """
    if not query:
        return ""

    q = query.strip()

    # Conversational prefixes removal
    prefix_patterns = [
        r'^(what is the meaning of|what is meant by|what do you mean by)\s+',
        r'^(what is|what are|what was|what were|explain|tell me about|can you explain)\s+',
        r'^(describe|give me|how does|how do|list the|list of|definition of|details about)\s+',
        r'^(difference between|compare|what is the difference between)\s+'
    ]
    for pattern in prefix_patterns:
        q = re.sub(pattern, '', q, flags=re.IGNORECASE).strip()

    # Strip trailing punctuation
    q = re.sub(r'[?!.,;:]+$', '', q).strip()

    # Normalize hyphens to spaces
    q_norm = q.replace("-", " ")
    
    # Check if normalized version has a direct mapping
    q_lower = q_norm.lower().strip()
    if q_lower in TERM_NORMALIZATION_MAP:
        return TERM_NORMALIZATION_MAP[q_lower]

    return q


DISALLOWED_TOPICS = {
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'last', 'final',
    'the first', 'the second', 'the third', 'the fourth', 'the fifth', 'the sixth', 'the last', 'the final',
    'first property', 'second property', 'third property', 'fourth property', 'last property', 'final property',
    'the first property', 'the second property', 'the third property', 'the fourth property', 'the last property', 'the final property',
    'first type', 'second type', 'third type', 'fourth type', 'last type', 'final type',
    'the first type', 'the second type', 'the third type', 'the fourth type', 'the last type', 'the final type',
    'first point', 'second point', 'third point', 'fourth point', 'last point',
    'first one', 'second one', 'third one', 'fourth one', 'last one',
    'it', 'its', 'this', 'that', 'these', 'those', 'them', 'they', 'example', 'examples',
    'give an example', 'give example', 'an example', 'give an example of it', 'give me an example',
    'show example', 'give an example of this', 'example of it', 'example of this',
    'what about it', 'what about the last', 'what about the last property', 'what about the third',
    'what about the second', 'what about the first',
    'more', 'details', 'simple words', 'previous', 'above', 'type', 'types', 'property', 'properties',
    'it simply', 'it in simple words'
}


def extract_topic_from_text(text: str) -> str:
    """
    Extracts the core technical subject / topic entity from a message or query.
    Ignores generic follow-up phrases and ordinals.
    """
    if not text:
        return ""

    clean = text.strip()
    lower = clean.lower()

    # 1. Check technical keywords match (multi-word first, then single word)
    sorted_keywords = sorted(TECHNICAL_KEYWORDS, key=len, reverse=True)
    for kw in sorted_keywords:
        if re.search(rf'\b{re.escape(kw)}\b', lower):
            return kw

    # 2. Check definition patterns with explicit whitespace around articles
    def_patterns = [
        (r'^(what is the meaning of|what is meant by|what do you mean by)\s+(.+?)[\?!.,;]?$', 2),
        (r'^(what is|what are|what was|what were)\s+(.+?)[\?!.,;]?$', 2),
        (r'^(define|explain|describe|tell me about)\s+(.+?)[\?!.,;]?$', 2),
        (r'^([a-zA-Z0-9_\s\-]+)\s+(definition|meaning|stands for)[\?!.,;]?$', 1),
    ]
    for pattern, group_idx in def_patterns:
        m = re.match(pattern, clean, flags=re.IGNORECASE)
        if m:
            extracted = m.group(group_idx).strip()
            extracted = re.sub(r'[?!.,;:]+$', '', extracted).strip()
            clean_extracted = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', extracted, flags=re.IGNORECASE).strip().lower()
            if clean_extracted not in DISALLOWED_TOPICS and extracted.lower() not in DISALLOWED_TOPICS and len(clean_extracted.split()) <= 4:
                return clean_extracted

    # 3. Fallback to normalized query if short and not composed purely of generic action words
    GENERIC_WORDS = {
        'give', 'show', 'example', 'examples', 'it', 'this', 'that', 'them', 'these', 'those',
        'me', 'an', 'a', 'the', 'of', 'in', 'for', 'to', 'please', 'tell', 'explain', 'describe',
        'what', 'is', 'are', 'about', 'last', 'first', 'second', 'third', 'fourth', 'property',
        'type', 'types', 'properties', 'point', 'points', 'one', 'more', 'details', 'detail', 'simply'
    }
    norm = normalize_query(clean)
    if norm:
        clean_norm = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', norm, flags=re.IGNORECASE).strip().lower()
        words_in_norm = set(clean_norm.split())
        if clean_norm not in DISALLOWED_TOPICS and norm.lower() not in DISALLOWED_TOPICS and len(clean_norm.split()) <= 3:
            if not words_in_norm.issubset(GENERIC_WORDS):
                return clean_norm

    return ""


def extract_nth_item_from_assistant(text: str, n: int = 1) -> str:
    """
    Extracts the n-th item, property, type, or command from previous assistant answer.
    Supports comma-separated lists, numbered lists (1. Item), bullet points (- Item),
    bold headers (**Item:** or **Item**), and domain-specific concepts.
    n=-1 extracts the last item.
    """
    if not text:
        return ""

    # 1. Check for ACID specifically: Atomicity, Consistency, Isolation, Durability
    if "atomicity" in text.lower() and "consistency" in text.lower() and "durability" in text.lower():
        acid_items = ["Atomicity", "Consistency", "Isolation", "Durability"]
        if n == -1 or n == len(acid_items):
            return acid_items[-1]
        elif 1 <= n <= len(acid_items):
            return acid_items[n - 1]

    # 2. Check for DML commands in text: SELECT, INSERT, UPDATE, DELETE
    if "dml" in text.lower() or "manipulat" in text.lower() or ("insert" in text.lower() and "update" in text.lower()):
        dml_candidates = ["select", "insert", "update", "delete"]
        found_dml = []
        text_lower = text.lower()
        # Find order in text
        pos_list = []
        for cmd in dml_candidates:
            pos = text_lower.find(cmd)
            if pos != -1:
                pos_list.append((pos, cmd.upper()))
        pos_list.sort(key=lambda x: x[0])
        found_dml = [p[1] for p in pos_list]
        if not found_dml:
            found_dml = ["SELECT", "INSERT", "UPDATE", "DELETE"]
        if n == -1:
            return found_dml[-1]
        elif 1 <= n <= len(found_dml):
            return found_dml[n - 1]

    # 3. Check for DDL commands in text: CREATE, ALTER, DROP, TRUNCATE
    if "ddl" in text.lower() or "definition" in text.lower() or ("create" in text.lower() and "alter" in text.lower()):
        ddl_candidates = ["create", "alter", "drop", "truncate"]
        found_ddl = []
        text_lower = text.lower()
        pos_list = []
        for cmd in ddl_candidates:
            pos = text_lower.find(cmd)
            if pos != -1:
                pos_list.append((pos, cmd.upper()))
        pos_list.sort(key=lambda x: x[0])
        found_ddl = [p[1] for p in pos_list]
        if not found_ddl:
            found_ddl = ["CREATE", "ALTER", "DROP", "TRUNCATE"]
        if n == -1:
            return found_ddl[-1]
        elif 1 <= n <= len(found_ddl):
            return found_ddl[n - 1]

    # 4. Check for JOIN types specifically: INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN, CROSS JOIN, SELF JOIN, NATURAL JOIN
    join_keywords = ["inner join", "left join", "right join", "full join", "cross join", "self join", "natural join"]
    found_joins = []
    for jk in join_keywords:
        if jk in text.lower():
            label = jk.upper()
            if label not in found_joins:
                found_joins.append(label)
    if len(found_joins) >= 2:
        if n == -1:
            return found_joins[-1]
        elif 1 <= n <= len(found_joins):
            return found_joins[n - 1]

    # 5. Check for Normal Forms: 1NF, 2NF, 3NF, BCNF
    nf_keywords = ["1nf", "2nf", "3nf", "bcnf", "first normal form", "second normal form", "third normal form", "boyce-codd"]
    found_nfs = []
    for nf in nf_keywords:
        if nf in text.lower():
            label = nf.upper()
            if label not in found_nfs:
                found_nfs.append(label)
    if len(found_nfs) >= 2:
        if n == -1:
            return found_nfs[-1]
        elif 1 <= n <= len(found_nfs):
            return found_nfs[n - 1]

    # 6. Check numbered items: 1. Item, 1) Item, 1 - Item
    numbered_items = re.findall(r'(?:^|\n)\s*(?:\*\*)?(?:\d+[\.\)])\s*(?:\*\*)?([A-Za-z0-9_\s\(\)\-]+?)(?:\*\*)?(?::|\s*[-–—]|\s*\n|$)', text)
    valid_numbered = [b.strip() for b in numbered_items if len(b.strip().split()) <= 6 and len(b.strip()) > 1]
    if valid_numbered:
        if n == -1:
            return valid_numbered[-1]
        elif 1 <= n <= len(valid_numbered):
            return valid_numbered[n - 1]

    # 7. Check bullet items: - Item, • Item, * Item, - **Item**
    bullet_items = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s*(?:\*\*)?([A-Za-z0-9_\s\(\)\-]+?)(?:\*\*)?(?::|\s*[-–—]|\s*\n|$)', text)
    valid_bullets = [b.strip() for b in bullet_items if len(b.strip().split()) <= 6 and len(b.strip()) > 1]
    if valid_bullets:
        if n == -1:
            return valid_bullets[-1]
        elif 1 <= n <= len(valid_bullets):
            return valid_bullets[n - 1]

    # 8. Check comma/and separated lists: "A, B, C and D"
    list_match = re.search(r'(?:are|include|properties are|types are|forms are|commands are|consist of)\s+([A-Za-z0-9_\s,–—\-]+(?:and|or)\s+[A-Za-z0-9_\s\-]+)', text, re.IGNORECASE)
    if list_match:
        raw_list = list_match.group(1)
        raw_list = re.sub(r'\b(and|or)\b', ',', raw_list, flags=re.IGNORECASE)
        items = [i.strip() for i in raw_list.split(',') if i.strip() and len(i.strip().split()) <= 5]
        if items:
            if n == -1:
                return items[-1]
            elif 1 <= n <= len(items):
                return items[n - 1]

    return ""


# =============================================================================
# 2. CONVERSATION CONTEXT & QUERY REWRITING
# =============================================================================
def detect_followup_question(
    current_question: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    last_context: Optional[List[Any]] = None
) -> Tuple[bool, str, str, List[str]]:
    """
    Detects if the current question is a follow-up, clarification, example request, comparison follow-up,
    or standalone query based on conversational context and linguistic patterns.
    
    Returns:
        (is_followup, followup_type, context_topic, comparison_concepts)
    """
    if not current_question:
        return False, "STANDALONE", "", []

    clean = current_question.strip()
    lower = clean.lower()

    if not conversation_history and not last_context:
        return False, "STANDALONE", "", []

    # 1. Extract previous conversation context topics & comparison concepts
    prior_topic = ""
    comparison_concepts = []
    last_assistant_answer = ""
    last_subtopic = ""

    if conversation_history:
        recent_history = conversation_history[-10:]
        
        # Check if the most recent turn had comparison topics or answer
        for msg in reversed(recent_history):
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            
            # Extract comparison topics from previous turn if present
            if msg.get("comparison_topics"):
                comparison_concepts = msg.get("comparison_topics")
            elif role in ("user", "human"):
                is_comp, comp_topics = detect_comparison_question(content)
                if is_comp and comp_topics and not comparison_concepts:
                    comparison_concepts = comp_topics

            if role in ("assistant", "ai") and not last_assistant_answer and content:
                last_assistant_answer = content
                m_term = re.search(r'Term:\s*([^\n\r]+)', content)
                if m_term:
                    cand_sub = m_term.group(1).strip()
                    cand_sub = re.sub(r'[\*\(\)]+', '', cand_sub).strip()
                    if cand_sub.lower() not in DISALLOWED_TOPICS and len(cand_sub.split()) <= 4:
                        last_subtopic = cand_sub

            if role in ("user", "human") and content and not prior_topic:
                top = extract_topic_from_text(content)
                if top and top.lower() not in DISALLOWED_TOPICS:
                    prior_topic = top

    # Also inspect last_context if prior_topic not found
    if not prior_topic and last_context:
        for ctx in last_context:
            text = ctx.get("text", "") if isinstance(ctx, dict) else (getattr(ctx, "page_content", "") or str(ctx))
            top = extract_topic_from_text(text[:200])
            if top and top.lower() not in DISALLOWED_TOPICS:
                prior_topic = top
                break

    # Determine context topic label
    effective_topic = ""
    if comparison_concepts and len(comparison_concepts) >= 2:
        effective_topic = " and ".join(comparison_concepts)
    elif last_subtopic and last_subtopic.lower() != prior_topic.lower():
        effective_topic = f"{last_subtopic} ({prior_topic})" if prior_topic else last_subtopic
    elif prior_topic:
        effective_topic = prior_topic

    # 2. Check for Follow-Up Patterns
    pronoun_pattern = re.compile(
        r'\b(it|its|their|they|these|this|that|those|them|itself|previous|above|below|same|'
        r'the first|first one|first property|first type|first point|'
        r'the second|second one|second property|second type|second point|'
        r'the third|third one|third property|third type|third point|'
        r'the fourth|fourth one|fourth property|fourth type|fourth point|'
        r'the last|last one|last property|last type|last point|final one|final property|'
        r'which one|which is used|which of these|both of them|between them)\b',
        re.IGNORECASE
    )
    has_pronoun = bool(pronoun_pattern.search(clean))

    # Example request
    if re.search(r'\b(give (me )?(an )?example|give another example|give example|show (me )?(an )?example|syntax|code example|example\?|can you give an example)\b', lower):
        return True, "EXAMPLE_REQUEST", effective_topic or "Previous Concept", comparison_concepts

    # Clarification / Simplification
    if re.search(r'\b(in simple words|simple words|simply|simplify it|simplify|simple explanation|what does that mean|explain this|explain it simply|explain it|tell me more|explain more|more details|elaborate)\b', lower):
        return True, "CLARIFICATION", effective_topic or "Previous Concept", comparison_concepts

    # Summary request
    if re.search(r'\b(summarize it|summarize them|give a summary|summary of this|summary of it|summarize the common points)\b', lower):
        return True, "SUMMARY_REQUEST", effective_topic or "Previous Concept", comparison_concepts

    # Comparison follow-up
    if comparison_concepts and (has_pronoun or re.search(r'\b(similarities|differences|advantages|disadvantages|pros and cons|which one|which is used|compare them|between them|common points)\b', lower)):
        return True, "COMPARISON_FOLLOWUP", effective_topic or "Comparison Concepts", comparison_concepts

    # "What about X?" switch pattern
    what_about_match = re.match(r'^(?:what about|how about)\s+(.+?)[\?!.,;]?$', clean, re.IGNORECASE)
    if what_about_match:
        target = what_about_match.group(1).strip()
        if target.lower() in {'it', 'this', 'that', 'them', 'these', 'those', 'its advantages', 'the last', 'the first', 'the second', 'the third'}:
            return True, "FOLLOWUP", effective_topic or "Previous Concept", comparison_concepts
        else:
            # Topic switch to a specific named entity (e.g. "What about DDL?")
            return False, "STANDALONE", "", []

    # Standalone question check: If query specifies its own concept/definition, it is NOT a follow-up
    standalone_patterns = [
        r'^(what is|what are|what was|what were|define|explain|describe|tell me about)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s\-]+)[\?!.,;]?$',
        r'^(what is the meaning of|what is meant by|what do you mean by)\s+([a-zA-Z0-9_\s\-]+)[\?!.,;]?$',
        r'^([a-zA-Z0-9_\s\-]+)\s+(definition|meaning|stands for)[\?!.,;]?$',
        r'^(what does|how does)\s+([a-zA-Z0-9_\s\-]+)\s+(work|mean|do)[\?!.,;]?$'
    ]
    for sp in standalone_patterns:
        m = re.match(sp, clean, flags=re.IGNORECASE)
        if m:
            concept_candidate = m.group(m.lastindex).strip().lower()
            if concept_candidate not in DISALLOWED_TOPICS and not has_pronoun:
                return False, "STANDALONE", "", []

    # Explicit subtopic follow-up request (e.g. "Explain atomicity.") ONLY when prior was ACID
    explain_sub_match = re.match(r'^(?:explain|describe|what is|tell me about)\s+([a-zA-Z0-9_\s\-]+)[\?!.,;]?$', clean, re.IGNORECASE)
    if explain_sub_match and prior_topic:
        sub = explain_sub_match.group(1).strip()
        sub_lower = sub.lower()
        if sub_lower in {'atomicity', 'consistency', 'isolation', 'durability'} and 'acid' in prior_topic.lower():
            return True, "FOLLOWUP", f"{sub.title()} (ACID)", comparison_concepts
        if sub_lower in {'select', 'insert', 'update', 'delete'} and 'dml' in prior_topic.lower():
            return True, "FOLLOWUP", f"{sub.upper()} (DML)", comparison_concepts
        if sub_lower in {'create', 'alter', 'drop', 'truncate'} and 'ddl' in prior_topic.lower():
            return True, "FOLLOWUP", f"{sub.upper()} (DDL)", comparison_concepts

    # General Follow-up triggers with explicit pronouns/references
    if has_pronoun or re.search(r'\b(why is (it|that|this)|its types|its advantages|its disadvantages|its features)\b', lower):
        if effective_topic:
            return True, "FOLLOWUP", effective_topic, comparison_concepts
        return True, "FOLLOWUP", "Previous Concept", comparison_concepts

    return False, "STANDALONE", "", []


def classify_user_query(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    last_context: Optional[List[Any]] = None
) -> Tuple[str, bool, str, List[str]]:
    """
    Classifies the user query into exactly one of the 5 canonical RAG query categories:
    1. NEW QUESTION (Fresh retrieval, zero memory contamination)
    2. FOLLOW-UP QUESTION (Conversational follow-up with pronouns / ellipsis)
    3. CLARIFICATION (Request to simplify or give examples of previous topic)
    4. COMPARISON (Request to compare 2+ concepts or documents)
    5. SUMMARY REQUEST (Request to summarize documents or previous points)

    Returns:
        (category_name, is_followup, context_topic, comparison_concepts)
    """
    if not query or not query.strip():
        return "NEW QUESTION", False, "", []

    clean = query.strip()
    lower = clean.lower()

    # 1. Comparison check
    comp_res = detect_comparison_question(clean)
    if comp_res.is_comparison:
        return "COMPARISON", False, "", comp_res.topics

    # 2. Summary check
    if re.search(r'\b(summarize (all|the|both|these|it|them|documents|the document|selected documents)?|give a summary|summary of (this|it|the document|the project))\b', lower):
        if conversation_history or last_context:
            is_fol, fol_type, ctx_top, comp_c = detect_followup_question(clean, conversation_history, last_context)
            if is_fol:
                return "SUMMARY REQUEST", True, ctx_top, comp_c
        return "SUMMARY REQUEST", False, "", []

    # 3. Follow-up / Clarification check
    if conversation_history or last_context:
        is_fol, fol_type, ctx_top, comp_c = detect_followup_question(clean, conversation_history, last_context)
        if is_fol:
            if fol_type in ("CLARIFICATION", "EXAMPLE_REQUEST"):
                return "CLARIFICATION", True, ctx_top, comp_c
            elif fol_type == "SUMMARY_REQUEST":
                return "SUMMARY REQUEST", True, ctx_top, comp_c
            elif fol_type == "COMPARISON_FOLLOWUP":
                return "COMPARISON", True, ctx_top, comp_c
            else:
                return "FOLLOW-UP QUESTION", True, ctx_top, comp_c

    # 4. Default is NEW QUESTION (Guarantees fresh retrieval & zero answer contamination)
    return "NEW QUESTION", False, "", []


def rewrite_query(
    current_question: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    last_context: Optional[List[Any]] = None
) -> Tuple[str, str]:
    """
    Robust Conversation-Aware Query Rewriting:
    - Inspects recent conversation turns and last retrieved context.
    - Resolves pronouns, ellipsis, example requests, list references, and comparison follow-ups into a standalone search query.
    - Strict isolation: Standalone questions ALWAYS return the original query with empty prior_topic.
    
    Returns:
        (rewritten_query, prior_topic)
    """
    if not current_question:
        return "", ""

    clean_curr = current_question.strip()
    if not conversation_history and not last_context:
        return clean_curr, ""

    curr_lower = clean_curr.lower()

    # Step A: Follow-up detection
    is_followup, followup_type, context_topic, comp_concepts = detect_followup_question(
        clean_curr, conversation_history, last_context
    )

    if not is_followup:
        # Crucial Isolation Fix: Standalone queries NEVER inherit prior topics
        return clean_curr, ""

    # Step B: Extract prior topic and comparison concepts from conversation history
    prior_topic = ""
    last_assistant_answer = ""
    last_subtopic = ""

    if conversation_history:
        recent_history = conversation_history[-10:]
        
        for msg in reversed(recent_history):
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if role in ("assistant", "ai") and not last_assistant_answer and content:
                last_assistant_answer = content
                m_term = re.search(r'Term:\s*([^\n\r]+)', content)
                if m_term:
                    cand_sub = m_term.group(1).strip()
                    cand_sub = re.sub(r'[\*\(\)]+', '', cand_sub).strip()
                    if cand_sub.lower() not in DISALLOWED_TOPICS and len(cand_sub.split()) <= 4:
                        last_subtopic = cand_sub

            if role in ("user", "human") and content and content.lower() != curr_lower and not prior_topic:
                topic = extract_topic_from_text(content)
                if topic and topic.lower() not in DISALLOWED_TOPICS:
                    prior_topic = topic

    if not prior_topic and last_context:
        for ctx in last_context:
            text = ctx.get("text", "") if isinstance(ctx, dict) else (getattr(ctx, "page_content", "") or str(ctx))
            topic = extract_topic_from_text(text[:200])
            if topic and topic.lower() not in DISALLOWED_TOPICS:
                prior_topic = topic
                break

    # Step C: Check for Comparison Memory Follow-up (e.g. "What are their similarities?", "What about their advantages?")
    if comp_concepts and len(comp_concepts) >= 2:
        c1, c2 = comp_concepts[0], comp_concepts[1]
        if "similarit" in curr_lower or "common" in curr_lower:
            return f"What are the similarities and common points between {c1} and {c2}?", f"{c1} and {c2}"
        if "differen" in curr_lower or "distinguish" in curr_lower:
            return f"What is the difference between {c1} and {c2}?", f"{c1} and {c2}"
        if "advantage" in curr_lower or "benefit" in curr_lower or "feature" in curr_lower:
            return f"What are the advantages and features of {c1} and {c2}?", f"{c1} and {c2}"
        if "which one" in curr_lower or "which is used" in curr_lower:
            action_match = re.search(r'(?:which one|which)\s+(?:is used to|is used for|does|can)\s+(.+)', clean_curr, re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip().rstrip('?!.,')
                return f"Which one is used to {action}: {c1} or {c2}?", f"{c1} and {c2}"
            return f"{clean_curr} in comparison of {c1} and {c2}", f"{c1} and {c2}"

    # Step D: Check "What about [Entity]?" topic shift
    what_about_match = re.match(r'^(?:what about|how about)\s+(.+?)[\?!.,;]?$', clean_curr, re.IGNORECASE)
    if what_about_match:
        target = what_about_match.group(1).strip()
        if target.lower() not in {'it', 'this', 'that', 'them', 'these', 'those', 'its advantages', 'the last', 'the first', 'the second', 'the third', 'its types'}:
            return f"What is {target}?", target

    # Step E: Sub-concept follow-up resolution (e.g. "Explain atomicity" -> "Explain Atomicity in ACID properties")
    explain_sub_match = re.match(r'^(?:explain|describe|what is|tell me about)\s+([a-zA-Z0-9_\s\-]+)[\?!.,;]?$', clean_curr, re.IGNORECASE)
    if explain_sub_match and prior_topic:
        sub = explain_sub_match.group(1).strip()
        sub_lower = sub.lower()
        if sub_lower in {'atomicity', 'consistency', 'isolation', 'durability'} and 'acid' in prior_topic.lower():
            return f"Explain {sub.title()} in ACID properties and database transactions.", f"{sub.title()} (ACID)"
        if sub_lower in {'select', 'insert', 'update', 'delete'} and 'dml' in prior_topic.lower():
            return f"Explain {sub.upper()} command in DML commands.", f"{sub.upper()} (DML)"
        if sub_lower in {'create', 'alter', 'drop', 'truncate'} and 'ddl' in prior_topic.lower():
            return f"Explain {sub.upper()} command in DDL commands.", f"{sub.upper()} (DDL)"

    if not prior_topic:
        return clean_curr, ""

    # Step F: Apply intent-specific rewriting based on prior topic
    if followup_type == "EXAMPLE_REQUEST" or re.search(r'\b(example|syntax)\b', curr_lower):
        target_concept = last_subtopic if last_subtopic and last_subtopic.lower() != prior_topic.lower() else prior_topic
        if 'acid' in prior_topic.lower() or 'acid' in target_concept.lower():
            rewritten = f"Give an example of {target_concept} in ACID properties and database transactions."
        elif 'join' in prior_topic.lower() or 'join' in target_concept.lower():
            rewritten = f"Give an example and syntax of {target_concept} in SQL."
        elif 'dml' in prior_topic.lower() or 'dml' in target_concept.lower():
            rewritten = f"Give an example and syntax of {target_concept} in DML commands."
        elif 'ddl' in prior_topic.lower() or 'ddl' in target_concept.lower():
            rewritten = f"Give an example and syntax of {target_concept} in DDL commands."
        elif 'normalization' in prior_topic.lower() or 'normalization' in target_concept.lower():
            rewritten = f"Give an example of {target_concept} in DBMS."
        else:
            rewritten = f"Give an example of {target_concept}."
        return rewritten, prior_topic

    if followup_type == "CLARIFICATION" or re.search(r'\b(simple words|simply|simplify)\b', curr_lower):
        target_concept = last_subtopic if last_subtopic and last_subtopic.lower() != prior_topic.lower() else prior_topic
        return f"Explain {target_concept} in simple words and clear concepts", target_concept

    if re.search(r'\b(advantage|advantages|benefit|benefits|feature|features|disadvantage|disadvantages)\b', curr_lower):
        if "advantage" in curr_lower or "benefit" in curr_lower:
            return f"What are the advantages of {prior_topic}?", prior_topic
        elif "feature" in curr_lower:
            return f"What are the features of {prior_topic}?", prior_topic
        elif "disadvantage" in curr_lower:
            return f"What are the disadvantages of {prior_topic}?", prior_topic
        else:
            return f"{clean_curr} {prior_topic}", prior_topic

    # 6. Third item request: "Explain the third point/type/property"
    if re.search(r'\b(third property|third type|third one|3rd)\b', curr_lower):
        third_sub = extract_nth_item_from_assistant(last_assistant_answer, 3)
        if third_sub:
            return f"What is {third_sub} in {prior_topic}?", prior_topic
        elif 'acid' in prior_topic.lower():
            return "What is Isolation in ACID properties?", prior_topic
        elif 'dml' in prior_topic.lower():
            return "What is UPDATE in DML commands?", prior_topic
        return f"What is the third property of {prior_topic}?", prior_topic

    # 7. Second item request: "Explain the second type."
    if re.search(r'\b(second property|second type|second one|2nd)\b', curr_lower):
        second_sub = extract_nth_item_from_assistant(last_assistant_answer, 2)
        if second_sub:
            return f"What is {second_sub} in {prior_topic}?", prior_topic
        elif 'acid' in prior_topic.lower():
            return "What is Consistency in ACID properties?", prior_topic
        return f"Explain the second property and type of {prior_topic}", prior_topic

    # 8. First item request: "Explain the first point."
    if re.search(r'\b(first property|first type|first one|1st)\b', curr_lower):
        first_sub = extract_nth_item_from_assistant(last_assistant_answer, 1)
        if first_sub:
            return f"What is {first_sub} in {prior_topic}?", prior_topic
        elif 'acid' in prior_topic.lower():
            return "What is Atomicity in ACID properties?", prior_topic
        return f"Explain the first property of {prior_topic}", prior_topic

    # 9. General pronoun substitution
    if re.search(r'\b(it|its|their|they|these|this|that|those|them)\b', curr_lower):
        expanded = re.sub(r'\b(its|their|these|this|that|those)\b', f"the {prior_topic}'s", clean_curr, flags=re.IGNORECASE)
        expanded = re.sub(r'\b(it|them|they)\b', prior_topic, expanded, flags=re.IGNORECASE)
        return expanded, prior_topic

    # 10. Fallback append
    return f"{clean_curr} {prior_topic}", prior_topic


def validate_retrieved_context(
    query: str,
    candidate_chunks: List[Any],
    min_score: float = MIN_CHUNK_RELEVANCE
) -> Tuple[bool, str, List[Any]]:
    """
    Validates that retrieved candidate chunks genuinely contain relevant factual content
    for the user's current question before proceeding to LLM generation.
    Enforces that:
    1. Chunks below min_score are discarded.
    2. Core search terms/entities must match at least one chunk (or have high cross-encoder relevance).
    3. If no chunk contains the requested information, returns (False, reason, []).
    """
    if not candidate_chunks:
        return False, "No chunks retrieved", []

    stop_words = {
        'what', 'is', 'the', 'a', 'an', 'are', 'explain', 'describe', 'tell', 'me', 
        'about', 'how', 'does', 'do', 'in', 'of', 'for', 'to', 'and', 'with', 'by', 
        'its', 'their', 'model', 'concept', 'give', 'show', 'please', 'list', 'details',
        'can', 'you', 'definition'
    }
    tokens = set(re.findall(r'[a-zA-Z0-9_\-]+', query.lower()))
    keywords = [w for w in tokens if len(w) > 1 and w not in stop_words]
    stems = set(stem_token(w) for w in keywords)

    valid_chunks = []
    for item in candidate_chunks:
        if isinstance(item, (tuple, list)):
            doc = item[0]
            score = float(item[1]) if len(item) > 1 else 0.8
        elif isinstance(item, dict):
            doc = item.get("doc") or item
            score = float(item.get("reranker_score", item.get("final_score", item.get("score", 0.8))))
        else:
            doc = item
            score = 0.8

        text = getattr(doc, "page_content", "") if hasattr(doc, "page_content") else (item.get("text", "") if isinstance(item, dict) else str(doc))
        text_lower = text.lower()
        chunk_stems = set(stem_token(w) for w in re.findall(r'[a-zA-Z0-9_\-]+', text_lower))

        has_stem_match = bool(stems.intersection(chunk_stems)) if stems else True

        if score >= min_score and (has_stem_match or score >= 0.85):
            valid_chunks.append(item)

    if not valid_chunks:
        return False, f"Retrieved chunks do not contain required concepts for: {', '.join(keywords)}", []

    return True, "Valid", valid_chunks



# Backward-compatibility alias
resolve_followup_query = rewrite_query


# =============================================================================
# 3. HYBRID KEYWORD, CONTEXT & FAISS RETRIEVAL
# =============================================================================
def get_search_terms_and_synonyms(query: str, prior_topic: str = "") -> List[str]:
    """
    Extracts high-priority search tokens and domain component terms for keyword matching.
    """
    terms = set()
    combined_text = f"{query} {prior_topic}".lower()

    # Check for domain acronyms and multi-word terms
    for key, syn_list in TERM_SYNONYM_MAP.items():
        if key in combined_text:
            for s in syn_list:
                terms.add(s)

    # Add words from query and topic (filtering common stop words)
    stop_words = {
        'what', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'of', 'for',
        'to', 'and', 'or', 'by', 'with', 'about', 'give', 'me', 'explain', 'show',
        'tell', 'can', 'you', 'please', 'it', 'its', 'this', 'that', 'these', 'those',
        'point', 'type', 'property'
    }
    for word in re.findall(r'[a-zA-Z0-9_\-]+', combined_text):
        if len(word) > 2 and word not in stop_words:
            terms.add(word)

    return sorted(list(terms), key=len, reverse=True)


def search_previous_context(
    query: str,
    last_context: Optional[List[Any]] = None,
    selected_documents: Optional[List[str]] = None,
    prior_topic: str = ""
) -> List[Tuple[Document, float]]:
    """
    Performs context-first search in last retrieved context chunks.
    Filters strictly by selected_documents when provided.
    """
    if not last_context:
        return []

    selected_set = set(selected_documents) if selected_documents else None
    search_terms = get_search_terms_and_synonyms(query, prior_topic)
    scored_context = []

    for item in last_context:
        doc_name = ""
        page_num = 1
        text = ""

        if isinstance(item, dict):
            doc_name = item.get("document") or item.get("source", "Document")
            page_num = item.get("page", 1)
            text = item.get("text", "")
        elif hasattr(item, "page_content"):
            doc_name = item.metadata.get("document") or item.metadata.get("source", "Document")
            page_num = item.metadata.get("page", 1)
            text = item.page_content

        if not text:
            continue

        if selected_set is not None and doc_name not in selected_set:
            continue

        text_lower = text.lower()
        matched = sum(1 for t in search_terms if re.search(rf'\b{re.escape(t)}\b', text_lower, re.IGNORECASE))
        if matched > 0 or (prior_topic and prior_topic.lower() in text_lower):
            doc_obj = Document(
                page_content=text,
                metadata={"document": doc_name, "source": doc_name, "page": page_num, "chunk_id": "ctx"}
            )
            score = 0.85 + min(0.10, matched * 0.03)
            scored_context.append((doc_obj, score))

    return scored_context


def tokenize_bm25_text(text: str) -> List[str]:
    """Tokenizes text into normalized words for BM25 retrieval."""
    if not text:
        return []
    # Tokenize words/numbers and preserve technical acronyms
    tokens = re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25IndexCache:
    """
    Caches BM25Okapi instance across search calls to prevent re-indexing for every query.
    Rebuilds automatically if document chunk count changes or if explicitly cleared.
    """
    def __init__(self):
        self._doc_count: int = 0
        self._bm25: Optional[Any] = None
        self._docs: List[Document] = []

    def clear(self):
        """Clears the cached BM25 index."""
        self._doc_count = 0
        self._bm25 = None
        self._docs = []
        logger.info("BM25 index cache cleared.")

    def get_or_build(self, vectorstore: FAISS) -> Tuple[Optional[Any], List[Document]]:
        if not vectorstore or not hasattr(vectorstore, "docstore") or not hasattr(vectorstore.docstore, "_dict"):
            return None, []

        current_docs = list(vectorstore.docstore._dict.values())
        current_count = len(current_docs)

        if self._bm25 is not None and self._doc_count == current_count:
            return self._bm25, self._docs

        if not current_docs or not HAS_RANK_BM25:
            return None, current_docs

        tokenized_corpus = [tokenize_bm25_text(doc.page_content) for doc in current_docs]
        tokenized_corpus = [tokens if tokens else ["empty"] for tokens in tokenized_corpus]

        try:
            self._bm25 = BM25Okapi(tokenized_corpus)
            self._doc_count = current_count
            self._docs = current_docs
            return self._bm25, self._docs
        except Exception as err:
            logger.warning(f"Failed to build BM25Okapi index: {str(err)}")
            return None, current_docs


_BM25_INDEX_CACHE = BM25IndexCache()


def clear_bm25_cache():
    """Invalidates the in-memory BM25Okapi cache so it rebuilds when documents change."""
    _BM25_INDEX_CACHE.clear()


def build_bm25_index(chunks: Optional[List[Document]] = None, vectorstore: Optional[FAISS] = None) -> Optional[Any]:
    """
    Builds or retrieves the cached BM25Okapi index from document chunks or FAISS docstore.
    """
    if chunks:
        tokenized_corpus = [tokenize_bm25_text(doc.page_content if hasattr(doc, 'page_content') else str(doc)) for doc in chunks]
        tokenized_corpus = [tokens if tokens else ["empty"] for tokens in tokenized_corpus]
        if HAS_RANK_BM25:
            try:
                return BM25Okapi(tokenized_corpus)
            except Exception as err:
                logger.warning(f"Failed to build BM25Okapi index from chunks: {err}")
                return None
    elif vectorstore is not None:
        bm25_model, _ = _BM25_INDEX_CACHE.get_or_build(vectorstore)
        return bm25_model
    return None


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalizes a list of arbitrary scores into [0.0, 1.0].
    Handles max_score == min_score gracefully without division by zero.
    """
    if not scores:
        return []
    min_s = float(min(scores))
    max_s = float(max(scores))
    if math.isclose(max_s, min_s):
        return [1.0 if max_s > 0 else 0.0 for _ in scores]
    diff = max_s - min_s
    return [max(0.0, min(1.0, (float(s) - min_s) / diff)) for s in scores]


def search_vector(
    vectorstore: FAISS,
    query: str,
    top_k: int = VECTOR_TOP_K,
    selected_documents: Optional[List[str]] = None
) -> List[Tuple[Document, float]]:
    """
    Performs FAISS semantic vector search and converts distances to similarity scores (0.0 to 1.0).
    Filters strictly by selected_documents when specified.
    """
    if vectorstore is None or not query or not query.strip():
        return []

    selected_set = set(selected_documents) if selected_documents else None
    results = []
    
    fetch_k = top_k * 3 if selected_set is not None else top_k
    try:
        raw_faiss = vectorstore.similarity_search_with_score(query=query.strip(), k=fetch_k)
        for doc, distance in raw_faiss:
            d_name = doc.metadata.get("document") or doc.metadata.get("source", "Document")
            if selected_set is None or d_name in selected_set:
                sem_score = compute_semantic_score(float(distance))
                results.append((doc, sem_score))
                if len(results) >= top_k:
                    break
    except Exception as err:
        logger.warning(f"Error in search_vector: {err}")

    return results[:top_k]


def search_bm25(
    vectorstore: FAISS,
    query: str,
    top_k: int = BM25_TOP_K,
    selected_documents: Optional[List[str]] = None,
    prior_topic: str = ""
) -> Tuple[List[Tuple[Document, float]], List[str]]:
    """
    Performs BM25 keyword search using rank_bm25 (BM25Okapi) across indexed document chunks.
    Filters strictly by selected_documents when specified.
    """
    return bm25_keyword_search(
        vectorstore=vectorstore,
        query=query,
        prior_topic=prior_topic,
        top_k=top_k,
        selected_documents=selected_documents
    )


def bm25_keyword_search(
    vectorstore: FAISS,
    query: str,
    prior_topic: str = "",
    top_k: int = BM25_TOP_K,
    selected_documents: Optional[List[str]] = None
) -> Tuple[List[Tuple[Document, float]], List[str]]:
    """
    Performs BM25 keyword search using rank_bm25 (BM25Okapi).
    Filters strictly by selected_documents when provided.
    Preserves all chunk metadata (chunk_id, document_name, page_number, etc.).
    
    Returns:
        (scored_chunks, matched_term_names)
    """
    if not vectorstore or not hasattr(vectorstore, "docstore") or not hasattr(vectorstore.docstore, "_dict"):
        return [], []

    search_terms = get_search_terms_and_synonyms(query, prior_topic)
    if not search_terms:
        return [], []

    selected_set = set(selected_documents) if selected_documents else None
    matched_terms_overall = set()
    scored_candidates = []

    # 1. Try rank_bm25 if available
    bm25_model, corpus_docs = _BM25_INDEX_CACHE.get_or_build(vectorstore)

    if bm25_model is not None and corpus_docs:
        # Build query tokens combining query and domain synonyms
        query_tokens = tokenize_bm25_text(query)
        if prior_topic:
            query_tokens.extend(tokenize_bm25_text(prior_topic))
        for t in search_terms:
            query_tokens.extend(tokenize_bm25_text(t))
        query_tokens = list(dict.fromkeys(query_tokens))

        if query_tokens:
            try:
                raw_bm25_scores = bm25_model.get_scores(query_tokens)
                for doc, raw_score in zip(corpus_docs, raw_bm25_scores):
                    d_name = doc.metadata.get("document") or doc.metadata.get("source", "Document")
                    if selected_set is not None and d_name not in selected_set:
                        continue
                    if raw_score > 0.0:
                        content_lower = doc.page_content.lower()
                        for t in search_terms:
                            if re.search(rf'\b{re.escape(t)}\b', content_lower, re.IGNORECASE):
                                matched_terms_overall.add(t)
                        scored_candidates.append((doc, float(raw_score)))
            except Exception as err:
                logger.warning(f"BM25 scoring failed, falling back to term scanner: {str(err)}")
                scored_candidates = []

    # 2. Fallback term-frequency scanner if BM25 produced no hits or is unavailable
    if not scored_candidates:
        avg_chunk_words = 120.0
        k1 = 1.2
        b = 0.75

        for _doc_id, doc in vectorstore.docstore._dict.items():
            d_name = doc.metadata.get("document") or doc.metadata.get("source", "Document")
            if selected_set is not None and d_name not in selected_set:
                continue

            content = doc.page_content
            if not content:
                continue

            content_lower = content.lower()
            words = content.split()
            doc_len = max(10, len(words))
            lines = content.split('\n')
            top_lines = " ".join(lines[:2]).lower()

            matched_count = 0
            bm25_score_sum = 0.0
            exact_phrase_match = False
            heading_match = False

            clean_q_lower = query.lower().strip()
            if len(clean_q_lower) > 3 and clean_q_lower in content_lower:
                exact_phrase_match = True

            for t in search_terms:
                t_pattern = re.compile(rf'\b{re.escape(t)}\b', re.IGNORECASE)
                counts = len(t_pattern.findall(content_lower))
                if counts > 0:
                    matched_count += 1
                    matched_terms_overall.add(t)
                    tf = (counts * (k1 + 1.0)) / (counts + k1 * (1.0 - b + b * (doc_len / avg_chunk_words)))
                    term_weight = 1.0 if len(t) > 4 else 0.8
                    bm25_score_sum += tf * term_weight
                    if t_pattern.search(top_lines):
                        heading_match = True

            if matched_count > 0:
                term_overlap = matched_count / max(1, min(len(search_terms), 4))
                raw_kw_score = (0.50 * term_overlap) + min(0.35, bm25_score_sum * 0.12)
                if exact_phrase_match:
                    raw_kw_score += 0.15
                if heading_match:
                    raw_kw_score += 0.10
                scored_candidates.append((doc, raw_kw_score))

    # Normalize BM25 raw scores to [0.0, 1.0]
    if scored_candidates:
        raw_vals = [s[1] for s in scored_candidates]
        norm_vals = normalize_scores(raw_vals)
        normalized_candidates = [(doc, float(norm_val)) for (doc, _), norm_val in zip(scored_candidates, norm_vals)]
        normalized_candidates.sort(key=lambda x: x[1], reverse=True)
        return normalized_candidates[:top_k], sorted(list(matched_terms_overall))

    return [], []


def exact_keyword_search(
    vectorstore: FAISS,
    query: str,
    prior_topic: str = "",
    top_k: int = BM25_TOP_K,
    selected_documents: Optional[List[str]] = None
) -> Tuple[List[Tuple[Document, float]], List[str]]:
    """Backward-compatible wrapper for BM25 keyword search."""
    return bm25_keyword_search(
        vectorstore=vectorstore,
        query=query,
        prior_topic=prior_topic,
        top_k=top_k,
        selected_documents=selected_documents
    )


def compute_semantic_score(distance: float) -> float:
    """
    Converts FAISS L2 distance into cosine similarity score between 0.0 and 1.0.
    Since embeddings are normalized unit vectors: d^2 = 2 - 2*cos_sim => cos_sim = 1 - d^2/2.
    """
    d = max(0.0, float(distance))
    cos_sim = 1.0 - (d * d / 2.0)
    return max(0.0, min(1.0, cos_sim))


def compute_metadata_score(query: str, text: str, doc_name: str) -> float:
    """
    Computes metadata score (0.0 to 1.0) based on title match and domain keywords.
    """
    if not query or not text:
        return 0.0

    score = 0.0
    q_lower = query.lower().strip()
    doc_lower = doc_name.lower()

    lines = text.split('\n')
    top_lines = " ".join(lines[:2]).lower()

    q_words = [w for w in q_lower.split() if len(w) > 2]
    if any(w in top_lines for w in q_words):
        score += 0.40

    intro = text.lower()[:150]
    if any(w in intro for w in q_words):
        score += 0.30

    domain_terms = ['sql', 'join', 'dbms', 'acid', 'python', 'table', 'normalization', 'key']
    for dt in domain_terms:
        if dt in q_lower and dt in doc_lower:
            score += 0.30
            break

    return max(0.0, min(1.0, score))


def is_near_duplicate(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """
    Checks if two chunks are near-duplicates using word n-gram Jaccard similarity.
    """
    if not text1 or not text2:
        return False
    if text1.strip() == text2.strip():
        return True

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return False

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    jaccard = intersection / union if union > 0 else 0.0

    return jaccard >= threshold


# =============================================================================
# 4. HYBRID FUSION & RERANKING
from concurrent.futures import ThreadPoolExecutor

def hybrid_search(
    vectorstore: FAISS,
    query: str,
    selected_documents: Optional[List[str]] = None,
    vector_top_k: int = VECTOR_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    prior_topic: str = ""
) -> List[Dict[str, Any]]:
    """
    Performs Concurrent Hybrid Search combining FAISS vector search and BM25 keyword search:
    Executes FAISS vector search and BM25 search in parallel using ThreadPoolExecutor.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_vec = executor.submit(
            search_vector,
            vectorstore=vectorstore,
            query=query,
            top_k=vector_top_k,
            selected_documents=selected_documents
        )
        future_bm25 = executor.submit(
            search_bm25,
            vectorstore=vectorstore,
            query=query,
            top_k=bm25_top_k,
            selected_documents=selected_documents,
            prior_topic=prior_topic
        )

        vector_results = future_vec.result()
        bm25_results, _ = future_bm25.result()

    return fuse_search_results(
        semantic_results=vector_results,
        keyword_results=bm25_results,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        top_k=RERANK_CANDIDATES
    )


def fuse_search_results(
    semantic_results: List[Tuple[Document, float]],
    keyword_results: List[Tuple[Document, float]],
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    k: int = 60,
    top_k: int = RERANK_CANDIDATES
) -> List[Dict[str, Any]]:
    """
    Combines FAISS semantic and BM25 keyword search results with score normalization and weighted fusion:
    Formula:
        hybrid_score = (vector_weight * normalized_vector_score) + (bm25_weight * normalized_bm25_score)
    Preserves all chunk metadata: document_name, document_id, page_number, chunk_id, vector_score, bm25_score, etc.
    """
    raw_vec_scores = [float(s) for _, s in semantic_results] if semantic_results else []
    norm_vec_scores = normalize_scores(raw_vec_scores) if raw_vec_scores else []

    raw_bm25_scores = [float(s) for _, s in keyword_results] if keyword_results else []
    norm_bm25_scores = normalize_scores(raw_bm25_scores) if raw_bm25_scores else []

    doc_map: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    rrf_scores: Dict[Tuple[str, int, str], float] = {}

    # Ingest Vector candidates
    for rank, ((doc, raw_s), norm_s) in enumerate(zip(semantic_results, norm_vec_scores), start=1):
        d_name = doc.metadata.get("document") or doc.metadata.get("source", "Document")
        p_num = int(doc.metadata.get("page", 1))
        c_id = str(doc.metadata.get("chunk_id", ""))
        key = (d_name, p_num, c_id if c_id else doc.page_content[:60])

        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        if key not in doc_map:
            doc_map[key] = {
                "doc": doc,
                "document_name": d_name,
                "document_id": d_name,
                "page_number": p_num,
                "chunk_id": c_id or f"{d_name}_p{p_num}",
                "text": doc.page_content,
                "vector_score": round(float(raw_s), 4),
                "bm25_score": 0.0,
                "normalized_vector_score": round(float(norm_s), 4),
                "normalized_bm25_score": 0.0,
                "semantic_score": round(float(raw_s), 4),
                "keyword_score": 0.0,
                "metadata": doc.metadata,
                "is_context": False
            }
        else:
            doc_map[key]["vector_score"] = max(doc_map[key].get("vector_score", 0.0), round(float(raw_s), 4))
            doc_map[key]["normalized_vector_score"] = max(doc_map[key].get("normalized_vector_score", 0.0), round(float(norm_s), 4))
            doc_map[key]["semantic_score"] = doc_map[key]["vector_score"]

    # Ingest BM25 candidates
    for rank, ((doc, raw_s), norm_s) in enumerate(zip(keyword_results, norm_bm25_scores), start=1):
        d_name = doc.metadata.get("document") or doc.metadata.get("source", "Document")
        p_num = int(doc.metadata.get("page", 1))
        c_id = str(doc.metadata.get("chunk_id", ""))
        key = (d_name, p_num, c_id if c_id else doc.page_content[:60])

        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        if key not in doc_map:
            doc_map[key] = {
                "doc": doc,
                "document_name": d_name,
                "document_id": d_name,
                "page_number": p_num,
                "chunk_id": c_id or f"{d_name}_p{p_num}",
                "text": doc.page_content,
                "vector_score": 0.0,
                "bm25_score": round(float(raw_s), 4),
                "normalized_vector_score": 0.0,
                "normalized_bm25_score": round(float(norm_s), 4),
                "semantic_score": 0.0,
                "keyword_score": round(float(raw_s), 4),
                "metadata": doc.metadata,
                "is_context": False
            }
        else:
            doc_map[key]["bm25_score"] = max(doc_map[key].get("bm25_score", 0.0), round(float(raw_s), 4))
            doc_map[key]["normalized_bm25_score"] = max(doc_map[key].get("normalized_bm25_score", 0.0), round(float(norm_s), 4))
            doc_map[key]["keyword_score"] = doc_map[key]["bm25_score"]

    candidates = []
    for key, item in doc_map.items():
        norm_vec = item.get("normalized_vector_score", 0.0)
        norm_bm = item.get("normalized_bm25_score", 0.0)
        
        hybrid = (vector_weight * norm_vec) + (bm25_weight * norm_bm)
        item["hybrid_score"] = round(float(hybrid), 4)
        item["rrf_score"] = rrf_scores.get(key, 0.0)
        candidates.append(item)

    candidates.sort(key=lambda x: (x["hybrid_score"], x["rrf_score"]), reverse=True)
    return candidates[:top_k]


def rerank_chunks(
    query: str,
    candidate_chunks: List[Dict[str, Any]],
    top_k: int = FINAL_TOP_K,
    extracted_term: str = "",
    prior_topic: str = ""
) -> List[Dict[str, Any]]:
    """
    Reranks hybrid candidate chunks:
    Receives:
        - user query
        - candidate chunks with text and scores
    Returns:
        - reranked chunks with reranker_score, original metadata, chunk text, and scores.
    """
    return rerank_results(
        query=query,
        candidate_chunks=candidate_chunks,
        extracted_term=extracted_term,
        prior_topic=prior_topic,
        top_k=top_k
    )


def rerank_results(
    query: str,
    candidate_chunks: List[Dict[str, Any]],
    extracted_term: str = "",
    prior_topic: str = "",
    top_k: int = FINAL_TOP_K,
    topic_category: str = ""
) -> List[Dict[str, Any]]:
    """
    Reranks fused candidate chunks using real Cross-Encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)
    with high-precision heuristic fallback and topic conflict filtering.
    - Preserves all chunk metadata: document_name, page_number, chunk_id, chunk_text, vector_score, bm25_score, hybrid_score, reranker_score, final_rank.
    - Discards chunks with scores below configurable MIN_RELEVANCE_THRESHOLD.
    - Discards chunks that conflict with query intent topic.
    """
    if not candidate_chunks:
        return []

    # Ensure top_k is an integer
    effective_top_k = top_k if isinstance(top_k, int) and top_k > 0 else FINAL_TOP_K

    # Detect topic category if not supplied
    if not topic_category:
        topic_info = classify_query_intent_and_topic(query)
        topic_category = topic_info.get("category", "GENERAL")

    # Bounded candidate pool for fast neural reranking (up to 20 candidates)
    RERANK_TOP_K_POOL = 20
    pool = candidate_chunks[:RERANK_TOP_K_POOL] if len(candidate_chunks) > RERANK_TOP_K_POOL else candidate_chunks

    search_terms = get_search_terms_and_synonyms(query, prior_topic)
    pairs = [(query, (item.get("text", "") or "")[:350]) for item in pool]
    
    # 1. Attempt Cross-Encoder neural scoring (with pair score caching)
    raw_cross_scores = _CROSS_ENCODER.score_pairs(pairs) if _CROSS_ENCODER.is_available() else None
    norm_cross_scores = normalize_scores(raw_cross_scores) if raw_cross_scores else None
    
    scored_items = []

    for idx, item in enumerate(pool):
        doc = item["doc"]
        d_name = item["document_name"]
        text = item["text"]
        text_lower = text.lower()

        # Check topic conflict
        is_conflict = check_chunk_topic_conflict(topic_category, text)

        sem_score = item.get("vector_score", item.get("semantic_score", 0.5))
        kw_score = item.get("bm25_score", item.get("keyword_score", 0.0))
        norm_vec = item.get("normalized_vector_score", sem_score)
        norm_bm = item.get("normalized_bm25_score", kw_score)
        hybrid_score = item.get("hybrid_score", (VECTOR_WEIGHT * norm_vec) + (BM25_WEIGHT * norm_bm))

        # 2. Term coverage & bonuses
        local_kw_hits = sum(1 for t in search_terms if re.search(rf'\b{re.escape(t)}\b', text_lower, re.IGNORECASE))
        term_coverage = min(1.0, local_kw_hits / max(1, min(len(search_terms), 4)))
        effective_kw_score = max(norm_bm, term_coverage)
        meta_score = compute_metadata_score(query, text, d_name)

        exact_term_bonus = 0.0
        if extracted_term and len(extracted_term) > 2 and re.search(rf'\b{re.escape(extracted_term.lower())}\b', text_lower):
            exact_term_bonus = 0.08

        heading_bonus = 0.0
        first_lines = " ".join(text.split('\n')[:2]).lower()
        if any(t in first_lines for t in search_terms if len(t) > 2):
            heading_bonus = 0.06

        context_bonus = 0.04 if item.get("is_context", False) else 0.0

        if is_conflict:
            reranker_score = 0.0
        elif norm_cross_scores is not None and idx < len(norm_cross_scores):
            norm_cross = norm_cross_scores[idx]
            # Cross-encoder weighted fusion with hybrid signal + bonuses
            reranker_score = (0.70 * norm_cross) + (0.30 * hybrid_score) + exact_term_bonus + heading_bonus + context_bonus
        else:
            # Fallback multi-factor reranker
            base_fused = (VECTOR_WEIGHT * norm_vec) + (BM25_WEIGHT * effective_kw_score)
            reranker_score = base_fused + exact_term_bonus + heading_bonus + context_bonus + (0.05 * meta_score)

        final_score = round(max(0.0, min(1.0, reranker_score)), 4)
        relevance_pct = int(round(final_score * 100))

        scored_items.append({
            "doc": doc,
            "document_name": d_name,
            "document_id": d_name,
            "page_number": item["page_number"],
            "chunk_id": item.get("chunk_id", ""),
            "chunk_text": text,
            "text": text,
            "vector_score": round(sem_score, 3),
            "bm25_score": round(kw_score, 3),
            "normalized_vector_score": round(norm_vec, 3),
            "normalized_bm25_score": round(norm_bm, 3),
            "hybrid_score": round(hybrid_score, 3),
            "reranker_score": round(final_score, 3),
            "semantic_score": round(sem_score, 3),
            "keyword_score": round(kw_score, 3),
            "rrf_score": item.get("rrf_score", 0.0),
            "final_score": final_score,
            "relevance_pct": relevance_pct,
            "source_query": item.get("source_query", query)
        })

    # Sort by final reranker score descending
    scored_items.sort(key=lambda x: x["reranker_score"], reverse=True)
    top_score = scored_items[0]["reranker_score"] if scored_items else 0.0

    # =========================================================================
    # STAGE 1: CHUNK-LEVEL RELEVANCE FILTERING
    # =========================================================================
    stage1_chunks = []
    for candidate in scored_items:
        # 1. Check topic conflict
        if check_chunk_topic_conflict(topic_category, candidate["text"]):
            continue

        # 2. Check configurable minimum chunk relevance threshold
        if candidate["reranker_score"] < MIN_CHUNK_RELEVANCE:
            continue

        # 3. If top score is high, discard candidate chunks that are significantly lower
        if top_score >= 0.70 and candidate["reranker_score"] < (0.40 * top_score):
            continue

        stage1_chunks.append(candidate)

    # =========================================================================
    # STAGE 2: DOCUMENT-LEVEL RELEVANCE AGGREGATION & SELECTION
    # =========================================================================
    doc_chunks_map: Dict[str, List[Dict[str, Any]]] = {}
    for c in stage1_chunks:
        d_name = c["document_name"]
        if d_name not in doc_chunks_map:
            doc_chunks_map[d_name] = []
        doc_chunks_map[d_name].append(c)

    # Calculate document relevance for each document
    doc_scores: Dict[str, float] = {}
    for d_name, d_chunks in doc_chunks_map.items():
        doc_scores[d_name] = calculate_document_relevance(d_chunks)

    # Sort documents by document relevance descending
    sorted_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
    top_doc_score = doc_scores[sorted_docs[0]] if sorted_docs else 0.0

    is_comp_query = (topic_category == "COMPARISON" or "compare" in query.lower() or " vs " in query.lower() or " versus " in query.lower())

    # Filter documents based on MIN_DOCUMENT_RELEVANCE and dominance
    approved_docs = []
    for d_name in sorted_docs:
        d_score = doc_scores[d_name]
        # Check minimum document relevance
        if d_score < MIN_DOCUMENT_RELEVANCE and len(approved_docs) > 0:
            continue
        # If top document is dominant and not a comparison query, drop low secondary documents
        if not is_comp_query and top_doc_score >= 0.75 and d_score < (0.50 * top_doc_score):
            continue
        approved_docs.append(d_name)
        if len(approved_docs) >= MAX_RELEVANT_DOCUMENTS:
            break

    # If all docs fell below threshold but top doc has high chunk score, keep top doc
    if not approved_docs and sorted_docs and top_doc_score >= MIN_CHUNK_RELEVANCE:
        approved_docs = [sorted_docs[0]]

    # =========================================================================
    # STAGE 3: SELECT MAX_CHUNKS_PER_DOCUMENT & ASSEMBLE FINAL CANDIDATES
    # =========================================================================
    final_candidates = []
    for d_name in approved_docs:
        d_chunks = doc_chunks_map.get(d_name, [])
        d_chunks.sort(key=lambda x: x["reranker_score"], reverse=True)
        selected_from_doc = d_chunks[:MAX_CHUNKS_PER_DOCUMENT]

        for c in selected_from_doc:
            c["document_score"] = doc_scores[d_name]
            c["document_relevance_pct"] = int(round(doc_scores[d_name] * 100))
            final_candidates.append(c)

    # Global sort by reranker score descending
    final_candidates.sort(key=lambda x: x["reranker_score"], reverse=True)

    # Deduplicate near-duplicates among approved candidate chunks
    filtered_results = []
    for candidate in final_candidates:
        is_dup = False
        for existing in filtered_results:
            if is_near_duplicate(candidate["text"], existing["text"]):
                is_dup = True
                break
        if not is_dup:
            filtered_results.append(candidate)
        if len(filtered_results) >= effective_top_k:
            break

    final_top = filtered_results[:effective_top_k]

    # Assign 1-indexed final_rank and document metadata to every top chunk
    for rank_idx, cand in enumerate(final_top, start=1):
        cand["final_rank"] = rank_idx
        cand["rank"] = rank_idx
        doc = cand["doc"]
        if hasattr(doc, "metadata"):
            doc.metadata["document_name"] = cand["document_name"]
            doc.metadata["document_id"] = cand["document_name"]
            doc.metadata["page_number"] = cand["page_number"]
            doc.metadata["chunk_id"] = cand.get("chunk_id", "")
            doc.metadata["vector_score"] = cand["vector_score"]
            doc.metadata["bm25_score"] = cand["bm25_score"]
            doc.metadata["normalized_vector_score"] = cand["normalized_vector_score"]
            doc.metadata["normalized_bm25_score"] = cand["normalized_bm25_score"]
            doc.metadata["hybrid_score"] = cand["hybrid_score"]
            doc.metadata["reranker_score"] = cand["reranker_score"]
            doc.metadata["document_score"] = cand.get("document_score", cand["final_score"])
            doc.metadata["document_relevance_pct"] = cand.get("document_relevance_pct", cand["relevance_pct"])
            doc.metadata["final_rank"] = rank_idx
            doc.metadata["source_query"] = cand.get("source_query", query)

    # Absolute sanity check on top candidate score
    if final_top and final_top[0]["final_score"] < MIN_CHUNK_RELEVANCE:
        final_top = []

    return final_top


def retrieve_context(
    vectorstore: FAISS,
    query: str,
    top_k: int = FINAL_TOP_K,
    selected_documents: Optional[List[str]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    last_context: Optional[List[Any]] = None
) -> Tuple[List[Tuple[Document, float, int]], Dict[str, Any]]:
    """
    Context retrieval wrapper matching standard signature.
    """
    items, _, _, _, stats = retrieve_relevant_chunks(
        vectorstore=vectorstore,
        query=query,
        top_k=top_k,
        chat_history=chat_history,
        selected_documents=selected_documents,
        last_context=last_context
    )
    return items, stats


def multi_query_hybrid_search(
    vectorstore: FAISS,
    queries: List[str],
    selected_documents: Optional[List[str]] = None,
    vector_top_k: int = 10,
    bm25_top_k: int = 10,
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    prior_topic: str = ""
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes parallel Dense (FAISS) + Sparse (BM25) search for each generated query,
    applies hybrid score fusion per query branch, and combines + deduplicates all unique candidate chunks.
    
    Returns:
        (unique_candidate_chunks, multi_query_stats)
    """
    if not vectorstore or not queries:
        return [], {
            "original_query_results": 0,
            "expanded_query_results": 0,
            "total_candidates": 0,
            "unique_candidates": 0,
            "dense_candidates": 0,
            "sparse_candidates": 0
        }

    merged_candidates_map: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    query_result_counts = []
    total_dense = 0
    total_sparse = 0

    for idx, q_text in enumerate(queries, start=1):
        q_clean = q_text.strip()
        if not q_clean:
            continue

        # 1 & 2. FAISS dense search + BM25 sparse search in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_faiss = executor.submit(
                search_vector,
                vectorstore=vectorstore,
                query=q_clean,
                top_k=vector_top_k,
                selected_documents=selected_documents
            )
            future_bm25 = executor.submit(
                search_bm25,
                vectorstore=vectorstore,
                query=q_clean,
                top_k=bm25_top_k,
                selected_documents=selected_documents,
                prior_topic=prior_topic if idx == 1 else ""
            )
            faiss_res = future_faiss.result()
            bm25_res, _ = future_bm25.result()

        total_dense += len(faiss_res)
        total_sparse += len(bm25_res)
        logger.info(f"[FAISS] Query {idx} results: {len(faiss_res)}")
        logger.info(f"[BM25] Query {idx} results: {len(bm25_res)}")

        # 3. Hybrid fusion for this query branch
        fused_branch = fuse_search_results(
            semantic_results=faiss_res,
            keyword_results=bm25_res,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=RERANK_CANDIDATES
        )
        query_result_counts.append(len(fused_branch))

        # 4. Ingest and deduplicate across query branches
        for item in fused_branch:
            d_name = item["document_name"]
            p_num = item["page_number"]
            c_id = item.get("chunk_id", "")
            key = (d_name, p_num, c_id if c_id else item["text"][:60])

            item_copy = dict(item)
            item_copy["source_query"] = q_clean

            if key not in merged_candidates_map:
                merged_candidates_map[key] = item_copy
            else:
                existing = merged_candidates_map[key]
                # Keep highest scores for duplicates
                existing["vector_score"] = max(existing.get("vector_score", 0.0), item_copy.get("vector_score", 0.0))
                existing["bm25_score"] = max(existing.get("bm25_score", 0.0), item_copy.get("bm25_score", 0.0))
                existing["normalized_vector_score"] = max(existing.get("normalized_vector_score", 0.0), item_copy.get("normalized_vector_score", 0.0))
                existing["normalized_bm25_score"] = max(existing.get("normalized_bm25_score", 0.0), item_copy.get("normalized_bm25_score", 0.0))
                existing["hybrid_score"] = max(existing.get("hybrid_score", 0.0), item_copy.get("hybrid_score", 0.0))
                existing["rrf_score"] = existing.get("rrf_score", 0.0) + item_copy.get("rrf_score", 0.0)

    unique_candidates = list(merged_candidates_map.values())
    unique_candidates.sort(key=lambda x: (x.get("hybrid_score", 0.0), x.get("rrf_score", 0.0)), reverse=True)

    orig_count = query_result_counts[0] if query_result_counts else 0
    exp_count = sum(query_result_counts[1:]) if len(query_result_counts) > 1 else 0

    stats = {
        "original_query_results": orig_count,
        "expanded_query_results": exp_count,
        "total_candidates": sum(query_result_counts),
        "unique_candidates": len(unique_candidates),
        "dense_candidates": total_dense,
        "sparse_candidates": total_sparse
    }

    return unique_candidates, stats


# =============================================================================
# 5. END-TO-END MULTI-QUERY HYBRID RETRIEVAL PIPELINE
# =============================================================================
def retrieve_relevant_chunks(
    vectorstore: FAISS,
    query: str,
    top_k: Optional[int] = None,
    candidate_pool_size: int = VECTOR_TOP_K,
    chat_history: Optional[List[Dict[str, str]]] = None,
    selected_documents: Optional[List[str]] = None,
    last_context: Optional[List[Any]] = None,
    llm_client: Optional[Any] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    skip_query_expansion: bool = False
) -> Tuple[List[Tuple[Document, float, int]], str, str, str, Dict[str, Any]]:
    """
    End-to-End Query Expansion + Multi-Query Parallel Hybrid Retrieval + Cross-Encoder Reranking Pipeline:
    1. Conversational Memory Query Rewriting (resolves pronouns/context).
    2. Query Expansion (Gemini generates up to 2 alternative search queries, max 3 total).
    3. For EACH query:
       - Dense Vector Search using FAISS (top_k = 10)
       - Sparse Keyword Search using BM25 (top_k = 10)
       - Hybrid Score Fusion (0.6 FAISS + 0.4 BM25)
    4. Multi-Query Result Fusion & Deduplication (by document_name + page_number + chunk_id).
    5. Cross-Encoder Reranking (scores original question vs chunk_text using ms-marco-MiniLM-L-6-v2).
    6. Final Top-K Chunks Selection (Top 5).
    
    Returns:
        (retrieved_items, query_type, extracted_term, rewritten_query, retrieval_stats)
    """
    default_stats = {
        "strategy": "MULTI-QUERY HYBRID RAG",
        "method": "Multi-Query Hybrid Search",
        "dense_method": "FAISS",
        "sparse_method": "BM25",
        "reranker_method": "Cross-Encoder (ms-marco-MiniLM-L-6-v2)" if _CROSS_ENCODER.is_available() else "Lightweight Reranker",
        "original_query": query,
        "rewritten_query": query,
        "expanded_queries": [query] if query else [],
        "original_results_count": 0,
        "expanded_results_count": 0,
        "total_candidates": 0,
        "unique_candidates": 0,
        "semantic_candidates": 0,
        "keyword_candidates": 0,
        "deduped_candidates": 0,
        "reranked_candidates": 0,
        "final_chunks": 0,
        "query_expansion_ms": 0.0,
        "retrieval_ms": 0.0,
        "reranking_ms": 0.0
    }

    if vectorstore is None:
        logger.warning("Vectorstore is None. Cannot perform retrieval.")
        return [], "NORMAL_QUESTION", "", query, default_stats

    raw_query = query.strip() if query else ""
    if not raw_query:
        return [], "NORMAL_QUESTION", "", "", default_stats

    # Handle case where selected_documents was passed as 3rd positional argument
    if isinstance(top_k, (list, tuple, set)):
        selected_documents = list(top_k)
        top_k = FINAL_TOP_K

    target_top_k = top_k if isinstance(top_k, int) and top_k > 0 else FINAL_TOP_K
    selected_set = set(selected_documents) if selected_documents else None

    # Step 1: Follow-Up Understanding & Query Rewriting using Conversation Memory (Strict Isolation)
    is_followup, followup_type, context_topic, comp_concepts = detect_followup_question(raw_query, chat_history, last_context)
    if is_followup:
        rewritten_query, prior_topic = rewrite_query(raw_query, chat_history, last_context)
    else:
        rewritten_query = raw_query
        prior_topic = ""
        last_context = None

    # Step 2: Query Classification, Intent Detection & Query Normalization
    query_type, extracted_term = classify_query(rewritten_query)
    topic_info = classify_query_intent_and_topic(rewritten_query)
    query_topic = topic_info.get("topic", "General / Document Retrieval")
    topic_category = topic_info.get("category", "GENERAL")
    normalized_retrieval_q = normalize_retrieval_query(rewritten_query)

    try:
        # Step 3: Query Expansion with Gemini for complex queries, or FAST MODE for simple queries
        t_exp_start = time.perf_counter()
        expanded_queries = [normalized_retrieval_q] if normalized_retrieval_q != rewritten_query else [rewritten_query]
        from rag.llm import is_simple_query, expand_query_with_gemini

        if not skip_query_expansion and not is_simple_query(rewritten_query):
            try:
                expanded_queries = expand_query_with_gemini(
                    llm_client=llm_client,
                    query=rewritten_query,
                    api_key=api_key,
                    model_name=model_name,
                    max_expanded=2
                )
                if normalized_retrieval_q not in expanded_queries:
                    expanded_queries.append(normalized_retrieval_q)
            except Exception as err:
                logger.warning(f"Query expansion skipped or failed ({err}). Using normalized query.")
                expanded_queries = [normalized_retrieval_q]
        else:
            logger.info(f"[FAST_MODE] Direct retrieval for query: '{rewritten_query}' (normalized: '{normalized_retrieval_q}')")
        t_expansion_ms = round((time.perf_counter() - t_exp_start) * 1000, 1)

        # Logging Query Expansion
        logger.info("----------------------------------")
        logger.info(f"[MULTI_QUERY] Original query: {raw_query}")
        if rewritten_query != raw_query:
            logger.info(f"[MULTI_QUERY] Resolved query: {rewritten_query}")
        logger.info(f"[MULTI_QUERY] Normalized retrieval query: {normalized_retrieval_q}")
        logger.info(f"[TOPIC_INTENT] Classified Topic: {query_topic} ({topic_category})")
        logger.info(f"[MULTI_QUERY] Generated queries: {len(expanded_queries)}")
        for q_i, q_str in enumerate(expanded_queries, 1):
            logger.info(f"[MULTI_QUERY] Query {q_i}: {q_str}")

        # Step 4: Parallel Multi-Query Hybrid Search (FAISS top 10 + BM25 top 10 per query)
        t_retrieval_start = time.perf_counter()
        unique_candidates, mq_stats = multi_query_hybrid_search(
            vectorstore=vectorstore,
            queries=expanded_queries,
            selected_documents=selected_documents,
            vector_top_k=10,
            bm25_top_k=10,
            vector_weight=VECTOR_WEIGHT,
            bm25_weight=BM25_WEIGHT,
            prior_topic=prior_topic
        )
        t_retrieval_ms = round((time.perf_counter() - t_retrieval_start) * 1000, 1)

        logger.info(f"[FUSION] Total candidates: {mq_stats['total_candidates']}")
        logger.info(f"[DEDUP] Unique candidates: {mq_stats['unique_candidates']}")

        # Step 5: Neural Cross-Encoder Reranking with topic conflict filtering & relevance threshold
        t_rerank_start = time.perf_counter()
        final_top = rerank_results(
            query=rewritten_query,
            candidate_chunks=unique_candidates,
            extracted_term=extracted_term,
            prior_topic=prior_topic,
            top_k=target_top_k,
            topic_category=topic_category
        )
        t_rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000, 1)

        # Step 6: Final Context Validation (Requirement 5)
        is_valid, validation_reason, validated_top = validate_retrieved_context(
            query=rewritten_query,
            candidate_chunks=final_top,
            min_score=MIN_CHUNK_RELEVANCE
        )
        if not is_valid:
            logger.info(f"[CONTEXT_VALIDATION] Failed for query '{rewritten_query}': {validation_reason}. Rejecting candidates.")
            final_top = []

        candidates_count = len(unique_candidates)
        after_reranking_count = min(candidates_count, 20)
        final_chunks_count = len(final_top)
        removed_count = max(0, candidates_count - final_chunks_count)

        logger.info(f"[RERANK] Candidates: {candidates_count}")
        logger.info(f"[RERANK] Final chunks: {final_chunks_count} (Removed as irrelevant: {removed_count})")
        logger.info(f"[TIMING] Expansion: {t_expansion_ms}ms | Retrieval: {t_retrieval_ms}ms | Reranking: {t_rerank_ms}ms")
        logger.info(f"Selected documents: {', '.join(selected_set) if selected_set else 'All Documents'}")
        if final_top:
            top_res = final_top[0]
            logger.info(f"[TOP RESULT] {top_res['document_name']} P{top_res['page_number']} | Hybrid: {top_res.get('hybrid_score', 0.0)} | Rerank: {top_res.get('reranker_score', top_res['final_score'])}")
        logger.info("----------------------------------")

        relevant_docs_set = set(res["document_name"] for res in final_top)
        relevant_docs_count = len(relevant_docs_set)
        top_doc_name = final_top[0]["document_name"] if final_top else "None"
        top_page_num = final_top[0]["page_number"] if final_top else 1
        primary_source_label = f"{top_doc_name} — Page {top_page_num}" if final_top else "None"

        # Build document relevance map
        doc_rel_map = {}
        for res in final_top:
            d = res["document_name"]
            if d not in doc_rel_map:
                doc_rel_map[d] = {
                    "document": d,
                    "document_score": res.get("document_score", res["final_score"]),
                    "relevance_pct": res.get("document_relevance_pct", res["relevance_pct"]),
                    "top_page": res["page_number"]
                }

        retrieval_stats = {
            "strategy": "MULTI-QUERY HYBRID RAG",
            "method": "Multi-Query Hybrid Search",
            "dense_method": "FAISS",
            "sparse_method": "BM25",
            "reranker_method": "Cross-Encoder (ms-marco-MiniLM-L-6-v2)" if _CROSS_ENCODER.is_available() else "Lightweight Reranker",
            "current_query": raw_query,
            "query": raw_query,
            "normalized_query": normalized_retrieval_q,
            "query_intent": query_topic,
            "query_topic": query_topic,
            "topic_category": topic_category,
            "candidates_retrieved": candidates_count,
            "retrieved_chunks": candidates_count,
            "rejected_chunks": removed_count,
            "removed_as_irrelevant": removed_count,
            "after_reranking": after_reranking_count,
            "reranked_chunks": after_reranking_count,
            "after_chunk_filter": final_chunks_count,
            "after_relevance_filtering": final_chunks_count,
            "relevant_documents": relevant_docs_count,
            "final_context_chunks": final_chunks_count,
            "source_documents": list(doc_rel_map.keys()),
            "top_relevance_scores": {d: info["relevance_pct"] for d, info in doc_rel_map.items()},
            "primary_source": primary_source_label,
            "primary_document": top_doc_name,
            "primary_page": top_page_num,
            "document_relevance_scores": doc_rel_map,
            "original_query": raw_query,
            "rewritten_query": rewritten_query,
            "expanded_queries": expanded_queries,
            "original_results_count": mq_stats.get("original_query_results", 0),
            "expanded_results_count": mq_stats.get("expanded_query_results", 0),
            "total_candidates": mq_stats.get("total_candidates", 0),
            "unique_candidates": candidates_count,
            "semantic_candidates": mq_stats.get("dense_candidates", 0),
            "keyword_candidates": mq_stats.get("sparse_candidates", 0),
            "deduped_candidates": candidates_count,
            "reranked_candidates": candidates_count,
            "final_chunks": final_chunks_count,
            "query_expansion_ms": t_expansion_ms,
            "retrieval_ms": t_retrieval_ms,
            "reranking_ms": t_rerank_ms
        }

        return [(res["doc"], res["final_score"], res["relevance_pct"]) for res in final_top], query_type, extracted_term, rewritten_query, retrieval_stats

    except Exception as err:
        logger.error(f"Error during multi-query hybrid retrieval: {str(err)}", exc_info=True)
        return [], query_type, extracted_term, rewritten_query, default_stats



def group_chunks_by_document(retrieved_items: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups retrieved chunks by document filename for structured comparison processing.
    """
    grouped = {}
    for item in retrieved_items:
        if isinstance(item, (tuple, list)):
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

        if doc_name not in grouped:
            grouped[doc_name] = []

        grouped[doc_name].append({
            "doc": doc,
            "document": doc_name,
            "filename": doc_name,
            "page": page_num,
            "text": chunk_text,
            "score": round(score, 3),
            "relevance_pct": rel_pct,
            "pdf_url": f"/view_pdf/{doc_name}#page={page_num}"
        })

    return grouped


def build_grouped_sources_payload(sources: List[Dict[str, Any]], primary_doc: str = "") -> List[Dict[str, Any]]:
    """
    Groups a list of source chunks by document, aggregating page numbers and document-level relevance.
    """
    grouped_sources_map: Dict[str, Dict[str, Any]] = {}
    for src in sources:
        d = src.get("document") or src.get("filename") or "Document"
        if d not in grouped_sources_map:
            doc_score = src.get("document_score", src.get("score", 0.9))
            rel_pct = src.get("document_relevance_pct", src.get("relevance_pct", int(round(doc_score * 100))))
            grouped_sources_map[d] = {
                "document": d,
                "document_id": d,
                "filename": d,
                "document_score": round(float(doc_score), 3),
                "relevance_pct": int(rel_pct),
                "is_primary": bool(src.get("is_primary", False) or (primary_doc and d == primary_doc)),
                "pages": [],
                "chunks": []
            }
        if src.get("page") and src["page"] not in grouped_sources_map[d]["pages"]:
            grouped_sources_map[d]["pages"].append(src["page"])
        grouped_sources_map[d]["chunks"].append(src)

    grouped_sources_list = sorted(grouped_sources_map.values(), key=lambda x: x["document_score"], reverse=True)
    for g in grouped_sources_list:
        g["pages"] = sorted(list(set(g["pages"])))
    return grouped_sources_list


def retrieve_multi_document_context(
    vectorstore: FAISS,
    query: str,
    comparison_topics: List[str],
    selected_documents: Optional[List[str]] = None,
    candidate_pool_size: int = 16,
    final_k_per_doc: int = 3,
    top_k_final: int = 8,
    chat_history: Optional[List[Dict[str, str]]] = None,
    last_context: Optional[List[Any]] = None,
    comparison_subtype: str = "COMPARISON_TABLE"
) -> Tuple[List[Tuple[Document, float, int]], Dict[str, List[Dict[str, Any]]], str, str, Dict[str, Any]]:
    """
    Document-Wise Balanced Retrieval Pipeline for Multi-Document Comparison & Synthesis:
    1. Retrieves relevant chunks separately from each selected document to prevent document starvation.
    2. Maintains document_name, page_number, chunk_text, and relevance_score.
    3. Runs dedicated retrieval branches for comparison topics + general query.
    4. Reranks and selects balanced top-K chunks per document.
    5. Groups results by document.
    
    Returns:
        (retrieved_items, grouped_chunks, query_type, rewritten_query, retrieval_stats)
    """
    default_stats = {
        "method": "Multi-Document Hybrid Search",
        "semantic_candidates": 0,
        "keyword_candidates": 0,
        "deduped_candidates": 0,
        "reranked_candidates": 0,
        "final_chunks": 0,
        "documents_compared": 0
    }

    if vectorstore is None or not query:
        return [], {}, "COMPARISON_QUESTION", query, default_stats

    # Determine target documents for document-wise retrieval
    target_docs = list(selected_documents) if selected_documents else []
    if not target_docs and hasattr(vectorstore, "docstore") and hasattr(vectorstore.docstore, "_dict"):
        found_in_vs = set()
        for d_obj in vectorstore.docstore._dict.values():
            meta = getattr(d_obj, "metadata", {})
            name = meta.get("document") or meta.get("source")
            if name:
                found_in_vs.add(name)
        if len(found_in_vs) >= 2:
            target_docs = sorted(list(found_in_vs))

    # Formulate retrieval sub-queries
    queries_to_run = [query]
    if comparison_topics:
        for t in comparison_topics:
            if t and t.lower() not in query.lower():
                queries_to_run.append(t)
            elif t:
                queries_to_run.append(f"{t} definition concepts details")
    if comparison_subtype == "SUMMARY_ALL":
        queries_to_run.append("overview key concepts summary main topics")
    elif comparison_subtype in ("COMMON_INFO", "SIMILARITIES"):
        queries_to_run.append("common features similarities principles concepts")
    elif comparison_subtype == "DIFFERENCES":
        queries_to_run.append("differences contrast distinct characteristics")

    all_retrieved_candidates = []
    seen_keys = set()
    total_semantic_count = 0
    total_keyword_count = 0

    # Balanced per-document retrieval:
    # When two or more target documents are selected, querying all documents in a single
    # global search can lead to "document starvation" — where one document with higher lexical
    # density or term matches consumes the entire candidate pool, starving other documents.
    # To ensure balanced comparison, retrieve candidates separately for each target document,
    # allocating a safe per-document candidate pool while respecting relevance filtering.
    if len(target_docs) >= 2:
        per_doc_pool = max(final_k_per_doc * 2, max(4, candidate_pool_size // len(target_docs)))
        for doc_name in target_docs:
            for q_branch in queries_to_run[:3]:
                items, _, _, _, stats = retrieve_relevant_chunks(
                    vectorstore=vectorstore,
                    query=q_branch,
                    top_k=per_doc_pool,
                    candidate_pool_size=per_doc_pool,
                    chat_history=chat_history,
                    selected_documents=[doc_name],
                    last_context=last_context,
                    skip_query_expansion=True
                )
                total_semantic_count += stats.get("semantic_candidates", 0)
                total_keyword_count += stats.get("keyword_candidates", 0)

                for item in items:
                    doc = item[0]
                    score = float(item[1])
                    rel_pct = int(item[2]) if len(item) > 2 else int(round(score * 100))
                    d_name = doc.metadata.get("document") or doc.metadata.get("source", doc_name)
                    p_num = doc.metadata.get("page", 1)
                    c_id = doc.metadata.get("chunk_id", "")
                    key = (d_name, p_num, c_id) if c_id else (d_name, p_num, doc.page_content[:60])

                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_retrieved_candidates.append({
                            "doc": doc,
                            "document_name": d_name,
                            "page_number": p_num,
                            "chunk_id": c_id,
                            "text": doc.page_content,
                            "score": score,
                            "relevance_pct": rel_pct
                        })
    else:
        # Single document or unconstrained retrieval: preserve existing behavior exactly
        target_filter = target_docs if (selected_documents and len(selected_documents) > 0) else None
        for q_branch in queries_to_run[:3]:
            items, _, _, _, stats = retrieve_relevant_chunks(
                vectorstore=vectorstore,
                query=q_branch,
                top_k=candidate_pool_size,
                candidate_pool_size=candidate_pool_size,
                chat_history=chat_history,
                selected_documents=target_filter,
                last_context=last_context,
                skip_query_expansion=True
            )
            total_semantic_count += stats.get("semantic_candidates", 0)
            total_keyword_count += stats.get("keyword_candidates", 0)

            for item in items:
                doc = item[0]
                score = float(item[1])
                rel_pct = int(item[2]) if len(item) > 2 else int(round(score * 100))
                d_name = doc.metadata.get("document") or doc.metadata.get("source", "Document")
                p_num = doc.metadata.get("page", 1)
                c_id = doc.metadata.get("chunk_id", "")
                key = (d_name, p_num, c_id) if c_id else (d_name, p_num, doc.page_content[:60])

                if key not in seen_keys:
                    seen_keys.add(key)
                    all_retrieved_candidates.append({
                        "doc": doc,
                        "document_name": d_name,
                        "page_number": p_num,
                        "chunk_id": c_id,
                        "text": doc.page_content,
                        "score": score,
                        "relevance_pct": rel_pct
                    })

    # Balanced per-document candidate grouping
    per_doc_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for cand in all_retrieved_candidates:
        d = cand["document_name"]
        if d not in per_doc_buckets:
            per_doc_buckets[d] = []
        per_doc_buckets[d].append(cand)

    # Sort each document bucket by relevance score descending
    for d in per_doc_buckets:
        per_doc_buckets[d].sort(key=lambda x: x["score"], reverse=True)

    # Balanced selection: take up to final_k_per_doc per document
    balanced_candidates = []
    max_depth = max((len(b) for b in per_doc_buckets.values()), default=0)
    for depth in range(max_depth):
        for d, bucket in per_doc_buckets.items():
            if depth < len(bucket) and depth < final_k_per_doc:
                balanced_candidates.append(bucket[depth])

    # Deduplicate near-duplicate chunks
    final_filtered = []
    for cand in balanced_candidates:
        if not any(is_near_duplicate(cand["text"], exist["text"]) for exist in final_filtered):
            final_filtered.append(cand)
        if len(final_filtered) >= max(top_k_final, len(per_doc_buckets) * final_k_per_doc):
            break

    final_top = final_filtered
    retrieved_items = [(res["doc"], res["score"], res["relevance_pct"]) for res in final_top]
    grouped_chunks = group_chunks_by_document(retrieved_items)
    documents_used = sorted(list(grouped_chunks.keys()))

    retrieval_stats = {
        "method": "Multi-Document Hybrid Search",
        "semantic_candidates": total_semantic_count,
        "keyword_candidates": total_keyword_count,
        "deduped_candidates": len(all_retrieved_candidates),
        "reranked_candidates": len(balanced_candidates),
        "final_chunks": len(retrieved_items),
        "documents_compared": len(documents_used),
        "comparison_subtype": comparison_subtype
    }

    logger.info("----------------------------------")
    logger.info("[MULTI-DOCUMENT COMPARISON SEARCH]")
    logger.info(f"Query: {query}")
    logger.info(f"Subtype: {comparison_subtype}")
    logger.info(f"Comparison topics: {comparison_topics}")
    logger.info(f"Documents used ({len(documents_used)}): {', '.join(documents_used)}")
    logger.info(f"Final chunks retrieved: {len(retrieved_items)}")
    logger.info("----------------------------------")

    return retrieved_items, grouped_chunks, "COMPARISON_QUESTION", query, retrieval_stats


# =============================================================================
# 4.5. COMPLEX QUERY DECOMPOSITION RETRIEVAL PIPELINE
# =============================================================================
def retrieve_with_query_decomposition(
    vectorstore: FAISS,
    query: str,
    sub_queries: List[str],
    selected_documents: Optional[List[str]] = None,
    vector_top_k: int = 10,
    bm25_top_k: int = 10,
    subquery_top_candidates: int = 8,
    final_top_k: int = 6,
    prior_topic: str = ""
) -> Tuple[List[Tuple[Document, float, int]], Dict[str, Any]]:
    """
    Complex Query Decomposition Retrieval Pipeline:
    1. Receives 2 to 5 sub-queries decomposed from a complex multi-part/comparative user question.
    2. For EACH sub-query:
       - Runs Hybrid Search (FAISS dense + BM25 sparse).
       - Applies Cross-Encoder reranking on the sub-query to capture the best chunks for that sub-task.
       - Preserves document name, page number, and chunk ID.
    3. Merges and deduplicates retrieved chunks across all sub-queries (key: document_name + page_number + chunk_id),
       preserving maximum scores and tracking matched sub-queries.
    4. Performs global Cross-Encoder reranking using the original complex question.
    5. Returns the final top-K chunks and comprehensive decomposition statistics.
    """
    default_stats = {
        "strategy": "QUERY DECOMPOSITION HYBRID RAG",
        "method": "Query Decomposition",
        "is_decomposed": True,
        "dense_method": "FAISS",
        "sparse_method": "BM25",
        "reranker_method": "Cross-Encoder (ms-marco-MiniLM-L-6-v2)" if _CROSS_ENCODER.is_available() else "Lightweight Reranker",
        "original_query": query,
        "sub_queries": sub_queries or [],
        "sub_query_results": [],
        "original_results_count": 0,
        "expanded_results_count": 0,
        "total_candidates": 0,
        "unique_candidates": 0,
        "semantic_candidates": 0,
        "keyword_candidates": 0,
        "deduped_candidates": 0,
        "reranked_candidates": 0,
        "final_chunks": 0,
        "retrieval_ms": 0.0,
        "reranking_ms": 0.0
    }

    if vectorstore is None or not query or not sub_queries:
        return [], default_stats

    t_retrieval_start = time.perf_counter()
    total_dense = 0
    total_sparse = 0
    sub_query_info_list = []
    merged_candidates_map: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for sq_idx, sub_q in enumerate(sub_queries, 1):
        clean_sq = sub_q.strip()
        if not clean_sq:
            continue

        # 1 & 2. Vector Search + BM25 Search in parallel for sub-query
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_v = executor.submit(
                search_vector,
                vectorstore=vectorstore,
                query=clean_sq,
                top_k=vector_top_k,
                selected_documents=selected_documents
            )
            future_b = executor.submit(
                search_bm25,
                vectorstore=vectorstore,
                query=clean_sq,
                top_k=bm25_top_k,
                selected_documents=selected_documents,
                prior_topic=prior_topic
            )
            v_res = future_v.result()
            b_res, _ = future_b.result()

        total_dense += len(v_res)
        total_sparse += len(b_res)

        # 3. Hybrid fusion for sub-query
        fused = fuse_search_results(
            semantic_results=v_res,
            keyword_results=b_res,
            vector_weight=VECTOR_WEIGHT,
            bm25_weight=BM25_WEIGHT,
            top_k=RERANK_CANDIDATES
        )

        # 4. Cross-Encoder rerank for this sub-query
        sq_reranked = rerank_chunks(
            query=clean_sq,
            candidate_chunks=fused,
            top_k=subquery_top_candidates
        )

        top_doc_name = sq_reranked[0]["document_name"] if sq_reranked else "None"
        top_page_num = sq_reranked[0]["page_number"] if sq_reranked else 1

        sub_query_info_list.append({
            "sub_query": clean_sq,
            "subquery_index": sq_idx,
            "dense_results": len(v_res),
            "sparse_results": len(b_res),
            "fused_results": len(fused),
            "reranked_results": len(sq_reranked),
            "top_document": top_doc_name,
            "top_page": top_page_num
        })

        # Tag chunks with sub-query provenance and merge into global candidate map
        for chunk_item in sq_reranked:
            d_name = chunk_item["document_name"]
            p_num = chunk_item["page_number"]
            c_id = chunk_item.get("chunk_id", f"{d_name}_p{p_num}")
            key = (d_name, p_num, c_id)

            item_copy = dict(chunk_item)
            item_copy["source_query"] = clean_sq
            item_copy["sub_queries_matched"] = [clean_sq]

            if key not in merged_candidates_map:
                merged_candidates_map[key] = item_copy
            else:
                existing = merged_candidates_map[key]
                if clean_sq not in existing.get("sub_queries_matched", []):
                    existing["sub_queries_matched"] = existing.get("sub_queries_matched", []) + [clean_sq]
                existing["vector_score"] = max(existing.get("vector_score", 0.0), item_copy.get("vector_score", 0.0))
                existing["bm25_score"] = max(existing.get("bm25_score", 0.0), item_copy.get("bm25_score", 0.0))
                existing["normalized_vector_score"] = max(existing.get("normalized_vector_score", 0.0), item_copy.get("normalized_vector_score", 0.0))
                existing["normalized_bm25_score"] = max(existing.get("normalized_bm25_score", 0.0), item_copy.get("normalized_bm25_score", 0.0))
                existing["hybrid_score"] = max(existing.get("hybrid_score", 0.0), item_copy.get("hybrid_score", 0.0))
                existing["rrf_score"] = existing.get("rrf_score", 0.0) + item_copy.get("rrf_score", 0.0)

    t_retrieval_ms = round((time.perf_counter() - t_retrieval_start) * 1000, 1)

    unique_candidates = list(merged_candidates_map.values())
    unique_candidates.sort(key=lambda x: (x.get("hybrid_score", 0.0), x.get("rrf_score", 0.0)), reverse=True)

    # 5. Global Cross-Encoder Reranking using the original complex question
    t_rerank_start = time.perf_counter()
    final_top = rerank_chunks(
        query=query,
        candidate_chunks=unique_candidates,
        top_k=final_top_k
    )
    t_rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000, 1)

    logger.info("----------------------------------")
    logger.info("[QUERY DECOMPOSITION RETRIEVAL]")
    logger.info(f"Original complex query: {query}")
    logger.info(f"Sub-queries executed ({len(sub_queries)}):")
    for i, sq in enumerate(sub_queries, 1):
        logger.info(f"  [{i}] {sq}")
    logger.info(f"Total candidates across sub-queries: {total_dense + total_sparse}")
    logger.info(f"Unique deduplicated chunks: {len(unique_candidates)}")
    logger.info(f"Final selected chunks: {len(final_top)}")
    logger.info("----------------------------------")

    retrieval_stats = {
        "strategy": "QUERY DECOMPOSITION HYBRID RAG",
        "method": "Query Decomposition",
        "is_decomposed": True,
        "dense_method": "FAISS",
        "sparse_method": "BM25",
        "reranker_method": "Cross-Encoder (ms-marco-MiniLM-L-6-v2)" if _CROSS_ENCODER.is_available() else "Lightweight Reranker",
        "original_query": query,
        "sub_queries": sub_queries,
        "sub_query_results": sub_query_info_list,
        "original_results_count": total_dense + total_sparse,
        "expanded_results_count": 0,
        "total_candidates": total_dense + total_sparse,
        "unique_candidates": len(unique_candidates),
        "semantic_candidates": total_dense,
        "keyword_candidates": total_sparse,
        "deduped_candidates": len(unique_candidates),
        "reranked_candidates": len(unique_candidates),
        "final_chunks": len(final_top),
        "retrieval_ms": t_retrieval_ms,
        "reranking_ms": t_rerank_ms
    }

    retrieved_items = [(res["doc"], res["final_score"], res["relevance_pct"]) for res in final_top]
    return retrieved_items, retrieval_stats


# =============================================================================
# 5. FAST DOCUMENT SEARCH (NO GEMINI CALLS)
# =============================================================================
def build_context_snippet(text: str, terms: List[str], max_len: int = 240) -> str:
    """
    Builds a clean, contextual excerpt snippet centered around matched keywords.
    """
    if not text:
        return ""
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean

    # Find earliest match among search terms
    first_pos = -1
    for t in terms:
        if not t or len(t) < 2:
            continue
        pos = clean.lower().find(t.lower())
        if pos != -1 and (first_pos == -1 or pos < first_pos):
            first_pos = pos

    if first_pos == -1 or first_pos < 30:
        return clean[:max_len].rsplit(' ', 1)[0] + "..."

    start = max(0, first_pos - 35)
    end = min(len(clean), start + max_len)
    snippet = clean[start:end]
    
    if start > 0:
        # Find first space to avoid cutting word in half
        space_idx = snippet.find(' ')
        if space_idx > 0 and space_idx < 15:
            snippet = snippet[space_idx + 1:]
        snippet = "..." + snippet.lstrip()

    if end < len(clean):
        snippet = snippet.rsplit(' ', 1)[0] + "..."

    return snippet


def search_documents_fast(
    vectorstore: FAISS,
    query: str,
    selected_documents: Optional[List[str]] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Fast, hybrid semantic + keyword document search WITHOUT calling Gemini LLM.
    - Searches indexed chunks across selected documents.
    - Accurately normalizes FAISS distance into cosine similarity.
    - Extracts matched terms for UI highlighting.
    - Returns structured results with document, page, snippet, full excerpt, and score.
    
    Target execution: Fast response (< 1s).
    """
    if vectorstore is None or not query or not query.strip():
        return []

    clean_query = query.strip()
    selected_set = set(selected_documents) if selected_documents else None
    search_terms = get_search_terms_and_synonyms(clean_query)
    
    # 1. Exact & Partial Keyword Search across docstore
    keyword_results, matched_terms_overall = exact_keyword_search(
        vectorstore=vectorstore,
        query=clean_query,
        top_k=top_k * 2,
        selected_documents=selected_documents
    )
    
    # 2. FAISS Similarity Search (semantic vectors)
    faiss_candidates = []
    try:
        faiss_k = top_k * 3 if selected_set is not None else top_k * 2
        raw_faiss = vectorstore.similarity_search_with_score(
            query=clean_query,
            k=faiss_k
        )
        for doc, distance in raw_faiss:
            d_name = doc.metadata.get("document") or doc.metadata.get("source", "Document")
            if selected_set is None or d_name in selected_set:
                faiss_candidates.append((doc, distance))
    except Exception as err:
        logger.warning(f"FAISS search warning during document search: {err}")

    # 3. Merge & Deduplicate candidates
    merged = {}
    
    for doc, kw_score in keyword_results:
        d_name = doc.metadata.get("document") or doc.metadata.get("source", "Document")
        p_num = doc.metadata.get("page", 1)
        c_id = doc.metadata.get("chunk_id", "")
        key = (d_name, p_num, c_id) if c_id else (d_name, p_num, doc.page_content[:60])
        
        merged[key] = {
            "doc": doc,
            "document_name": d_name,
            "page_number": p_num,
            "chunk_id": c_id,
            "text": doc.page_content,
            "distance": 0.40,  # strong initial distance for verified keyword match
            "kw_score": kw_score
        }

    for doc, distance in faiss_candidates:
        d_name = doc.metadata.get("document") or doc.metadata.get("source", "Document")
        p_num = doc.metadata.get("page", 1)
        c_id = doc.metadata.get("chunk_id", "")
        key = (d_name, p_num, c_id) if c_id else (d_name, p_num, doc.page_content[:60])
        
        if key not in merged:
            merged[key] = {
                "doc": doc,
                "document_name": d_name,
                "page_number": p_num,
                "chunk_id": c_id,
                "text": doc.page_content,
                "distance": float(distance),
                "kw_score": 0.0
            }
        else:
            merged[key]["distance"] = min(merged[key]["distance"], float(distance))

    # 4. Score each merged candidate
    scored_results = []
    
    for item in merged.values():
        text = item["text"]
        d_name = item["document_name"]
        p_num = item["page_number"]
        
        sem_score = compute_semantic_score(item["distance"])
        kw_scan_score = item.get("kw_score", 0.0)
        
        # Calculate local matched terms
        local_matched_terms = []
        text_lower = text.lower()
        for t in search_terms:
            if re.search(rf'\b{re.escape(t)}\b', text_lower, re.IGNORECASE):
                local_matched_terms.append(t)
                
        local_kw_score = min(1.0, len(local_matched_terms) / max(1, min(len(search_terms), 4)))
        kw_score = max(kw_scan_score, local_kw_score)
        meta_score = compute_metadata_score(clean_query, text, d_name)
        
        final_score = max(sem_score, (0.45 * sem_score) + (0.45 * kw_score) + (0.10 * meta_score))
        final_score = round(max(0.0, min(1.0, final_score)), 4)
        
        if final_score < 0.15:
            continue
            
        relevance_pct = int(round(final_score * 100))
        
        # Build contextual snippet
        snippet = build_context_snippet(text, local_matched_terms or search_terms)
        
        scored_results.append({
            "document_id": d_name,
            "document": d_name,
            "filename": d_name,
            "page": p_num,
            "score": round(final_score, 3),
            "relevance_pct": relevance_pct,
            "snippet": snippet,
            "excerpt": text,
            "pdf_url": f"/view_pdf/{d_name}#page={p_num}",
            "matched_terms": local_matched_terms or search_terms[:3]
        })

    # Sort descending by score
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Near duplicate suppression
    unique_results = []
    for cand in scored_results:
        if not any(is_near_duplicate(cand["excerpt"], u["excerpt"]) for u in unique_results):
            unique_results.append(cand)
        if len(unique_results) >= top_k:
            break
            
    return unique_results


# =============================================================================
# 9. ADAPTIVE RETRIEVAL WITH AUTOMATIC RETRY PIPELINE
# =============================================================================
def generate_adaptive_retry_queries(
    original_query: str,
    weak_claims: List[str],
    llm_client: Optional[Any] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> List[str]:
    """
    Generates focused, targeted search queries to locate document evidence for weak/unsupported claims.
    Uses Gemini LLM when available; falls back to domain entity & keyword extraction.
    Guarantees:
    - Never searches the internet. Only searches indexed PDF documents.
    - Generates 2-4 high-precision search queries targeting the weak assertions.
    """
    if not weak_claims:
        return [original_query]

    queries = []

    try:
        from rag.llm import load_gemini_api_key, is_valid_api_key_format, get_llm_client, get_configured_model_name, types
        key = api_key or load_gemini_api_key()
        if is_valid_api_key_format(key):
            client = llm_client if llm_client is not None else get_llm_client(key)
            model = model_name or get_configured_model_name()

            weak_claims_text = "\n".join(f"- {c}" for c in weak_claims[:4])
            prompt = (
                "You are an expert retrieval optimization assistant for an indexed document RAG system.\n"
                "The current document search failed to verify the following factual assertions in the generated answer:\n\n"
                f"ORIGINAL USER QUESTION:\n\"{original_query}\"\n\n"
                f"UNSUPPORTED / WEAK CLAIMS:\n{weak_claims_text}\n\n"
                "TASK:\n"
                "Generate 2 to 3 concise, highly-targeted search queries to locate relevant evidence or sections in the uploaded PDF documents.\n"
                "Focus on key technical nouns, entity names, concepts, and section topics. Do NOT include conversational filler.\n\n"
                "OUTPUT INSTRUCTION:\n"
                "Return ONLY a valid JSON object in this format without markdown fences:\n"
                "{\n"
                '  "queries": ["<targeted search query 1>", "<targeted search query 2>", "<targeted search query 3>"]\n'
                "}"
            )

            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=200
            ) if types else None

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )

            if response and response.text:
                raw_t = response.text.strip()
                raw_t = re.sub(r'^```(?:json)?\s*', '', raw_t, flags=re.IGNORECASE)
                raw_t = re.sub(r'\s*```$', '', raw_t).strip()
                import json
                parsed = json.loads(raw_t)
                if isinstance(parsed, dict) and "queries" in parsed:
                    for q in parsed["queries"]:
                        q_str = str(q).strip().strip('"').strip("'")
                        if q_str and q_str.lower() != original_query.lower() and q_str not in queries:
                            queries.append(q_str)
    except Exception as gen_err:
        logger.warning(f"Gemini retry query generation error: {gen_err}")

    # Fallback heuristic query generator
    if not queries:
        try:
            from rag.evaluator import extract_informative_tokens
        except ImportError:
            def extract_informative_tokens(t):
                return [w for w in re.findall(r'[a-zA-Z0-9_\-]+', t.lower()) if len(w) > 2]

        for c in weak_claims[:3]:
            words = extract_informative_tokens(c)
            if words:
                top_words = words[:4]
                q_heuristic = f"{original_query} {' '.join(top_words)}"
                if q_heuristic not in queries:
                    queries.append(q_heuristic)
            else:
                queries.append(f"{original_query} {c[:40]}")

    if not queries:
        queries = [original_query]

    # Deduplicate while preserving order, max 3 queries
    seen = set()
    deduped = []
    for q in queries:
        norm = q.lower().strip()
        if norm not in seen and norm:
            seen.add(norm)
            deduped.append(q.strip())

    return deduped[:3]


def adaptive_retrieval_pipeline(
    vectorstore: FAISS,
    query: str,
    rewritten_query: Optional[str] = None,
    selected_documents: Optional[List[str]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    last_context: Optional[List[Any]] = None,
    llm_client: Optional[Any] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    max_retries: int = 1
) -> Dict[str, Any]:
    """
    Adaptive Retrieval with Automatic Retry Orchestrator.
    Workflow:
    1. Runs Initial Multi-Query Hybrid Search + Cross-Encoder Reranking.
    2. Generates initial answer with Gemini.
    3. Runs Claim Verification & Answer Correction.
    4. Evaluates retrieval quality (supported claims, unsupported claims, faithfulness, groundedness).
    5. If low quality detected on complex queries and max_retries > 0:
       - Identifies weak/unsupported claims.
       - Generates targeted retry queries.
       - Runs Hybrid Search on new queries.
       - Merges & deduplicates with previous candidate chunks.
       - Reranks augmented candidates with Cross-Encoder.
       - Re-generates answer and re-verifies.
       - Compares attempts and retains the highest quality result.
    6. Returns complete payload with attempts history and UI process flow steps.
    """
    t_start = time.perf_counter()
    from rag.llm import generate_rag_answer, FALLBACK_RESPONSE, is_simple_query
    from rag.evaluator import evaluate_rag_response

    # --- Attempt 1: Initial Retrieval ---
    effective_q = rewritten_query if (rewritten_query and len(rewritten_query.strip()) >= len(query.strip())) else query
    retrieved_items, query_type, extracted_term, auto_rewritten, retrieval_stats = retrieve_relevant_chunks(
        vectorstore=vectorstore,
        query=effective_q,
        top_k=FINAL_TOP_K,
        candidate_pool_size=VECTOR_TOP_K,
        chat_history=chat_history,
        selected_documents=selected_documents,
        last_context=last_context,
        llm_client=llm_client,
        api_key=api_key,
        model_name=model_name
    )
    rewritten_query = auto_rewritten or effective_q

    t_retrieval_ms = retrieval_stats.get("retrieval_ms", 0.0)
    t_rerank_ms = retrieval_stats.get("reranking_ms", 0.0)

    # If no chunks found, return fallback immediately
    if not retrieved_items:
        total_sec = round(time.perf_counter() - t_start, 2)
        timing_info = {
            "retrieval_ms": t_retrieval_ms,
            "retrieval_sec": round(t_retrieval_ms / 1000.0, 2),
            "reranking_ms": t_rerank_ms,
            "reranking_sec": round(t_rerank_ms / 1000.0, 2),
            "generation_sec": 0.0,
            "verification_sec": 0.0,
            "retry_count": 0,
            "total_sec": total_sec
        }
        eval_data = evaluate_rag_response(
            query=query,
            answer=FALLBACK_RESPONSE,
            retrieved_items=[],
            response_time_sec=total_sec,
            is_fallback=True,
            retrieval_stats=retrieval_stats,
            timing_breakdown=timing_info,
            cross_encoder=_CROSS_ENCODER
        )
        process_steps = [
            {"icon": "🔎", "title": "Initial Retrieval", "desc": "0 candidates found in selected documents"},
            {"icon": "❌", "title": "Final Status", "desc": "Information not found in selected documents"}
        ]
        return {
            "answer": FALLBACK_RESPONSE,
            "draft_answer": FALLBACK_RESPONSE,
            "final_verified_answer": FALLBACK_RESPONSE,
            "retrieved_items": [],
            "sources": [],
            "retrieved_context": [],
            "documents_used": [],
            "chunks_used": 0,
            "evaluation": eval_data,
            "retrieval_stats": retrieval_stats,
            "query_type": query_type,
            "term": extracted_term,
            "rewritten_query": rewritten_query,
            "response_time": total_sec,
            "timing_breakdown": timing_info,
            "adaptive_retrieval": {
                "retries_performed": 0,
                "max_retries": max_retries,
                "triggered": False,
                "reason": "No document chunks retrieved for query",
                "attempts": [],
                "selected_attempt": 1,
                "process_steps": process_steps
            }
        }

    # --- Attempt 1: Contextual Compression Layer ---
    from rag.contextual_compression import compress_context
    t_comp_start_1 = time.perf_counter()
    compression_res_1 = compress_context(
        query=rewritten_query if rewritten_query else query,
        retrieved_chunks=retrieved_items,
        cross_encoder=_CROSS_ENCODER
    )
    t_comp_sec_1 = round(time.perf_counter() - t_comp_start_1, 3)
    t_comp_ms_1 = round(t_comp_sec_1 * 1000.0, 2)
    comp_summary_1 = compression_res_1.get("summary", {})

    # Generate Attempt 1 Answer using compressed context
    retrieved_chunks_for_llm = [(doc, score) for doc, score, _ in retrieved_items]
        
    t_gen_start = time.perf_counter()
    from rag.llm import LLMGenerationError

    try:
        answer_1 = generate_rag_answer(
            llm_client=llm_client,
            query=rewritten_query if rewritten_query else query,
            retrieved_chunks=retrieved_chunks_for_llm,
            query_type=query_type,
            extracted_term=extracted_term,
            conversation_history=chat_history,
            rewritten_query=rewritten_query,
            api_key=api_key,
            compressed_context=compression_res_1.get("formatted_context")
        )
    except Exception as gen_err:
        logger.warning(f"[ADAPTIVE_RETRIEVAL] Generation error: {gen_err}. Falling back to direct grounded evidence extraction.")
        from rag.llm import extract_direct_answer_fallback
        answer_1 = extract_direct_answer_fallback(query, retrieved_chunks_for_llm, extracted_term)

    t_gen_sec_1 = round(time.perf_counter() - t_gen_start, 2)
    t_elapsed_1 = round(time.perf_counter() - t_start, 2)

    eval_1 = evaluate_rag_response(
        query=query,
        answer=answer_1,
        retrieved_items=retrieved_items,
        response_time_sec=t_elapsed_1,
        is_fallback=False,
        retrieval_stats=retrieval_stats,
        timing_breakdown={
            "retrieval_sec": round(t_retrieval_ms / 1000.0, 2),
            "reranking_sec": round(t_rerank_ms / 1000.0, 2),
            "compression_sec": t_comp_sec_1,
            "generation_sec": t_gen_sec_1,
            "total_sec": t_elapsed_1
        },
        cross_encoder=_CROSS_ENCODER
    )

    tot_claims_1 = eval_1.get("total_claims", 0)
    sup_claims_1 = eval_1.get("supported_claims", 0)
    part_claims_1 = eval_1.get("partial_claims", 0)
    unsup_claims_1 = eval_1.get("unsupported_claims", 0)
    faith_1 = eval_1.get("faithfulness", 100.0)

    sources_attempt_1 = list(set(
        getattr(doc, "metadata", {}).get("document") or getattr(doc, "metadata", {}).get("source", "Document")
        for doc, _, _ in retrieved_items
    ))

    attempts_history = [
        {
            "attempt": 1,
            "type": "Initial Retrieval",
            "queries": retrieval_stats.get("expanded_queries", [rewritten_query]),
            "candidates_count": retrieval_stats.get("total_candidates", len(retrieved_items)),
            "final_chunks_count": len(retrieved_items),
            "sources": sources_attempt_1,
            "supported_claims": sup_claims_1,
            "partial_claims": part_claims_1,
            "unsupported_claims": unsup_claims_1,
            "total_claims": tot_claims_1,
            "faithfulness": faith_1,
            "groundedness": eval_1.get("groundedness_score", 0),
            "status": "ACCEPTED"
        }
    ]

    red_pct_1 = comp_summary_1.get("reduction_percentage", 0)
    orig_chars_1 = comp_summary_1.get("original_characters", 0)
    comp_chars_1 = comp_summary_1.get("compressed_characters", 0)

    process_steps = [
        {"icon": "🔎", "title": "Initial Retrieval", "desc": f"{retrieval_stats.get('total_candidates', len(retrieved_items))} chunks retrieved"},
        {"icon": "⚡", "title": "Reranking", "desc": f"{len(retrieved_items)} final chunks selected"},
        {"icon": "🗜️", "title": "Contextual Compression", "desc": f"{len(compression_res_1.get('compressed_chunks', []))} chunks compressed ({red_pct_1}% reduction)"},
        {"icon": "🧠", "title": "Answer Verification", "desc": f"{sup_claims_1}/{tot_claims_1 or 1} claims supported"}
    ]

    # --- Quality Decision Check ---
    is_fallback_ans = any(answer_1.strip().lower().startswith(fb) for fb in [
        "i couldn't find", "i could not find", "the selected documents do not provide",
        "the selected documents do not contain", "the answer is not available",
        "not enough supporting evidence"
    ])
    is_simple = is_simple_query(query)
    # Simple questions or answers with sufficient evidence return immediately without retrying
    needs_retry = (
        not is_fallback_ans and
        not is_simple and
        len(retrieved_items) > 0 and
        (unsup_claims_1 > 0 and sup_claims_1 == 0 and faith_1 < 50.0)
    )

    best_attempt = {
        "attempt_num": 1,
        "answer": answer_1,
        "eval_data": eval_1,
        "retrieved_items": retrieved_items,
        "compression_summary": comp_summary_1,
        "compression_chunks": compression_res_1.get("compressed_chunks", []),
        "compression_ms": t_comp_ms_1,
        "compression_sec": t_comp_sec_1,
        "score_tuple": (sup_claims_1, -unsup_claims_1, faith_1)
    }

    retries_count = 0

    if needs_retry and max_retries > 0:
        logger.info(f"[ADAPTIVE_RETRIEVAL] Low quality detected ({unsup_claims_1} unsupported claims, {faith_1}% faithfulness). Starting retry pipeline (max {max_retries})...")
        attempts_history[0]["status"] = "RETRY_TRIGGERED"
        process_steps.append({"icon": "🔄", "title": "Adaptive Retrieval", "desc": "Additional retrieval triggered"})

        # Candidate pool cache: chunk_key -> chunk_dict
        candidate_pool_map = {}
        for doc, sc, r_pct in retrieved_items:
            d_name = getattr(doc, "metadata", {}).get("document") or getattr(doc, "metadata", {}).get("source", "Document")
            p_num = int(getattr(doc, "metadata", {}).get("page", 1))
            c_id = str(getattr(doc, "metadata", {}).get("chunk_id", f"{d_name}_p{p_num}"))
            key = (d_name, p_num, c_id)
            candidate_pool_map[key] = {
                "doc": doc,
                "document_name": d_name,
                "document_id": d_name,
                "page_number": p_num,
                "chunk_id": c_id,
                "text": doc.page_content if hasattr(doc, "page_content") else str(doc),
                "vector_score": float(sc),
                "bm25_score": 0.5,
                "normalized_vector_score": float(sc),
                "normalized_bm25_score": 0.5,
                "hybrid_score": float(sc)
            }

        # --- Retry Loop (up to max_retries) ---
        current_eval = eval_1
        for r_idx in range(1, max_retries + 1):
            retries_count += 1
            weak_claims = [
                c.get("original_claim") or c.get("claim")
                for c in current_eval.get("claims", [])
                if c.get("status") in ("NOT_SUPPORTED", "PARTIALLY_SUPPORTED")
            ]

            if not weak_claims:
                break

            logger.info(f"[ADAPTIVE_RETRIEVAL] Retry Attempt {r_idx}: Weak claims to resolve: {len(weak_claims)}")
            retry_queries = generate_adaptive_retry_queries(
                original_query=rewritten_query if rewritten_query else query,
                weak_claims=weak_claims,
                llm_client=llm_client,
                api_key=api_key,
                model_name=model_name
            )
            logger.info(f"[ADAPTIVE_RETRIEVAL] Retry queries generated ({len(retry_queries)}): {retry_queries}")

            # Run parallel Hybrid Search on new targeted queries
            new_candidates, mq_stats = multi_query_hybrid_search(
                vectorstore=vectorstore,
                queries=retry_queries,
                selected_documents=selected_documents,
                vector_top_k=10,
                bm25_top_k=10,
                vector_weight=VECTOR_WEIGHT,
                bm25_weight=BM25_WEIGHT,
                prior_topic=extracted_term
            )

            # Merge and deduplicate candidates
            added_new_count = 0
            for nc in new_candidates:
                d_name = nc["document_name"]
                p_num = int(nc["page_number"])
                c_id = str(nc.get("chunk_id", f"{d_name}_p{p_num}"))
                key = (d_name, p_num, c_id)
                if key not in candidate_pool_map:
                    candidate_pool_map[key] = nc
                    added_new_count += 1
                else:
                    existing = candidate_pool_map[key]
                    existing["hybrid_score"] = max(existing.get("hybrid_score", 0.0), nc.get("hybrid_score", 0.0))

            all_candidates = list(candidate_pool_map.values())

            # Cross-Encoder Reranking on merged candidates
            reranked_top = rerank_chunks(
                query=rewritten_query if rewritten_query else query,
                candidate_chunks=all_candidates,
                extracted_term=extracted_term,
                prior_topic=extracted_term,
                top_k=FINAL_TOP_K
            )
            retrieved_items_retry = [(res["doc"], res["final_score"], res["relevance_pct"]) for res in reranked_top]

            # Contextual Compression for Retry Attempt
            t_retry_comp_start = time.perf_counter()
            compression_res_retry = compress_context(
                query=rewritten_query if rewritten_query else query,
                retrieved_chunks=retrieved_items_retry,
                cross_encoder=_CROSS_ENCODER
            )
            t_retry_comp_sec = round(time.perf_counter() - t_retry_comp_start, 3)
            t_retry_comp_ms = round(t_retry_comp_sec * 1000.0, 2)
            comp_summary_retry = compression_res_retry.get("summary", {})

            # Re-generate answer with Gemini using the compressed context
            chunks_for_llm_retry = [(doc, score) for doc, score, _ in retrieved_items_retry]

            t_retry_gen_start = time.perf_counter()
            try:
                answer_retry = generate_rag_answer(
                    llm_client=llm_client,
                    query=rewritten_query if rewritten_query else query,
                    retrieved_chunks=chunks_for_llm_retry,
                    query_type=query_type,
                    extracted_term=extracted_term,
                    conversation_history=chat_history,
                    rewritten_query=rewritten_query,
                    api_key=api_key,
                    compressed_context=compression_res_retry.get("formatted_context")
                )
            except Exception as retry_err:
                logger.warning(f"[ADAPTIVE_RETRIEVAL] Retry {r_idx} LLM generation failed: {retry_err}. Retaining previous best attempt.")
                break
            t_retry_gen_sec = round(time.perf_counter() - t_retry_gen_start, 2)
            t_retry_elapsed = round(time.perf_counter() - t_start, 2)

            # Re-verify and evaluate
            eval_retry = evaluate_rag_response(
                query=query,
                answer=answer_retry,
                retrieved_items=retrieved_items_retry,
                response_time_sec=t_retry_elapsed,
                is_fallback=False,
                retrieval_stats=retrieval_stats,
                timing_breakdown={
                    "retrieval_sec": round(t_retrieval_ms / 1000.0, 2),
                    "reranking_sec": round(t_rerank_ms / 1000.0, 2),
                    "compression_sec": t_retry_comp_sec,
                    "generation_sec": t_retry_gen_sec,
                    "total_sec": t_retry_elapsed
                },
                cross_encoder=_CROSS_ENCODER
            )

            tot_claims_r = eval_retry.get("total_claims", 0)
            sup_claims_r = eval_retry.get("supported_claims", 0)
            part_claims_r = eval_retry.get("partial_claims", 0)
            unsup_claims_r = eval_retry.get("unsupported_claims", 0)
            faith_r = eval_retry.get("faithfulness", 0.0)

            sources_attempt_r = list(set(
                getattr(doc, "metadata", {}).get("document") or getattr(doc, "metadata", {}).get("source", "Document")
                for doc, _, _ in retrieved_items_retry
            ))

            process_steps.append({"icon": "🔎", "title": f"Retry Retrieval {r_idx}", "desc": f"{added_new_count} additional candidates found"})
            process_steps.append({"icon": "🗜️", "title": f"Contextual Compression {r_idx + 1}", "desc": f"{len(compression_res_retry.get('compressed_chunks', []))} chunks compressed ({comp_summary_retry.get('reduction_percentage', 0)}% reduction)"})
            process_steps.append({"icon": "🧠", "title": f"Verification {r_idx + 1}", "desc": f"{sup_claims_r}/{tot_claims_r or 1} claims supported"})

            score_tuple_r = (sup_claims_r, -unsup_claims_r, faith_r)
            is_better = score_tuple_r > best_attempt["score_tuple"]

            attempt_record = {
                "attempt": r_idx + 1,
                "type": f"Adaptive Retry {r_idx}",
                "queries": retry_queries,
                "weak_claims": weak_claims,
                "candidates_count": len(all_candidates),
                "additional_chunks_found": added_new_count,
                "final_chunks_count": len(retrieved_items_retry),
                "sources": sources_attempt_r,
                "supported_claims": sup_claims_r,
                "partial_claims": part_claims_r,
                "unsupported_claims": unsup_claims_r,
                "total_claims": tot_claims_r,
                "faithfulness": faith_r,
                "groundedness": eval_retry.get("groundedness_score", 0),
                "status": "BETTER_RESULT" if is_better else "RETAINED_PREVIOUS"
            }
            attempts_history.append(attempt_record)

            if is_better:
                logger.info(f"[ADAPTIVE_RETRIEVAL] Retry {r_idx} improved answer: {sup_claims_r}/{tot_claims_r} supported ({faith_r}% faithful). Adopting.")
                best_attempt = {
                    "attempt_num": r_idx + 1,
                    "answer": answer_retry,
                    "eval_data": eval_retry,
                    "retrieved_items": retrieved_items_retry,
                    "compression_summary": comp_summary_retry,
                    "compression_chunks": compression_res_retry.get("compressed_chunks", []),
                    "compression_ms": t_retry_comp_ms,
                    "compression_sec": t_retry_comp_sec,
                    "score_tuple": score_tuple_r
                }
                current_eval = eval_retry

            # Early exit if perfect verification achieved
            if unsup_claims_r == 0 and faith_r >= 85.0:
                logger.info(f"[ADAPTIVE_RETRIEVAL] Target quality achieved on retry {r_idx}. Halting retries.")
                break

    # Build final process status tag
    final_sup = best_attempt["eval_data"].get("supported_claims", 0)
    final_tot = best_attempt["eval_data"].get("total_claims", 0)
    final_risk = best_attempt["eval_data"].get("risk_level", "LOW")

    if final_risk == "LOW" and (final_tot == 0 or final_sup > 0):
        process_steps.append({"icon": "✅", "title": "Final Status", "desc": "High-confidence evidence-grounded answer"})
    elif final_risk == "MEDIUM":
        process_steps.append({"icon": "⚠️", "title": "Final Status", "desc": "Partially verified evidence answer"})
    else:
        process_steps.append({"icon": "ℹ️", "title": "Final Status", "desc": "Evidence-filtered verified answer"})

    # Prepare final deduplicated source cards list
    final_items = best_attempt["retrieved_items"]
    raw_sources = []
    retrieved_context_data = []
    retrieved_tables = []

    primary_doc_name = retrieval_stats.get("primary_document", final_items[0][0].metadata.get("document", "Document") if final_items else "Document")

    for item_idx, item in enumerate(final_items):
        doc = item[0]
        f_score = item[1]
        rel_pct = item[2] if len(item) > 2 else 90
        doc_meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        doc_name = doc_meta.get("document") or doc_meta.get("source", "Document")
        page_num = doc_meta.get("page", 1)
        chunk_content = doc.page_content.strip() if hasattr(doc, "page_content") else str(doc).strip()
        
        is_tbl = bool(doc_meta.get("is_table") or chunk_content.startswith("[Table]"))
        tbl_num = doc_meta.get("table_number")
        tbl_md = doc_meta.get("table_markdown", "")
        tbl_cols = doc_meta.get("columns", [])
        tbl_rows = doc_meta.get("table_rows", [])

        doc_score_val = doc_meta.get("document_score", f_score)
        doc_rel_pct_val = doc_meta.get("document_relevance_pct", rel_pct)

        source_item = {
            "document_id": doc_name,
            "document": doc_name,
            "filename": doc_name,
            "page": page_num,
            "relevance_pct": rel_pct,
            "score": round(float(f_score), 3),
            "document_score": round(float(doc_score_val), 3),
            "document_relevance_pct": int(doc_rel_pct_val),
            "is_primary": (item_idx == 0 or doc_name == primary_doc_name),
            "pdf_url": f"/view_pdf/{doc_name}#page={page_num}",
            "snippet": chunk_content[:500] + ("..." if len(chunk_content) > 500 else ""),
            "excerpt": chunk_content[:500] + ("..." if len(chunk_content) > 500 else ""),
            "is_table": is_tbl
        }

        if is_tbl:
            source_item["table_number"] = tbl_num
            source_item["table_markdown"] = tbl_md
            source_item["columns"] = tbl_cols
            source_item["table_rows"] = tbl_rows
            source_item["citation_label"] = f"{doc_name} — Page {page_num} (Table {tbl_num})" if tbl_num else f"{doc_name} — Page {page_num} (Table)"

            # Filter relevant rows for query
            try:
                from rag.table_extractor import filter_table_rows_for_query
                relevant_rows = filter_table_rows_for_query(query, {"columns": tbl_cols, "rows": tbl_rows}, max_rows=8)
            except Exception:
                relevant_rows = tbl_rows[:8]

            table_entry = {
                "document": doc_name,
                "filename": doc_name,
                "page": page_num,
                "table_number": tbl_num or 1,
                "columns": tbl_cols,
                "rows": relevant_rows if relevant_rows else tbl_rows[:8],
                "num_rows": len(tbl_rows),
                "num_cols": len(tbl_cols),
                "markdown": tbl_md,
                "citation": f"{doc_name} — Page {page_num} (Table {tbl_num})" if tbl_num else f"{doc_name} — Page {page_num} (Table)"
            }
            retrieved_tables.append(table_entry)

        raw_sources.append(source_item)

        retrieved_context_data.append({
            "document_id": doc_name,
            "document": doc_name,
            "filename": doc_name,
            "page": page_num,
            "text": chunk_content[:600],
            "pdf_url": f"/view_pdf/{doc_name}#page={page_num}",
            "is_table": is_tbl,
            "table_number": tbl_num
        })

    if retrieved_tables:
        logger.info(f"TABLES RETRIEVED: {len(retrieved_tables)} table(s) selected.")
        for t in retrieved_tables:
            logger.info(f"TABLE SOURCE: {t['document']} | TABLE PAGE: {t['page']} | TABLE ROWS USED: {len(t.get('rows', []))}")

    # Deduplicate sources preserving order
    seen_sources = set()
    deduped_sources = []
    for src in raw_sources:
        key = (src["document"], src["page"], src.get("table_number"))
        if key not in seen_sources:
            seen_sources.add(key)
            deduped_sources.append(src)

    # Build grouped sources for UI
    grouped_sources_list = build_grouped_sources_payload(deduped_sources, primary_doc=primary_doc_name)

    total_sec_final = round(time.perf_counter() - t_start, 2)
    final_eval_data = best_attempt["eval_data"]
    final_ans_text = final_eval_data.get("final_verified_answer") or best_attempt["answer"]
    draft_ans_text = final_eval_data.get("draft_answer") or best_attempt["answer"]

    timing_breakdown_final = {
        "retrieval_ms": t_retrieval_ms,
        "retrieval_sec": round(t_retrieval_ms / 1000.0, 2),
        "reranking_ms": t_rerank_ms,
        "reranking_sec": round(t_rerank_ms / 1000.0, 2),
        "compression_ms": best_attempt.get("compression_ms", t_comp_ms_1),
        "compression_sec": best_attempt.get("compression_sec", t_comp_sec_1),
        "generation_sec": t_gen_sec_1,
        "verification_sec": final_eval_data.get("timing_breakdown", {}).get("verification_sec", 0.0),
        "retry_count": retries_count,
        "total_sec": total_sec_final
    }

    final_eval_data["timing_breakdown"] = timing_breakdown_final

    adaptive_payload = {
        "retries_performed": retries_count,
        "max_retries": max_retries,
        "triggered": bool(retries_count > 0),
        "reason": f"Resolved {unsup_claims_1} unsupported claims via targeted retry" if retries_count > 0 else "Initial retrieval met high-confidence threshold",
        "attempts": attempts_history,
        "selected_attempt": best_attempt["attempt_num"],
        "process_steps": process_steps
    }

    documents_used = list(set(src["document"] for src in deduped_sources))

    structured_data_payload = {
        "tables_detected": len(retrieved_tables),
        "tables_retrieved": len(retrieved_tables),
        "tables": retrieved_tables
    }

    return {
        "answer": final_ans_text,
        "draft_answer": draft_ans_text,
        "final_verified_answer": final_ans_text,
        "answer_correction": final_eval_data.get("answer_correction", {}),
        "retrieved_items": final_items,
        "sources": deduped_sources,
        "grouped_sources": grouped_sources_list,
        "primary_source": retrieval_stats.get("primary_source", f"{primary_doc_name} — Page {deduped_sources[0]['page']}" if deduped_sources else "None"),
        "retrieved_context": retrieved_context_data,
        "documents_used": documents_used,
        "chunks_used": len(final_items),
        "evaluation": final_eval_data,
        "retrieval_stats": retrieval_stats,
        "query_type": query_type,
        "term": extracted_term,
        "rewritten_query": rewritten_query,
        "response_time": total_sec_final,
        "timing_breakdown": timing_breakdown_final,
        "compression": best_attempt.get("compression_summary", comp_summary_1),
        "compression_details": best_attempt.get("compression_chunks", compression_res_1.get("compressed_chunks", [])),
        "adaptive_retrieval": adaptive_payload,
        "structured_data": structured_data_payload
    }



