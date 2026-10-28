-- =============================================================================
-- Supabase PostgreSQL Schema with pgvector for Intelligent Document RAG
-- Supports persistent document metadata, 384-dimensional chunk embeddings,
-- hybrid search (dense vector + lexical tsv), durable indexing jobs, and analytics.
-- =============================================================================

-- 1. Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Document Metadata Table (replaces local vectorstore/metadata.json)
CREATE TABLE IF NOT EXISTS documents_metadata (
    document_id TEXT PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Pending',   -- 'Pending', 'Indexing', 'Ready', 'Error'
    pages INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER NOT NULL DEFAULT 0,
    size_kb REAL NOT NULL DEFAULT 0.0,
    file_size BIGINT NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL,
    indexed_at TEXT,
    error_message TEXT
);

-- Index on filename for fast lookup and deduplication
CREATE INDEX IF NOT EXISTS idx_doc_metadata_filename ON documents_metadata (filename);
CREATE INDEX IF NOT EXISTS idx_doc_metadata_status ON documents_metadata (status);

-- 3. Document Chunks & 384-dim Embeddings Table (replaces local index.faiss)
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents_metadata(document_id) ON DELETE CASCADE,
    document_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),                     -- 384-dim matching all-MiniLM-L6-v2 & Gemini 384
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for filtering by document
CREATE INDEX IF NOT EXISTS idx_chunks_document_name ON document_chunks (document_name);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks (document_id);

-- Vector Cosine Similarity Search Index (HNSW for high-performance sub-millisecond retrieval)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Full-Text Search Index for Lexical / Keyword Search (Hybrid retrieval)
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON document_chunks USING gin (to_tsvector('english', content));

-- 4. Durable Indexing Jobs Table (replaces local indexing_jobs.json)
CREATE TABLE IF NOT EXISTS indexing_jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'indexing',    -- 'indexing', 'completed', 'error'
    stage TEXT NOT NULL DEFAULT 'starting',     -- 'starting', 'extracting', 'chunking', 'embedding', 'completed', 'failed'
    progress INTEGER NOT NULL DEFAULT 0,
    message TEXT,
    documents JSONB DEFAULT '[]'::jsonb,
    indexed_count INTEGER DEFAULT 0,
    failed_count INTEGER DEFAULT 0,
    failed_details JSONB DEFAULT '[]'::jsonb,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- 5. Interactions & Analytics Table (replaces local SQLite analytics.db)
CREATE TABLE IF NOT EXISTS interactions (
    id BIGSERIAL PRIMARY KEY,
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
    answer TEXT DEFAULT '',
    grounded_citations TEXT DEFAULT '[]',
    session_id TEXT DEFAULT ''
);

-- 6. User Feedback Table
CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    interaction_id INTEGER,
    rating TEXT NOT NULL,
    comment TEXT DEFAULT '',
    timestamp TEXT NOT NULL
);

-- 7. High-Performance Vector Similarity Search RPC Function
CREATE OR REPLACE FUNCTION match_chunks (
    query_embedding vector(384),
    match_count int DEFAULT 10,
    filter_documents text[] DEFAULT NULL
) RETURNS TABLE (
    id bigint,
    document_id text,
    document_name text,
    chunk_index int,
    page_number int,
    content text,
    metadata jsonb,
    similarity float
) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.document_name,
        c.chunk_index,
        c.page_number,
        c.content,
        c.metadata,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM document_chunks c
    WHERE (filter_documents IS NULL OR c.document_name = ANY(filter_documents))
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 8. Lexical / Keyword Search RPC Function (for Hybrid Retrieval)
CREATE OR REPLACE FUNCTION match_chunks_lexical (
    query_text text,
    match_count int DEFAULT 10,
    filter_documents text[] DEFAULT NULL
) RETURNS TABLE (
    id bigint,
    document_id text,
    document_name text,
    chunk_index int,
    page_number int,
    content text,
    metadata jsonb,
    rank float
) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.document_name,
        c.chunk_index,
        c.page_number,
        c.content,
        c.metadata,
        ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text))::float AS rank
    FROM document_chunks c
    WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
      AND (filter_documents IS NULL OR c.document_name = ANY(filter_documents))
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

