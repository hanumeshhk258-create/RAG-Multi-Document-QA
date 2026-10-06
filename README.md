# RAG Multi-Document Question Answering System

A production-grade, document-grounded Retrieval-Augmented Generation (RAG) web application built with **Python**, **Flask**, **FAISS**, **BM25**, **Sentence Transformers**, **Cross-Encoder Rerankers**, and the **Google Gemini API**.

---

## Overview

The **RAG Multi-Document Question Answering System** enables users to upload single or multiple PDF documents and perform fast, accurate, hallucination-resistant question answering. Rather than relying on generic model weights, the system grounds every generated claim in retrieved document context, verifies evidence before delivery, and cites exact source pages.

The application features a modern multi-workspace single-page interface with a collapsible navigation sidebar, real-time document search, conversational session management, an audit history trail, an analytics dashboard, an interactive "How RAG Works" educational walkthrough, and a dual-theme visual system (Light & Dark).

---

## Key Features

- **Multi-PDF Document Management**: Drag-and-drop or browse multiple PDF files simultaneously, tracking per-document page counts, chunk counts, and indexing readiness.
- **Hybrid Retrieval (Dense + Sparse)**: Combines semantic vector similarity search via **FAISS** with exact keyword matching via **Okapi BM25**, fused via **Reciprocal Rank Fusion (RRF)**.
- **Neural Cross-Encoder Reranking**: Re-scores top retrieval candidates with `ms-marco-MiniLM-L-6-v2` cross-attention, filtering out irrelevant chunks before LLM synthesis.
- **Contextual Compression & Deduplication**: Intelligently compresses retrieved chunks to remove redundant whitespace and text while preserving semantic density and exact metadata.
- **Grounded Generation & Verification**: Synthesizes answers strictly using **Google Gemini** (`gemini-3.5-flash-lite`), followed by automated claim verification to detect and prevent hallucinations.
- **Click-to-Verify Page Citations**: Every factual claim is attributed with structured citations (`[Document.pdf — Page X]`), opening an evidence inspector displaying the exact source excerpt and similarity score.
- **Balanced Multi-Document Comparison**: Dedicated balanced retrieval mode that allocates fair candidate pools across selected documents, preventing document starvation when comparing multiple PDFs.
- **Conversational Follow-Up Memory**: Contextual query re-writing resolves pronouns and ellipsis across chat turns (e.g., *"What is its timeout?"* -> *"What is the timeout for AcmeOS?"*).
- **Safe Out-of-Scope Refusal**: Queries asking for information absent from the indexed documents produce a safe, grounded fallback response rather than hallucinated external knowledge.
- **Browser-Saved Conversation Sessions**: Manage distinct conversation threads with instant client-side search, top pinning, inline renaming, and safe deletion.
- **Question History Audit Trail**: Chronological audit log of all answered queries with shortcuts to re-ask in Chat or explore passages in Search.
- **Analytics & Accuracy Metrics**: Real-time performance statistics, 12 summary cards, accuracy benchmarks (Hit@1, Hit@3, Groundedness), and full question history table with CSV export (charts cleanly decommissioned).
- **Interactive "How RAG Works" Workspace**: Interactive 9-stage visual architecture flow, live 4-step pipeline simulation stepper, and safe refusal guarantee.
- **Premium Dual-Theme System**: Warm Light Theme (`#064E3B` / `#F8E7C9`) and High-Contrast Dark Theme (`#23262F` / `#B6FF2F`) with smooth View Transition animations.
- **Reliable Error Handling**: Actionable error reporting that never mislabels client-side validation or backend errors as AI service failures.

---

## RAG Pipeline Architecture

```
[ Uploaded PDF Documents ]
            │
            ▼
[ Document Ingestion & Table Extraction (PyPDF) ]
            │
            ▼
[ Recursive Semantic Chunking ] (500 chars, 50 overlap, page metadata)
            │
     ┌──────┴─────────────────────────┐
     ▼                                ▼
[ Dense Embeddings ]         [ Sparse Term Frequency ]
(all-MiniLM-L6-v2, 384d)     (Okapi BM25 Index)
     │                                │
     ▼                                ▼
[ FAISS Vector Store ]       [ BM25 Keyword Search ]
     └──────┬─────────────────────────┘
            ▼
[ Hybrid Retrieval & RRF Fusion ] (k = 60)
            │
            ▼
[ Cross-Encoder Neural Reranking ] (ms-marco-MiniLM-L-6-v2)
            │
            ▼
[ Contextual Compression & Assembly ]
            │
            ▼
[ Grounded LLM Synthesis ] (Google Gemini API: gemini-3.5-flash-lite)
            │
            ▼
[ Automated Claim Verification & Hallucination Correction ]
            │
            ▼
[ Final Grounded Answer + Click-to-Verify Page Citations ]
```

---

## Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Backend Framework** | Python 3.10+, Flask | Modular REST API and static asset server |
| **Document Parsing** | PyPDF | Page-by-page text and table extraction |
| **Embeddings** | Hugging Face Sentence Transformers | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| **Dense Vector Index** | FAISS (CPU) | L2-normalized dense vector similarity search |
| **Sparse Index** | Rank-BM25 | Tokenized Okapi BM25 keyword index |
| **Neural Reranking** | Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Generative LLM** | Google Gemini API | `gemini-3.5-flash-lite` via `google-genai` SDK |
| **Frontend** | Vanilla JavaScript (ES6+), HTML5, CSS3 | Single-page multi-workspace architecture |
| **Styling & Themes** | CSS Custom Properties, View Transitions API | Dual-theme system (Light `#064E3B` / Dark `#23262F`) |
| **Testing** | unittest, pytest, Node.js | Unit, integration, reliability, and regression suites |

---

## Project Structure

```
RAG/
├── app.py                            # Flask server, REST API endpoints, state management
├── requirements.txt                  # Python package dependencies
├── .env.example                      # Environment template for Gemini API key
├── .gitignore                        # Git ignore rules (.env, venv, cache, vectorstore)
├── README.md                         # Project documentation
│
├── rag/                              # Core RAG pipeline modules
│   ├── __init__.py                   # Package initializer
│   ├── pdf_loader.py                 # PDF extraction, page metadata, corruption handling
│   ├── chunker.py                    # Recursive character text splitting
│   ├── embeddings.py                 # Hugging Face sentence-transformers loader
│   ├── vectorstore.py                # FAISS vector store creation, persistence & sync
│   ├── retriever.py                  # Hybrid retrieval, RRF fusion, Cross-Encoder reranking
│   ├── contextual_compression.py     # Chunk compression and redundant text reduction
│   ├── table_extractor.py            # Heuristic table structure extraction
│   ├── llm.py                        # Gemini API client, retries, claim verification, fallback
│   ├── analytics.py                  # SQLite analytics logging & performance metrics
│   ├── evaluator.py                  # Groundedness, relevance, and accuracy scoring
│   └── evaluation_runner.py          # Benchmark suite runner
│
├── templates/
│   └── index.html                    # Single-page multi-workspace HTML template
│
├── static/
│   ├── css/
│   │   └── style.css                 # Comprehensive dual-theme stylesheet and responsive styles
│   └── js/
│       └── app.js                    # Frontend workspace routing, chat, sessions, and animations
│
├── documents/                        # Local storage directory for active uploaded PDFs
├── vectorstore/                      # Local storage directory for FAISS index and metadata
│
└── tests/                            # Automated test suites
    ├── test_rag_pipeline.py          # 12-suite end-to-end pipeline verification
    ├── test_rag_reliability.py       # Deterministic reliability, mock failures & edge cases
    ├── test_balanced_comparison.py   # Multi-document starvation prevention & balanced retrieval
    ├── test_e2e_user_workflow.py     # 8-phase full user lifecycle regression test
    └── test_frontend_error_handling.js # JavaScript error mapping and API contract unit tests
```

---

## Installation & Setup

### Prerequisites

- **Python 3.10+** (recommended: 3.10 or 3.11)
- **Node.js 18+** (optional, for frontend unit tests)
- A **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/)

### Step 1: Clone the Repository

```bash
git clone https://github.com/sharathgowdaur-jpg/RAG-Multi-Document-QA.git
cd RAG-Multi-Document-QA
```

### Step 2: Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the root directory by copying `.env.example`:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and add your Gemini API Key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

> **Security Note:** Never commit your `.env` file. It is excluded by `.gitignore`.

---

## Running the Application

Start the Flask application server:

```bash
python app.py
```

The application starts at:
```
http://127.0.0.1:5000/
```

Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## Using the Application

1. **Dashboard**: Review collection statistics, recent document previews, and launch quick actions.
2. **Documents**: Drag-and-drop or select one or more PDF files. Click **Index Documents** to extract text, generate embeddings, and build the hybrid index.
3. **Search & Explore**: Perform instant keyword or concept queries across your entire indexed document library with page-level relevance scores.
4. **Chat**: Ask natural-language questions. View answers with click-to-preview page citations, claim verification badges, and retrieval diagnostics. Ask follow-up questions naturally.
5. **Multi-Document Comparison**: Select two or more documents and ask comparison questions (e.g., *"Compare authentication and session timeout between Doc A and Doc B"*). The balanced retrieval engine ensures fair representation.
6. **Sessions**: Create new conversation threads, search past sessions, pin important threads to the top, rename sessions, or delete them safely.
7. **History**: Review past questions and click to re-ask or inspect retrieved chunks.
8. **Analytics**: Inspect query volume, average response times, grounding percentages, evaluation benchmarks, and export interaction history as CSV.
9. **How RAG Works**: Explore the interactive 9-stage pipeline diagram and click through the step-by-step simulation to see how your documents become grounded answers.

---

## Workspaces Overview

| Workspace | Hash Route | Description |
|---|---|---|
| **Dashboard** | `#dashboard` | System overview, document counters, and quick navigation cards |
| **Documents** | `#documents` | Upload, index, rebuild, and delete PDF documents |
| **Search & Analyze** | `#search` | Real-time passage and keyword search across all indexed chunks |
| **Chat** | `#chat` | Conversational document Q&A with citations and follow-up memory |
| **Sessions** | `#sessions` | Browser-saved conversation management with search and pinning |
| **History** | `#history` | Audit trail of past questions with re-query shortcuts |
| **Analytics** | `#analytics` | Real-time performance metrics, evaluation benchmark, and CSV export |
| **How RAG Works** | `#rag` | Interactive 9-stage pipeline educational walkthrough & demo stepper |

---

## Running Automated Tests

The repository includes a comprehensive automated test suite covering the RAG pipeline, reliability under transient failures, balanced multi-document comparison, end-to-end user workflows, and frontend error handling:

### 1. RAG Pipeline Core Suite (12 Tests)
Tests PDF extraction, corrupted file handling, chunking, embeddings, FAISS build/save/incremental additions, semantic retrieval, and Flask REST API endpoints in an isolated temporary environment:
```bash
python tests/test_rag_pipeline.py
```

### 2. Deterministic Reliability & Failure Mode Suite (15 Tests)
Simulates Gemini 401 Auth, 429 Rate Limit, 503 Unavailable, Timeout, and Empty responses via mocks; verifies non-retryable vs retryable classification, input validation, and thread concurrency:
```bash
python -m unittest tests/test_rag_reliability.py
```

### 3. Balanced Multi-Document Comparison Regression Suite (5 Tests)
Verifies fair per-document candidate allocation, document starvation prevention, comparison intent classification, and relevance filtering:
```bash
python tests/test_balanced_comparison.py
```

### 4. Full End-to-End User Lifecycle Suite (8 Phases)
Tests baseline health, single PDF upload, factual Q&A, follow-up memory, unsupported negative query refusal, multi-PDF comparison, duplicate upload rejection, and document deletion sync against the running server:
```bash
python tests/test_e2e_user_workflow.py
```

### 5. Frontend Error Handling Contract Suite
Verifies client-side error code mapping, fallback handling, and HTTP status code extraction:
```bash
node tests/test_frontend_error_handling.js
```

---

## Reliability & Error Handling

- **Actionable User Guidance**: When documents are not indexed, the system surfaces clear instructions rather than generic technical errors.
- **Strict Error Classification**: Non-retryable errors (e.g., invalid API key, malformed input) fail immediately with clear diagnostic messages. Transient errors (rate limits, timeouts) trigger exponential backoff.
- **Accurate Error Attribution**: Client-side network disconnects and HTTP validation errors are never mislabeled as "AI service unavailable".
- **Zero-Hallucination Safe Refusal**: When a question cannot be answered from the retrieved document context, the system returns a safe, grounded refusal without fabricating facts.
- **State Synchronization**: Deleting a document removes its chunks from the FAISS index and BM25 cache, preventing deleted content from leaking into future answers.

---

## Known Limitations

- **Local / Single-User Design**: The application runs as a local Flask development server. Multi-tenant deployments require a production WSGI server (e.g., Gunicorn or Waitress) and user-scoped data directories.
- **Browser-Local Session Persistence**: Chat sessions are persisted in browser local storage (`localStorage`). Clearing browser site data will reset saved session threads.
- **Digital Text Extraction**: PyPDF extracts embedded text streams. Non-OCR scanned PDFs or image-only documents require pre-processing with an OCR tool before indexing.
- **External Model Dependency**: Final response synthesis requires access to the Google Gemini API. When offline, direct grounded extraction fallback provides retrieved excerpts without generative synthesis.

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for document and text splitting primitives.
- [FAISS](https://github.com/facebookresearch/faiss) by Meta Research for fast vector similarity search.
- [Sentence Transformers](https://www.sbert.net/) for local dense embeddings (`all-MiniLM-L6-v2`) and cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`).
- [Rank-BM25](https://github.com/dorianbrown/rank_bm25) for sparse lexical keyword retrieval.
- [Google Gemini API](https://ai.google.dev/) for multimodal LLM synthesis.
