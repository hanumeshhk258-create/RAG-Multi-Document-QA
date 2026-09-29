# 📄 Intelligent Document Question Answering System Using RAG

An end-to-end, beginner-friendly Retrieval-Augmented Generation (RAG) web application built with **Python**, **Streamlit**, **LangChain**, **FAISS**, **Hugging Face Sentence Transformers**, and the **Google Gemini API**.

---

## 📌 1. Project Overview

The **Intelligent Document Question Answering System** enables users to upload multiple PDF documents and ask questions about their content. Rather than relying on static pre-trained knowledge or suffering from hallucination, the system utilizes **Retrieval-Augmented Generation (RAG)**:
1. It extracts text page-by-page from uploaded PDFs.
2. It splits text into manageable chunks.
3. It generates vector embeddings using a local Hugging Face model (`all-MiniLM-L6-v2`).
4. It indexes the embeddings in a **FAISS** vector database.
5. When a user asks a question, the system retrieves the most relevant document chunks and provides them as context to the **Google Gemini LLM** to generate an accurate, grounded answer with page citations.

---

## ✨ 2. Key Features

* **Multi-PDF Support:** Upload single or multiple PDF documents simultaneously.
* **Page-Aware Extraction:** Preserves exact PDF page numbers and document filenames in chunk metadata.
* **Local Embedding Generation:** Free, offline embedding generation using Hugging Face's `all-MiniLM-L6-v2` model.
* **Fast Vector Search:** In-memory & persisted vector database powered by FAISS.
* **Strict Hallucination Prevention:** The LLM prompt is restricted to answer *only* from retrieved document context. If information is absent, it responds strictly with:  
  `"I could not find this information in the uploaded documents."`
* **Source Citations:** Expandable citations displaying document name, page number, similarity score, and retrieved text snippets for every answer.
* **ChatGPT-Style Interface:** Seamless conversational UI with session history and a "Clear Chat" button.
* **Interactive Sidebar Controls:** Adjust chunk size, chunk overlap, and Top-K retrieved chunks dynamically.

---

## 🛠️ 3. Technology Stack

* **Python 3.10+ / 3.11+**
* **Streamlit:** Interactive web UI frontend
* **LangChain:** Framework for RAG pipeline orchestration
* **PyPDF:** PDF text extraction
* **Hugging Face Sentence Transformers:** Local embedding model (`all-MiniLM-L6-v2`)
* **FAISS (CPU):** Vector database for fast similarity search
* **Google Gemini API (`gemini-1.5-flash`):** LLM for answer synthesis
* **python-dotenv:** Environment variable management

---

## 📐 4. System Architecture & Workflow

```
[ PDF Upload ]
       │
       ▼
[ Extract Text (PyPDF) ] ── (Preserves filename & page numbers)
       │
       ▼
[ Chunk Text (RecursiveCharacterTextSplitter) ]
       │
       ▼
[ Generate Embeddings (Sentence Transformers) ]
       │
       ▼
[ Store Embeddings in FAISS Vectorstore ]
       │
       ▼
[ User Asks Question ] ──► [ Vector Similarity Search (Top-K) ]
                                   │
                                   ▼
                    [ Send Query + Retrieved Context ]
                                   │
                                   ▼
                        [ Google Gemini API LLM ]
                                   │
                                   ▼
                     [ Grounded Answer + Source Citations ]
```

---

## 📁 5. Project Structure

```
RAG_Document_QA/
│
├── app.py                  # Main Streamlit Web Application UI
├── requirements.txt        # Python package dependencies
├── .env.example            # Environment template for Gemini API key
├── .gitignore              # Git ignore configuration
├── README.md               # Comprehensive documentation
│
├── rag/                    # Core RAG pipeline package
│   ├── __init__.py         # Package initializer
│   ├── pdf_loader.py       # PDF text extraction & page metadata handling
│   ├── chunker.py          # Document text splitting & chunk metadata assignment
│   ├── embeddings.py       # Modular Hugging Face embedding model loader
│   ├── vectorstore.py      # FAISS index creation, persistence & clearing
│   ├── retriever.py        # Similarity search & Top-K chunk retrieval
│   └── llm.py              # Gemini LLM setup & hallucination prevention prompt
│
├── documents/              # Storage directory for sample documents
└── vectorstore/            # Persisted FAISS vector index files
```

---

## ⚙️ 6. Installation & Setup Instructions

Follow these steps to set up and run the project locally on your machine.

### Step 1: Clone or Navigate to the Project Directory
```bash
cd c:\Users\hanum\Downloads\RAG
```

### Step 2: Create a Virtual Environment
```bash
# Windows
python -m venv venv

# Activate Virtual Environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Or Windows Command Prompt (cmd)
.\venv\Scripts\activate.bat

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 7. Gemini API Key Setup

1. Get a free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).
2. Create a `.env` file in the root project folder by copying `.env.example`:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` in a text editor and add your API key:
   ```env
   GEMINI_API_KEY=AIzaSy...your_actual_api_key_here
   ```

> ⚠️ **Security Note:** Never commit your `.env` file to version control. It is automatically ignored in `.gitignore`.

---

## 🚀 8. Running the Application

Launch the Streamlit web application by executing:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🧠 9. How RAG Works in This Project

1. **Document Processing:** When PDFs are uploaded, `pdf_loader.py` iterates over every page and extracts text along with metadata (filename and page number).
2. **Text Chunking:** `chunker.py` uses `RecursiveCharacterTextSplitter` to divide long pages into smaller chunks (e.g. 1000 characters) with overlap (e.g. 200 characters) so that sentence context is preserved across chunk boundaries.
3. **Embeddings:** `embeddings.py` converts textual chunks into 384-dimensional dense numerical vectors representing semantic meaning.
4. **Indexing:** `vectorstore.py` stores these vectors in a FAISS index.
5. **Retrieval:** When a user enters a query, `retriever.py` embeds the query and finds the Top-K closest document vectors using cosine/Euclidean similarity search.
6. **Augmented Prompt & Generation:** `llm.py` injects the retrieved text chunks into a strict system prompt and sends it to Gemini. Gemini synthesizes a direct answer using *only* the retrieved context and cites the exact document and page.

---

## 🧪 10. Testing the Application with a Sample PDF

1. Launch the app (`streamlit run app.py`).
2. In the sidebar under **Document Management**, click **Browse files** and upload any text-based PDF (e.g., a syllabus, textbook chapter, or research paper).
3. Click **⚡ Process & Index Documents**. Wait for the success message.
4. In the chat input at the bottom of the main page, type a question whose answer is contained in the PDF.
5. Review the generated answer and expand **📚 View Source Citations** to verify the page number and retrieved text snippet.
6. Ask an out-of-context question (e.g., "What is the capital of Mars?") to test **Hallucination Prevention**. The system will respond: `"I could not find this information in the uploaded documents."`

---

## ❓ 11. Troubleshooting

* **Missing API Key Error:**  
  Make sure you created a `.env` file in the project root containing `GEMINI_API_KEY=your_key`.
* **No text extracted from PDF:**  
  The PDF may contain scanned image pages. PyPDF extracts selectable text. Scanned images require OCR tools (e.g., Tesseract).
* **FAISS Deserialization Warning / Error:**  
  The loader uses `allow_dangerous_deserialization=True` which is required for loading trusted local pickle indexes in modern LangChain versions.
* **PyTorch / SentenceTransformers Download Delay:**  
  On the first run, Hugging Face automatically downloads the lightweight `all-MiniLM-L6-v2` model (~90MB). Subsequent launches will load instantly from local cache.

---

## 🔮 12. Future Improvements & Extensions

The modular architecture of `rag/` makes it easy to add advanced features in the future:
* 📄 **Multi-Format Ingestion:** Add support for `.docx`, `.txt`, and web URLs in `pdf_loader.py`.
* 🎙️ **Voice Q&A:** Integrate Speech-to-Text and Text-to-Speech widgets in Streamlit.
* 📝 **Automatic Document Summarization:** Add a sidebar button to generate a 1-page summary of any uploaded PDF.
* 🧠 **Conversation Memory:** Maintain full chat history across retrieval turns for follow-up questions.
* 📊 **RAG Evaluation:** Integrate Ragas / TruLens framework to benchmark retrieval accuracy.
* ☁️ **Cloud Deployment:** Deploy to Streamlit Community Cloud or Docker.
