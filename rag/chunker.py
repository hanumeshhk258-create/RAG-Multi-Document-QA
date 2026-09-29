import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def get_configured_chunk_settings():
    """Reads chunk_size and chunk_overlap from environment variables with safe defaults (700-1000 chars, 100-150 overlap)."""
    load_dotenv(override=True)
    try:
        size = int(os.getenv("RAG_CHUNK_SIZE", "850"))
    except ValueError:
        size = 850

    try:
        overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "130"))
    except ValueError:
        overlap = 130

    return size, overlap


def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:
    """
    Splits documents into meaningful, overlapping text chunks preserving headings, paragraphs, SQL/code blocks, and metadata.
    
    Args:
        documents: List of input Document objects extracted from PDFs.
        chunk_size: Maximum character length for each chunk (default: 850 chars).
        chunk_overlap: Number of characters to overlap between consecutive chunks (default: 130 chars).
        
    Returns:
        List of chunked Document objects with complete metadata (document, source, page, total_pages, chunk_id).
    """
    if not documents:
        return []

    default_size, default_overlap = get_configured_chunk_settings()
    actual_size = chunk_size if chunk_size is not None else default_size
    actual_overlap = chunk_overlap if chunk_overlap is not None else default_overlap

    # Configure text splitter prioritizing section breaks, headings, paragraphs, bullet points, and sentences
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=actual_size,
        chunk_overlap=actual_overlap,
        length_function=len,
        separators=["\n\n", "\n", ";\n", ". ", "; ", ", ", " ", ""]
    )

    # Separate table documents from regular text documents
    text_docs = []
    table_docs = []

    for doc in documents:
        if getattr(doc, "metadata", {}).get("is_table"):
            table_docs.append(doc)
        else:
            text_docs.append(doc)

    # Split regular text documents into chunks
    chunks = text_splitter.split_documents(text_docs) if text_docs else []

    # Append structured table documents intact
    chunks.extend(table_docs)

    # Attach unique chunk ID and standardized document metadata for tracking
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx + 1
        # Standardize document name
        if "source" in chunk.metadata and "document" not in chunk.metadata:
            chunk.metadata["document"] = chunk.metadata["source"]
        elif "document" in chunk.metadata and "source" not in chunk.metadata:
            chunk.metadata["source"] = chunk.metadata["document"]

    logger.info(f"Split {len(documents)} page/table documents into {len(chunks)} text & table chunks ({len(table_docs)} tables, size: {actual_size}, overlap: {actual_overlap}).")
    return chunks
