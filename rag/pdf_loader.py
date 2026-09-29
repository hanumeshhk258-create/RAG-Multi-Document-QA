import os
import io
import re
import logging
from typing import List, Tuple
import pypdf
from langchain_core.documents import Document

# Set up logging for PDF processing
logger = logging.getLogger(__name__)


def clean_pdf_text(raw_text: str) -> str:
    """
    Cleans and normalizes extracted PDF text:
    - Removes private use area Unicode glyphs and font artifacts.
    - Normalizes line breaks and whitespace while preserving paragraph and code structure.
    - Fixes hyphenated line breaks.
    - Preserves headings, code examples, SQL commands, and lists.
    
    Args:
        raw_text: Raw text extracted from PDF page.
        
    Returns:
        Cleaned, normalized text string.
    """
    if not raw_text:
        return ""

    # 1. Remove Private Use Area unicode characters (e.g. \ue000-\uf8ff)
    text = re.sub(r'[\ue000-\uf8ff]', ' ', raw_text)

    # 2. Standardize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 3. Clean email footers / common watermark lines if present
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    text = re.sub(r'Keep Learning & Keep Exploring!?', '', text, flags=re.IGNORECASE)

    # 4. Normalize multiple horizontal spaces and tabs into a single space
    text = re.sub(r'[ \t]+', ' ', text)

    # 5. Fix hyphenated word breaks at end of line (e.g., "defini-\ntion" -> "definition")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # 6. Normalize excessive blank lines (limit to max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def get_pdf_info(file_path: str, filename: str = None) -> dict:
    """
    Inspects a PDF file and returns summary metadata (page count, size in KB, extractable).
    """
    fname = filename or (os.path.basename(file_path) if isinstance(file_path, str) else "document.pdf")
    info = {
        "filename": fname,
        "page_count": 0,
        "size_kb": 0,
        "has_text": False
    }

    try:
        if isinstance(file_path, str) and os.path.exists(file_path):
            info["size_kb"] = round(os.path.getsize(file_path) / 1024, 1)
            reader = pypdf.PdfReader(file_path)
        elif hasattr(file_path, "read"):
            reader = pypdf.PdfReader(file_path)
        else:
            reader = pypdf.PdfReader(file_path)

        info["page_count"] = len(reader.pages)
        # Check first couple of pages for text
        for p in reader.pages[:3]:
            txt = p.extract_text() or ""
            if len(txt.strip()) > 20:
                info["has_text"] = True
                break
    except Exception as e:
        logger.warning(f"Could not read PDF info for {fname}: {e}")

    return info


def load_pdf_file(file_obj, filename: str) -> Tuple[List[Document], List[str]]:
    """
    Extracts text page-by-page from a single PDF file stream or path with cleaning and metadata preservation.
    
    Args:
        file_obj: File-like object, file path, or bytes.
        filename: Name of the PDF file for metadata attribution.
        
    Returns:
        Tuple containing:
        - List of LangChain Document objects with cleaned page text and metadata.
        - List of warning/error messages encountered during extraction.
    """
    documents = []
    warnings = []

    try:
        # Read PDF using PyPDF
        if isinstance(file_obj, str):
            reader = pypdf.PdfReader(file_obj)
        elif isinstance(file_obj, bytes):
            reader = pypdf.PdfReader(io.BytesIO(file_obj))
        elif hasattr(file_obj, "getvalue"):
            reader = pypdf.PdfReader(io.BytesIO(file_obj.getvalue()))
        elif hasattr(file_obj, "read"):
            reader = pypdf.PdfReader(file_obj)
        else:
            reader = pypdf.PdfReader(file_obj)

        if hasattr(reader, "is_encrypted") and reader.is_encrypted:
            warnings.append(f"'{filename}' is password protected or encrypted. Please provide an unencrypted PDF.")
            return documents, warnings

        num_pages = len(reader.pages)
        if num_pages == 0:
            warnings.append(f"'{filename}' appears to be empty (0 pages).")
            return documents, warnings

        extracted_text_count = 0

        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1  # 1-based page indexing for user clarity
            try:
                raw_page_text = page.extract_text() or ""
                cleaned_text = clean_pdf_text(raw_page_text)

                if cleaned_text:
                    extracted_text_count += len(cleaned_text)
                    # Create LangChain Document with rich metadata
                    doc = Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": filename,
                            "document": filename,
                            "page": page_number,
                            "total_pages": num_pages
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Extracted page {page_number}/{num_pages} from '{filename}' ({len(cleaned_text)} chars).")
                else:
                    logger.warning(f"No extractable text found on page {page_number} of '{filename}'.")
            except Exception as page_err:
                warnings.append(f"Failed to extract page {page_number} from '{filename}': {str(page_err)}")

        if extracted_text_count == 0:
            warnings.append(
                f"No extractable text found in '{filename}'. "
                "It may contain scanned images or require OCR."
            )

    except pypdf.errors.PdfReadError as pdf_err:
        warnings.append(f"Invalid or corrupted PDF file '{filename}': {str(pdf_err)}")
    except Exception as err:
        warnings.append(f"Unexpected error processing '{filename}': {str(err)}")

    # Step 2: Extract structured tables from the PDF
    try:
        from rag.table_extractor import extract_tables_from_pdf
        table_docs, table_metas = extract_tables_from_pdf(file_obj, filename)
        if table_docs:
            documents.extend(table_docs)
            logger.info(f"Appended {len(table_docs)} structured table chunk(s) from '{filename}' into document stream.")
    except Exception as table_err:
        logger.warning(f"Table extraction notice for '{filename}': {table_err}")

    return documents, warnings


def filter_new_documents(uploaded_files, already_indexed_names: List[str]) -> Tuple[List, List[str]]:
    """
    Separates newly uploaded files from files that are already indexed to prevent duplicate processing.
    """
    indexed_set = set(already_indexed_names or [])
    new_files = []
    duplicate_names = []

    for file in uploaded_files:
        filename = getattr(file, "name", getattr(file, "filename", "uploaded_document.pdf"))
        if filename in indexed_set:
            duplicate_names.append(filename)
        else:
            new_files.append(file)

    return new_files, duplicate_names


def process_multiple_pdfs(uploaded_files) -> Tuple[List[Document], List[str]]:
    """
    Processes multiple PDF files and aggregates their documents and warning messages.
    """
    all_documents = []
    all_warnings = []

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", getattr(uploaded_file, "filename", "uploaded_document.pdf"))
        docs, warnings = load_pdf_file(uploaded_file, filename)
        all_documents.extend(docs)
        all_warnings.extend(warnings)

    return all_documents, all_warnings
