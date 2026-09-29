"""
Table & Structured Data Extraction Module for Multi-Document RAG.

Extracts, parses, and converts tabular data from PDF files into RAG-friendly structured representations.
Supports:
- pdfplumber native table extraction with boundary detection
- Text-based Markdown/Pipe/ASCII table heuristic extraction fallback
- Conversion into both Markdown table and structured key-value format
- Granular row-level sentence synthesis for dense & sparse retrieval
- Full metadata preservation (document, page, table number, column headers, rows)
- Strict grounding with zero hallucination
"""

import os
import re
import io
import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

logger = logging.getLogger(__name__)


def clean_cell_text(cell_value: Any) -> str:
    """Cleans table cell text by stripping excessive whitespace, newlines, and non-printable characters."""
    if cell_value is None:
        return ""
    text = str(cell_value)
    # Replace newlines inside cell with spaces
    text = re.sub(r'[\r\n]+', ' ', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_markdown_table(columns: List[str], rows: List[List[str]]) -> str:
    """Converts column headers and row data into standard Markdown table format."""
    if not columns and not rows:
        return ""
    
    # Ensure all rows match column length
    max_cols = max(len(columns), max((len(r) for r in rows), default=0))
    if max_cols == 0:
        return ""

    norm_cols = [c if idx < len(columns) and c else f"Column {idx+1}" for idx, c in enumerate(columns[:max_cols])]
    while len(norm_cols) < max_cols:
        norm_cols.append(f"Column {len(norm_cols)+1}")

    header_line = "| " + " | ".join(norm_cols) + " |"
    separator_line = "| " + " | ".join(["---"] * max_cols) + " |"

    row_lines = []
    for row in rows:
        norm_row = [clean_cell_text(row[idx]) if idx < len(row) else "" for idx in range(max_cols)]
        row_lines.append("| " + " | ".join(norm_row) + " |")

    return "\n".join([header_line, separator_line] + row_lines)


def format_structured_table_text(
    source: str,
    page: int,
    table_number: int,
    columns: List[str],
    rows: List[List[str]]
) -> str:
    """
    Creates an LLM & embedding-friendly representation of the table.
    Includes explicit column keys and row values for maximum query matching accuracy.
    """
    cols_str = ", ".join(columns) if columns else "N/A"
    
    lines = [
        f"[Table]",
        f"Source: {source}",
        f"Page: {page}",
        f"Table: {table_number}",
        f"Columns: {cols_str}",
    ]

    # Add row-by-row key-value mapping
    for r_idx, row in enumerate(rows, 1):
        cell_pairs = []
        for c_idx, val in enumerate(row):
            col_name = columns[c_idx] if c_idx < len(columns) and columns[c_idx] else f"Col_{c_idx+1}"
            cell_pairs.append(f"{col_name}: {val}")
        row_str = " | ".join(cell_pairs) if cell_pairs else " | ".join(row)
        lines.append(f"Row {r_idx}: {row_str}")

    # Add Markdown representation
    md_table = format_markdown_table(columns, rows)
    if md_table:
        lines.append("\nTable Content:\n" + md_table)

    return "\n".join(lines)


def generate_row_sentences(
    source: str,
    page: int,
    table_number: int,
    columns: List[str],
    rows: List[List[str]]
) -> List[str]:
    """Generates natural language statements for each row to enhance semantic retrieval matching."""
    sentences = []
    for r_idx, row in enumerate(rows, 1):
        pairs = []
        for c_idx, val in enumerate(row):
            if val and val.strip():
                col_name = columns[c_idx] if c_idx < len(columns) and columns[c_idx] else f"Column {c_idx+1}"
                pairs.append(f"{col_name} is '{val.strip()}'")
        if pairs:
            stmt = f"In {source} (Page {page}, Table {table_number}, Row {r_idx}): " + ", ".join(pairs) + "."
            sentences.append(stmt)
    return sentences


def extract_tables_from_text_heuristics(
    page_text: str,
    source: str,
    page_number: int,
    starting_table_num: int = 1
) -> List[Dict[str, Any]]:
    """
    Heuristic fallback to extract Markdown or pipe-delimited tables embedded in raw text.
    Detects patterns like:
    | Col1 | Col2 |
    | --- | --- |
    | Val1 | Val2 |
    """
    extracted_tables = []
    if not page_text or "|" not in page_text:
        return extracted_tables

    lines = page_text.split("\n")
    table_blocks: List[List[str]] = []
    current_block: List[str] = []

    for line in lines:
        line_s = line.strip()
        if line_s.startswith("|") and line_s.endswith("|") and line_s.count("|") >= 2:
            current_block.append(line_s)
        elif re.match(r'^[+\-|=]{3,}$', line_s):
            # ASCII divider
            continue
        else:
            if len(current_block) >= 2:
                table_blocks.append(list(current_block))
            current_block = []

    if len(current_block) >= 2:
        table_blocks.append(list(current_block))

    table_counter = starting_table_num
    for block in table_blocks:
        parsed_rows = []
        for bline in block:
            # Skip delimiter rows (e.g., | --- | --- |)
            if re.match(r'^\|[\s\-:|]+\|$', bline):
                continue
            cells = [clean_cell_text(c) for c in bline.strip("|").split("|")]
            if any(cells):
                parsed_rows.append(cells)

        if len(parsed_rows) >= 2:
            # First row is header
            headers = parsed_rows[0]
            data_rows = parsed_rows[1:]
            
            # Filter empty rows
            data_rows = [r for r in data_rows if any(r)]
            if not data_rows:
                continue

            md_tbl = format_markdown_table(headers, data_rows)
            struct_txt = format_structured_table_text(source, page_number, table_counter, headers, data_rows)
            row_stmts = generate_row_sentences(source, page_number, table_counter, headers, data_rows)

            extracted_tables.append({
                "document": source,
                "page": page_number,
                "table_number": table_counter,
                "columns": headers,
                "rows": data_rows,
                "num_rows": len(data_rows),
                "num_cols": len(headers),
                "markdown": md_tbl,
                "structured_text": struct_txt,
                "row_sentences": row_stmts
            })
            table_counter += 1

    return extracted_tables


def extract_tables_from_pdf(
    file_path_or_obj: Any,
    filename: str
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    Extracts all tables from a PDF file using pdfplumber with heuristic text fallback.
    
    Args:
        file_path_or_obj: PDF file path (str), file-like object, or bytes.
        filename: Name of the PDF file for metadata attribution.
        
    Returns:
        Tuple containing:
        - List of LangChain Document objects representing table chunks for vector & BM25 indexing.
        - List of structured table metadata dictionaries.
    """
    table_documents: List[Document] = []
    table_meta_list: List[Dict[str, Any]] = []

    logger.info(f"TABLE DETECTION: Starting scan for document '{filename}'...")

    total_tables_found = 0

    # Approach 1: Use pdfplumber if available
    if HAS_PDFPLUMBER:
        try:
            pdf_source = file_path_or_obj
            if isinstance(file_path_or_obj, bytes):
                pdf_source = io.BytesIO(file_path_or_obj)
            elif hasattr(file_path_or_obj, "read") and not isinstance(file_path_or_obj, str):
                if hasattr(file_path_or_obj, "seek"):
                    file_path_or_obj.seek(0)
                pdf_source = io.BytesIO(file_path_or_obj.read())

            with pdfplumber.open(pdf_source) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    try:
                        raw_tables = page.extract_tables()
                        page_table_count = 0

                        if raw_tables:
                            for t_idx, raw_table in enumerate(raw_tables, 1):
                                if not raw_table or len(raw_table) < 2:
                                    continue

                                # Clean all cell text
                                cleaned_table = []
                                for row in raw_table:
                                    if row is not None:
                                        cleaned_row = [clean_cell_text(cell) for cell in row]
                                        # Only keep row if it contains at least one non-empty cell
                                        if any(c for c in cleaned_row):
                                            cleaned_table.append(cleaned_row)

                                if len(cleaned_table) < 2:
                                    continue

                                # Header and data rows
                                header_row = cleaned_table[0]
                                # Ensure headers are not all empty
                                if not any(header_row):
                                    header_row = [f"Col {i+1}" for i in range(len(header_row))]
                                else:
                                    header_row = [h if h else f"Col {i+1}" for i, h in enumerate(header_row)]

                                data_rows = cleaned_table[1:]
                                if not data_rows:
                                    continue

                                page_table_count += 1
                                total_tables_found += 1
                                table_num = page_table_count

                                md_tbl = format_markdown_table(header_row, data_rows)
                                struct_txt = format_structured_table_text(
                                    source=filename,
                                    page=page_num,
                                    table_number=table_num,
                                    columns=header_row,
                                    rows=data_rows
                                )
                                row_stmts = generate_row_sentences(
                                    source=filename,
                                    page=page_num,
                                    table_number=table_num,
                                    columns=header_row,
                                    rows=data_rows
                                )

                                table_meta = {
                                    "document": filename,
                                    "document_id": filename,
                                    "filename": filename,
                                    "source": filename,
                                    "page": page_num,
                                    "table_number": table_num,
                                    "columns": header_row,
                                    "rows": data_rows,
                                    "num_rows": len(data_rows),
                                    "num_cols": len(header_row),
                                    "markdown": md_tbl,
                                    "structured_text": struct_txt,
                                    "row_sentences": row_stmts,
                                    "is_table": True,
                                    "table_id": f"{filename}_p{page_num}_t{table_num}"
                                }
                                table_meta_list.append(table_meta)

                                # Create Document for RAG indexing
                                doc = Document(
                                    page_content=struct_txt,
                                    metadata={
                                        "source": filename,
                                        "document": filename,
                                        "filename": filename,
                                        "page": page_num,
                                        "total_pages": len(pdf.pages),
                                        "is_table": True,
                                        "table_number": table_num,
                                        "columns": header_row,
                                        "num_rows": len(data_rows),
                                        "num_cols": len(header_row),
                                        "table_markdown": md_tbl,
                                        "table_rows": data_rows,
                                        "table_id": f"{filename}_p{page_num}_t{table_num}"
                                    }
                                )
                                table_documents.append(doc)

                        if page_table_count > 0:
                            logger.info(
                                f"TABLE DETECTION: Found {page_table_count} table(s) on Page {page_num} of '{filename}'"
                            )

                    except Exception as page_table_err:
                        logger.warning(f"Error extracting tables on Page {page_num} of '{filename}': {page_table_err}")

        except Exception as pdf_open_err:
            logger.warning(f"pdfplumber table extraction failed for '{filename}': {pdf_open_err}")

    logger.info(f"TABLES FOUND: Total {total_tables_found} table(s) extracted from '{filename}'.")
    return table_documents, table_meta_list


def filter_table_rows_for_query(
    query: str,
    table_meta: Dict[str, Any],
    max_rows: int = 10
) -> List[List[str]]:
    """
    Selects the most relevant rows of a table for a specific user query.
    Used for table-aware contextual compression and UI structured data preview.
    """
    rows = table_meta.get("rows", [])
    columns = table_meta.get("columns", [])
    if not rows:
        return []

    q_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', query.lower())) - {
        "what", "is", "the", "a", "an", "and", "or", "in", "on", "of", "to", "for", "with", "show", "table"
    }
    if not q_tokens:
        return rows[:max_rows]

    scored_rows = []
    for idx, row in enumerate(rows):
        row_text = " ".join(str(c).lower() for c in row)
        row_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', row_text))
        matched = q_tokens.intersection(row_tokens)
        score = len(matched)
        scored_rows.append((score, idx, row))

    # If some rows matched, return matching rows first
    matched_rows = [r for score, idx, r in scored_rows if score > 0]
    if matched_rows:
        return matched_rows[:max_rows]

    return rows[:max_rows]
