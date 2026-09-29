"""
Helper to create valid multi-page PDF documents for testing RAG pipeline without external PDF compiler packages.
"""
import os

def create_simple_pdf(filename: str, pages_content: list):
    """
    Creates a standard, valid text PDF file with multiple pages.
    """
    objects = []
    
    # 1: Catalog
    # 2: Pages
    # 3..: Page and Content objects
    
    num_pages = len(pages_content)
    page_obj_ids = [3 + i * 2 for i in range(num_pages)]
    
    # Object 1: Catalog
    catalog = f"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    
    # Object 2: Pages
    kids_str = " ".join([f"{pid} 0 R" for pid in page_obj_ids])
    pages_obj = f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>\nendobj\n"
    
    objects.append((1, catalog))
    objects.append((2, pages_obj))
    
    for idx, page_text in enumerate(pages_content):
        page_id = 3 + idx * 2
        content_id = page_id + 1
        
        # Build text stream
        lines = page_text.strip().split("\n")
        stream_cmds = ["BT", "/F1 12 Tf", "50 750 Td", "16 TL"]
        
        for i, line in enumerate(lines):
            # Escape parenthesis
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if i == 0:
                stream_cmds.append(f"({escaped}) Tj")
            else:
                stream_cmds.append(f"T* ({escaped}) Tj")
                
        stream_cmds.append("ET")
        stream_data = "\n".join(stream_cmds)
        
        page_dict = (
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\n"
            f"endobj\n"
        )
        
        content_dict = (
            f"{content_id} 0 obj\n"
            f"<< /Length {len(stream_data.encode('utf-8'))} >>\n"
            f"stream\n"
            f"{stream_data}\n"
            f"endstream\n"
            f"endobj\n"
        )
        
        objects.append((page_id, page_dict))
        objects.append((content_id, content_dict))
        
    # Write PDF file
    objects.sort(key=lambda x: x[0])
    
    with open(filename, "wb") as f:
        f.write(b"%PDF-1.4\n")
        offsets = {}
        for obj_id, obj_content in objects:
            offsets[obj_id] = f.tell()
            f.write(obj_content.encode('latin-1'))
            
        xref_pos = f.tell()
        f.write(b"xref\n")
        f.write(f"0 {len(objects) + 1}\n".encode('ascii'))
        f.write(b"0000000000 65535 f \n")
        for obj_id in range(1, len(objects) + 1):
            offset = offsets[obj_id]
            f.write(f"{offset:010d} 00000 n \n".encode('ascii'))
            
        f.write(b"trailer\n")
        f.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode('ascii'))
        f.write(b"startxref\n")
        f.write(f"{xref_pos}\n".encode('ascii'))
        f.write(b"%%EOF\n")

    print(f"Created {filename} with {num_pages} pages.")


if __name__ == "__main__":
    os.makedirs("documents", exist_ok=True)
    
    # 1. DBMS Notes
    dbms_p1 = (
        "DBMS Fundamentals and Architecture\n"
        "A Database Management System (DBMS) is software that manages database creation, storage, and querying.\n"
        "Key Characteristics of DBMS:\n"
        "- Reduces data redundancy and inconsistency.\n"
        "- Supports concurrent access and transactions.\n"
        "- Enforces security and data integrity constraints.\n\n"
        "ACID Properties:\n"
        "1. Atomicity: All operations in a transaction succeed, or none are applied (all-or-nothing).\n"
        "2. Consistency: Database remains in a valid state before and after transaction execution.\n"
        "3. Isolation: Concurrent transactions execute independently without interference.\n"
        "4. Durability: Once committed, transaction results are permanently saved in persistent storage."
    )
    
    dbms_p2 = (
        "Relational Database Keys and Normalization\n"
        "Keys in Relational Database Management Systems:\n"
        "- Primary Key: A primary key is a column or set of columns that uniquely identifies each row in a database table. It cannot contain NULL values and must contain unique values for every record.\n"
        "- Candidate Key: A minimal superkey that can uniquely identify records in a relation.\n"
        "- Foreign Key: A column that establishes a link between data in two tables by referencing the primary key of another table.\n\n"
        "Normal Forms:\n"
        "- 1NF (First Normal Form): Eliminates duplicate columns and requires atomic (indivisible) values.\n"
        "- 2NF (Second Normal Form): Must be in 1NF and eliminate partial dependencies (all non-key attributes fully dependent on primary key).\n"
        "- 3NF (Third Normal Form): Must be in 2NF and eliminate transitive dependencies.\n"
        "- BCNF (Boyce-Codd Normal Form): A stricter version of 3NF where every determinant must be a candidate key."
    )
    
    create_simple_pdf("documents/DBMS_Notes.pdf", [dbms_p1, dbms_p2])
    
    # 2. Python Notes
    py_p1 = (
        "Python Programming and Core Concepts\n"
        "Python is a high-level, interpreted, dynamically-typed programming language known for readability.\n\n"
        "Key Data Structures in Python:\n"
        "- Lists: Ordered, mutable collections of items defined with square brackets [].\n"
        "- Tuples: Ordered, immutable collections of items defined with parentheses ().\n"
        "- Dictionaries: Key-value pairs with fast O(1) average lookup, defined with curly braces {}.\n"
        "- Sets: Unordered collections of unique elements."
    )
    
    py_p2 = (
        "Python Advanced Features: Decorators and Generators\n"
        "Decorators:\n"
        "A decorator is a design pattern and language feature that extends the behavior of a function\n"
        "or class without modifying its source code. Decorators use the @ syntax.\n\n"
        "Generators:\n"
        "Generators are functions that return an iterator using the 'yield' keyword instead of 'return'.\n"
        "They compute values on demand (lazy evaluation), which saves memory when processing large sequences."
    )
    
    create_simple_pdf("documents/Python_Notes.pdf", [py_p1, py_p2])

    # 3. SQL CheatSheet
    sql_p1 = (
        "SQL Overview and Relational Database Queries\n"
        "Structured Query Language (SQL) is the standard declarative language for interacting with relational databases.\n\n"
        "SQL Sub-Languages:\n"
        "1. DDL (Data Definition Language): CREATE, ALTER, DROP, TRUNCATE.\n"
        "2. DML (Data Manipulation Language): SELECT, INSERT, UPDATE, DELETE.\n"
        "3. DCL (Data Control Language): GRANT, REVOKE permissions.\n"
        "4. TCL (Transaction Control Language): COMMIT, ROLLBACK, SAVEPOINT."
    )
    sql_p2 = (
        "SQL Joins and Query Optimization\n"
        "Types of Joins (Joining Commands):\n"
        "- INNER JOIN: Returns records that have matching values in both tables.\n"
        "- LEFT (OUTER) JOIN: Returns all records from the left table and matched records from the right table.\n"
        "- RIGHT (OUTER) JOIN: Returns all records from the right table and matched records from the left table.\n"
        "- FULL (OUTER) JOIN: Returns all records when there is a match in either the left or right table.\n\n"
        "Query Optimization Best Practices:\n"
        "- Use appropriate indexes on frequently filtered columns (WHERE clause).\n"
        "- Avoid SELECT *; explicitly name needed columns.\n"
        "- Use EXPLAIN query plan to inspect execution bottlenecks."
    )
    create_simple_pdf("documents/SQL_CheatSheet.pdf", [sql_p1, sql_p2])

    # 4. CommerceOS Project Report
    comm_p1 = (
        "COMMERCEOS | COMPLETE PROJECT REPORT\n"
        "CommerceOS - AI Growth & Agentic Commerce Platform\n\n"
        "1. Executive Summary\n"
        "CommerceOS is an enterprise autonomous agent platform designed to streamline\n"
        "multi-channel commerce operations, inventory optimization, and automated conversion pipelines."
    )
    comm_p2 = (
        "CommerceOS Architecture & Systems\n"
        "2. Microservices Architecture Overview\n"
        "The system utilizes FastAPI microservices, Redis caching layer, and PostgreSQL persistent store.\n\n"
        "| Service | Tech Stack | Role |\n"
        "| Ingestion | Python / Kafka | Event stream ingestion |\n"
        "| Reasoning | Gemini 2.0 | Multi-agent decision logic |\n"
        "| Analytics | DuckDB / SQL | Real-time reporting |"
    )
    comm_p3 = (
        "CommerceOS Data Pipelines\n"
        "3. Real-Time Data Pipeline\n"
        "Incoming customer interaction events are processed through asynchronous event buses.\n\n"
        "| Pipeline Stage | Throughput | Latency |\n"
        "| Event Ingestion | 50,000 evt/s | 12 ms |\n"
        "| Transformation | 45,000 evt/s | 24 ms |\n"
        "| Vector Embedding | 10,000 doc/s | 45 ms |"
    )
    comm_p4 = (
        "COMMERCEOS | COMPLETE PROJECT REPORT\n"
        "CommerceOS - AI Growth & Agentic Commerce\n\n"
        "7. Waterfall Model\n"
        "The project can be documented using the Waterfall model because development progressed\n"
        "through planned, sequential stages starting from requirements analysis and system design,\n"
        "moving to implementation, testing, deployment, and operational maintenance.\n\n"
        "Each stage was systematically verified and signed off before transitioning to the subsequent phase."
    )
    create_simple_pdf("documents/CommerceOS_Complete_Project_Report.pdf", [comm_p1, comm_p2, comm_p3, comm_p4])

    # 5. Doc Alpha (Security and Key Management)
    alpha_p1 = (
        "Information Security and Key Management Protocols\n"
        "Key Escrow Architecture:\n"
        "Key Escrow is an arrangement in which cryptographic keys are held in escrow by a trusted third party\n"
        "so that, under authorized circumstances, authorized parties may gain access to the decryption keys.\n\n"
        "Key Escrow Mechanism and Security:\n"
        "- Secret keys are split into multiple components using threshold cryptography.\n"
        "- Requires dual authorization to reconstruct a master decryption key.\n"
        "- Used in compliance, data recovery, and enterprise security management."
    )
    create_simple_pdf("documents/Doc_Alpha.pdf", [alpha_p1])


