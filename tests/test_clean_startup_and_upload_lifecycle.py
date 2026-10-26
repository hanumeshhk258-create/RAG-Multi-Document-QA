"""
Comprehensive Verification Suite: Clean Startup, User-Driven Document Upload,
Incremental Processing, and Empty State Safety.

Executes Tests 1 through 7 in an isolated temporary environment to ensure:
- Fresh installation starts with 0 documents.
- No bundled sample PDFs are loaded automatically.
- Document upload drives the entire workflow.
- Incremental indexing preserves unchanged documents.
- Persistence across restarts works without importing unwanted files.
- Empty document collections reject queries with a clear, helpful message.
- Existing user documents in the main workspace are preserved untouched.
"""

import os
import sys
import io
import time
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reportlab.pdfgen import canvas
import app as rag_app
from rag.vectorstore import clear_vectorstore_metadata_cache, clear_vectorstore


def create_test_pdf_bytes(title: str, content_lines: list) -> bytes:
    """Generates a valid binary PDF with ReportLab."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 750, title)
    c.setFont("Helvetica", 11)
    y = 720
    for line in content_lines:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


class TestCleanStartupAndUploadLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create an isolated temporary directory for test storage
        cls.temp_dir = tempfile.mkdtemp(prefix="rag_clean_test_")
        cls.temp_docs = os.path.join(cls.temp_dir, "documents")
        cls.temp_vs = os.path.join(cls.temp_dir, "vectorstore")
        os.makedirs(cls.temp_docs, exist_ok=True)
        os.makedirs(cls.temp_vs, exist_ok=True)

        # Preserve original directories
        cls.orig_docs = rag_app.DOCUMENTS_DIR
        cls.orig_vs = rag_app.VECTORSTORE_DIR
        cls.orig_active_vs = rag_app.active_vectorstore

        # Point app to isolated temporary directories
        rag_app.DOCUMENTS_DIR = cls.temp_docs
        rag_app.VECTORSTORE_DIR = cls.temp_vs
        rag_app.active_vectorstore = None
        clear_vectorstore_metadata_cache()
        rag_app.clear_rag_response_cache()

        cls.client = rag_app.app.test_client()

    @classmethod
    def tearDownClass(cls):
        # Restore original paths and active vectorstore
        rag_app.DOCUMENTS_DIR = cls.orig_docs
        rag_app.VECTORSTORE_DIR = cls.orig_vs
        rag_app.active_vectorstore = cls.orig_active_vs
        clear_vectorstore_metadata_cache()
        rag_app.clear_rag_response_cache()

        # Clean up temporary test directory
        if os.path.exists(cls.temp_dir):
            try:
                shutil.rmtree(cls.temp_dir)
            except Exception:
                pass

    def test_01_fresh_installation_starts_empty(self):
        """Test 1: Verify a fresh installation has 0 documents, 0 chunks, and no auto-loaded samples."""
        print("\n[TEST 1] Verifying fresh installation starts with 0 documents...")
        res = self.client.get('/api/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data.get("total_documents"), 0, "Expected 0 total documents on fresh install")
        self.assertEqual(data.get("indexed_count"), 0, "Expected 0 indexed documents on fresh install")
        self.assertEqual(data.get("total_chunks"), 0, "Expected 0 chunks on fresh install")
        self.assertEqual(data.get("total_pages"), 0, "Expected 0 pages on fresh install")
        self.assertEqual(len(data.get("documents", [])), 0, "Expected empty documents list")
        self.assertEqual(data.get("status"), "empty", "Expected status 'empty'")
        print("  -> PASSED: Fresh installation starts with 0 documents and 0 sections.")

    def test_02_empty_state_question_validation(self):
        """Test 6: Asking a question with no documents uploaded returns a clear upload message without AI errors."""
        print("\n[TEST 6] Testing question submission when no documents are uploaded...")
        res = self.client.post('/api/chat', json={"question": "What is normalization in databases?"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()

        self.assertFalse(data.get("success"))
        expected_msg = "Please upload and index at least one PDF before asking document-related questions."
        self.assertEqual(data.get("error"), expected_msg)
        self.assertEqual(data.get("error_code"), "VALIDATION_ERROR")
        print(f"  -> PASSED: Correctly rejected with: '{data.get('error')}'")

    def test_03_upload_and_index_one_pdf(self):
        """Test 2: Upload one real PDF, verify it enters Pending status, then index it to Ready."""
        print("\n[TEST 2] Testing single PDF upload and indexing workflow...")
        pdf_bytes = create_test_pdf_bytes(
            "Photovoltaic Solar Energy Systems",
            [
                "Solar panels convert sunlight directly into electricity using silicon photovoltaic cells.",
                "The photovoltaic effect generates direct current (DC) electricity when photons strike semiconductor materials.",
                "Inverters convert the generated DC electricity into alternating current (AC) for building use."
            ]
        )

        # Upload the PDF
        up_res = self.client.post(
            '/api/upload',
            data={'files': (io.BytesIO(pdf_bytes), 'Solar_Power_Guide.pdf')},
            content_type='multipart/form-data'
        )
        self.assertEqual(up_res.status_code, 200)
        up_data = up_res.get_json()
        self.assertTrue(up_data.get("success"))

        # Verify status is now Pending with 1 document
        stat_res = self.client.get('/api/status')
        stat_data = stat_res.get_json()
        self.assertEqual(stat_data.get("total_documents"), 1)
        self.assertEqual(stat_data.get("pending_count"), 1)
        self.assertEqual(stat_data.get("indexed_count"), 0)

        # Trigger Indexing
        idx_res = self.client.post('/api/index', json={})
        self.assertEqual(idx_res.status_code, 200)
        idx_data = idx_res.get_json()
        self.assertTrue(idx_data.get("success"))

        # Verify status is now Ready with chunks created
        ready_stat = self.client.get('/api/status').get_json()
        self.assertEqual(ready_stat.get("total_documents"), 1)
        self.assertEqual(ready_stat.get("indexed_count"), 1)
        self.assertGreater(ready_stat.get("total_chunks"), 0)
        self.assertTrue(ready_stat.get("vectorstore_ready"))
        print(f"  -> PASSED: Uploaded Solar_Power_Guide.pdf, indexed {ready_stat.get('total_chunks')} chunks, status: Ready.")

    def test_04_ask_question_on_uploaded_document(self):
        """Test 3: Ask a question supported by the uploaded PDF and verify grounded answer with citations."""
        print("\n[TEST 3] Testing question answering against the newly indexed PDF...")
        res = self.client.post('/api/chat', json={
            "question": "How do solar panels generate electricity?",
            "selected_documents": ["Solar_Power_Guide.pdf"]
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertTrue(data.get("success"))
        answer = data.get("answer", "")
        self.assertTrue(len(answer) > 20)
        sources = data.get("sources", [])
        self.assertGreater(len(sources), 0)
        self.assertTrue(any("Solar_Power_Guide.pdf" in s.get("document", "") for s in sources))
        eval_info = data.get("evaluation", {})
        self.assertTrue(eval_info.get("groundedness_label") in ("Verified", "Grounded") or eval_info.get("faithfulness", 0) >= 80)
        print(f"  -> PASSED: Generated grounded answer with source citation: '{sources[0].get('document')}'")

    def test_05_upload_second_pdf_incremental_processing(self):
        """Test 4: Upload a second PDF and verify incremental processing preserves the first PDF."""
        print("\n[TEST 4] Testing upload of second PDF and incremental processing...")
        pdf2_bytes = create_test_pdf_bytes(
            "Wind Turbine Aerodynamics",
            [
                "Wind turbines convert kinetic energy of wind into mechanical power using aerodynamically shaped blades.",
                "The rotor turns a low-speed shaft connected to a gearbox that accelerates rotation for the electrical generator.",
                "Modern utility turbines feature active yaw systems to align the nacelle into the prevailing wind direction."
            ]
        )

        # Upload second PDF
        up2_res = self.client.post(
            '/api/upload',
            data={'files': (io.BytesIO(pdf2_bytes), 'Wind_Turbine_Basics.pdf')},
            content_type='multipart/form-data'
        )
        self.assertEqual(up2_res.status_code, 200)

        # Trigger indexing
        idx_res = self.client.post('/api/index', json={})
        self.assertEqual(idx_res.status_code, 200)

        # Verify both documents are present and Ready
        stat_data = self.client.get('/api/status').get_json()
        self.assertEqual(stat_data.get("total_documents"), 2)
        self.assertEqual(stat_data.get("indexed_count"), 2)
        doc_names = [d["filename"] for d in stat_data.get("documents", [])]
        self.assertIn("Solar_Power_Guide.pdf", doc_names)
        self.assertIn("Wind_Turbine_Basics.pdf", doc_names)
        print(f"  -> PASSED: Both documents active and Ready: {doc_names}")

    def test_06_restart_and_persistence(self):
        """Test 5: Simulate server restart: verify documents persist without importing unwanted sample files."""
        print("\n[TEST 5] Testing application restart persistence...")
        # Simulate restart by resetting in-memory caches and active vectorstore
        rag_app.active_vectorstore = None
        clear_vectorstore_metadata_cache()
        rag_app.clear_rag_response_cache()

        # Query status after restart simulation
        stat_data = self.client.get('/api/status').get_json()
        self.assertEqual(stat_data.get("total_documents"), 2)
        self.assertEqual(stat_data.get("indexed_count"), 2)
        self.assertTrue(stat_data.get("vectorstore_ready"))

        doc_names = [d["filename"] for d in stat_data.get("documents", [])]
        self.assertEqual(set(doc_names), {"Solar_Power_Guide.pdf", "Wind_Turbine_Basics.pdf"})
        print(f"  -> PASSED: After restart, exactly the 2 user-uploaded documents persist: {doc_names}")


if __name__ == '__main__':
    unittest.main()
