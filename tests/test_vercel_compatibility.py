import os
import io
import time
import json
import unittest

# Ensure root path
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.index import app as vercel_app, handler
from rag.storage_adapter import get_storage_adapter, LocalStorageAdapter, SupabaseStorageAdapter
from rag.embeddings import get_embedding_model, FastOnnxEmbeddings, GeminiEmbeddings
from create_sample_pdfs import create_simple_pdf


class TestVercelCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vercel_app.config["TESTING"] = True
        cls.client = vercel_app.test_client()

    def test_01_vercel_entrypoint_and_handler(self):
        """Verifies api.index entrypoint exports WSGI app and handler alias for Vercel."""
        self.assertIsNotNone(vercel_app)
        self.assertIsNotNone(handler)
        self.assertEqual(vercel_app.name, "app")

    def test_02_health_endpoint_response(self):
        """Verifies /healthz responds with HTTP 200 and required telemetry fields."""
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("storage", data)
        self.assertIn("gemini_configured", data)
        self.assertIn("version", data)

    def test_03_homepage_and_static_asset_resolution(self):
        """Verifies the UI homepage and static CSS/JS load with HTTP 200."""
        res_home = self.client.get("/")
        self.assertEqual(res_home.status_code, 200)
        self.assertIn(b"RAG Multi-Document", res_home.data)

        res_css = self.client.get("/static/css/style.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertGreater(len(res_css.data), 10000)

        res_js = self.client.get("/static/js/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertGreater(len(res_js.data), 10000)

    def test_04_embedding_dimensions_match_384(self):
        """Verifies embedding models strictly preserve the 384-dimensional contract."""
        emb = get_embedding_model()
        vec = emb.embed_query("Testing vector dimension")
        self.assertEqual(len(vec), 384, f"Expected 384 dimensions, got {len(vec)}")

    def test_05_storage_adapter_interface(self):
        """Verifies the unified storage adapter contract."""
        adapter = get_storage_adapter()
        self.assertIn(adapter.get_storage_type(), ("local", "supabase"))
        meta = adapter.load_metadata()
        self.assertIsInstance(meta, dict)
        self.assertIn("documents_info", meta)
        self.assertIn("total_documents", meta)

    def test_06_serverless_sync_indexing_lifecycle(self):
        """Simulates Vercel serverless environment with synchronous indexing and grounded QA."""
        # Enable serverless simulation flag
        os.environ["SERVERLESS"] = "1"
        test_filename = "vercel_lifecycle_test.pdf"

        try:
            # 1. Create text PDF
            create_simple_pdf(test_filename, [
                "The James Webb Space Telescope observes the universe in deep infrared wavelengths. Its primary mirror consists of eighteen hexagonal gold-coated beryllium segments.",
                "The James Webb Space Telescope is located near the second Sun-Earth Lagrange point (L2)."
            ])

            with open(test_filename, "rb") as f:
                pdf_bytes = f.read()

            # 2. Upload
            up_res = self.client.post(
                "/api/upload",
                data={"files": (io.BytesIO(pdf_bytes), test_filename)},
                content_type="multipart/form-data"
            )
            self.assertIn(up_res.status_code, (200, 201))
            up_data = up_res.get_json()
            self.assertTrue(up_data.get("success"))

            # 3. Synchronous Indexing on Serverless
            idx_res = self.client.post(
                "/api/index?async=true",
                json={"selected_documents": [test_filename]}
            )
            self.assertEqual(idx_res.status_code, 200)
            idx_data = idx_res.get_json()
            self.assertTrue(idx_data.get("success"))
            self.assertEqual(idx_data.get("status"), "ready")

            # 4. Status Verification
            st_res = self.client.get("/api/status")
            self.assertEqual(st_res.status_code, 200)
            st_data = st_res.get_json()
            doc_entry = next((d for d in st_data.get("documents", []) if d.get("filename") == test_filename), None)
            self.assertIsNotNone(doc_entry)
            self.assertEqual(doc_entry.get("status"), "Ready")
            self.assertGreater(doc_entry.get("chunks", 0), 0)

            # 5. Grounded QA Query
            chat_res = self.client.post(
                "/api/chat",
                json={
                    "question": "What is the primary mirror of the James Webb Space Telescope made of?",
                    "selected_documents": [test_filename]
                }
            )
            self.assertEqual(chat_res.status_code, 200)
            chat_data = chat_res.get_json()
            self.assertTrue(chat_data.get("success"))
            self.assertGreater(len(chat_data.get("sources", [])), 0)

            # 6. Document Deletion Verification
            del_res = self.client.post("/api/remove-document", json={"filename": test_filename})
            self.assertEqual(del_res.status_code, 200)
            del_data = del_res.get_json()
            self.assertTrue(del_data.get("success"))

            st_after = self.client.get("/api/status")
            docs_after = [d.get("filename") for d in st_after.get_json().get("documents", [])]
            self.assertNotIn(test_filename, docs_after)

        finally:
            # Cleanup test document safely
            if os.path.exists(test_filename):
                try:
                    os.remove(test_filename)
                except Exception:
                    pass
            os.environ.pop("SERVERLESS", None)

    def test_07_unconfigured_production_guard(self):
        """Verifies that in Vercel production without Supabase, the app blocks uploads and alerts rather than silently falling back."""
        os.environ["VERCEL"] = "1"
        os.environ.pop("ALLOW_LOCAL_FALLBACK", None)

        try:
            # 1. Check health endpoint reports unconfigured error safely
            res = self.client.get("/healthz")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data.get("storage"), "unconfigured_error")

            # 2. Check upload is cleanly blocked with HTTP 503 and clear error
            up_res = self.client.post(
                "/api/upload",
                data={"files": (io.BytesIO(b"%PDF-1.4 mock"), "sample.pdf")},
                content_type="multipart/form-data"
            )
            self.assertEqual(up_res.status_code, 503)
            up_data = up_res.get_json()
            self.assertFalse(up_data.get("success"))
            self.assertIn("Persistent cloud storage is not configured", up_data.get("error"))

        finally:
            os.environ.pop("VERCEL", None)


if __name__ == "__main__":
    unittest.main()

