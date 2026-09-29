import urllib.request
import urllib.error
import json

base_url = "http://127.0.0.1:5000"

print("1. Testing GET /api/status...")
try:
    with urllib.request.urlopen(f"{base_url}/api/status") as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print(f"Status: {resp.status}, Total docs: {data.get('total_documents')}, Vectorstore ready: {data.get('vectorstore_ready')}")
except Exception as e:
    print(f"Status failed: {e}")

print("\n2. Testing GET /view_pdf/DBMS_Notes.pdf...")
try:
    req = urllib.request.Request(f"{base_url}/view_pdf/DBMS_Notes.pdf")
    with urllib.request.urlopen(req) as resp:
        content_type = resp.headers.get("Content-Type")
        content_disp = resp.headers.get("Content-Disposition")
        content_len = len(resp.read())
        print(f"Status: {resp.status}, Content-Type: {content_type}, Content-Disposition: {content_disp}, Bytes read: {content_len}")
        assert "application/pdf" in content_type, f"Expected application/pdf, got {content_type}"
        assert content_disp is None or "attachment" not in content_disp, f"Should NOT force download, got {content_disp}"
        print("[OK] View PDF returned inline PDF correctly!")
except Exception as e:
    print(f"View PDF failed: {e}")

print("\n3. Testing GET /view_pdf/nonexistent_file.pdf (expecting 404)...")
try:
    req = urllib.request.Request(f"{base_url}/view_pdf/nonexistent_file.pdf")
    with urllib.request.urlopen(req) as resp:
        print(f"Unexpected status: {resp.status}")
except urllib.error.HTTPError as e:
    body = e.read().decode('utf-8')
    print(f"Status: {e.code}, Body: {body}")
    assert e.code == 404
    assert "PDF file is no longer available." in body
    print("[OK] 404 handler verified properly!")

print("\n4. Testing POST /api/search for 'ACID'...")
try:
    req_data = json.dumps({"query": "ACID", "selected_documents": ["DBMS_Notes.pdf"]}).encode('utf-8')
    req = urllib.request.Request(f"{base_url}/api/search", data=req_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        sdata = json.loads(resp.read().decode('utf-8'))
        print(f"Status: {resp.status}, Count: {sdata.get('count')}")
        if sdata.get("results"):
            res0 = sdata["results"][0]
            print(f"First result: doc_id={res0.get('document_id')}, filename={res0.get('filename')}, page={res0.get('page')}, score={res0.get('score')}, pdf_url={res0.get('pdf_url')}")
            assert "/view_pdf/" in res0.get("pdf_url")
            assert "#page=" in res0.get("pdf_url")
            print("[OK] Search result pdf_url format verified!")
except Exception as e:
    print(f"Search test failed: {e}")

print("\nAll verification checks complete.")
