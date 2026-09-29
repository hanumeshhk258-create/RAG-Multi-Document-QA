import requests
import json
import time

t0 = time.perf_counter()
resp = requests.post(
    "http://127.0.0.1:5000/api/chat",
    json={"question": "What is the conclusion?", "selected_documents": None},
    timeout=45
)
elapsed = round(time.perf_counter() - t0, 2)
print("STATUS:", resp.status_code, f"({elapsed}s)")
data = resp.json()
print("ANSWER:")
print(data.get("answer"))
print("\nEVALUATION:")
print(json.dumps(data.get("evaluation", {}), indent=2))
print("\nSOURCES:")
print(data.get("sources", []))
