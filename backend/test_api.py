"""Send test_payload.json to the running Flask server (start it first with: python backend/app.py).

Run from the project root:  python backend/test_api.py
"""
import json
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
payload = json.loads((BASE / "test_payload.json").read_text(encoding="utf-8"))
with requests.Session() as session:
    session.trust_env = False  # avoids an inherited proxy for localhost
    response = session.post("http://127.0.0.1:5000/predict", json=payload, timeout=60)
    print("HTTP Status:", response.status_code)
    print("Response:", response.text)
    response.raise_for_status()
print("PASS: API request")
