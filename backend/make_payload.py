"""Turn the first row of data/sample_inputs.csv into backend/test_payload.json (the JSON body for /predict).

Run from the project root:  python backend/make_payload.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # the saved pipeline uses preprocessing code from the credit_risk package
bundle = joblib.load(ROOT / "backend/models/model_bundle.joblib")
features = bundle["features"]
df = pd.read_csv(ROOT / "data/sample_inputs.csv", nrows=1)
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Sample CSV missing columns: {missing}")
row = df.loc[0, features]
payload = {}
for key, value in row.items():
    payload[key] = None if pd.isna(value) else value.item() if hasattr(value, "item") else value
out = ROOT / "backend/test_payload.json"
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Created {out} with {len(payload)} features")
