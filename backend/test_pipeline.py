"""Load the saved model directly (no Flask) and check it reproduces the known answer for the sample applicant.

Run from the project root:  python backend/test_pipeline.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # the saved pipeline uses preprocessing code from the credit_risk package

bundle = joblib.load(ROOT / "backend/models/model_bundle.joblib")
with (ROOT / "backend/test_payload.json").open(encoding="utf-8") as f:
    payload = json.load(f)
features = bundle["features"]
assert set(features) == set(payload), "Payload keys must match features"
frame = pd.DataFrame([[payload[f] for f in features]], columns=features)
probability = bundle["pipeline"].predict_proba(frame)[0, 1]
print(f"Probability of default: {probability:.4f} (threshold {bundle['threshold']:.4f})")
print("Prediction:", int(probability >= bundle["threshold"]))

expected = bundle["reference"]["probability"]
assert abs(probability - expected) < 1e-9, f"expected {expected:.6f} from training, got {probability:.6f}"
print("PASS: local inference matches the training-time result")
