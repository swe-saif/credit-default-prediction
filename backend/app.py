"""Flask REST API that serves the credit default model.

Run from the project root:  python backend/app.py   ->  http://127.0.0.1:5000
Endpoints: GET /health, POST /predict (JSON object with one value per model input).
"""
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))  # the saved pipeline uses preprocessing code from the credit_risk package

BUNDLE = joblib.load(BASE / "models" / "model_bundle.joblib")  # trusted file produced by credit_risk.export_model
MODEL = BUNDLE["pipeline"]
FEATURES = BUNDLE["features"]
SCHEMA = {field["name"]: field for field in BUNDLE["schema"]}
THRESHOLD = BUNDLE["threshold"]
TASK = BUNDLE.get("task", "classification")
NUMERIC = [f for f in FEATURES if SCHEMA[f]["kind"] == "number"]
app = Flask(__name__)


def validate(data):
    """List what is wrong with one applicant's inputs; an empty list means the inputs are usable."""
    problems = []
    for name, field in SCHEMA.items():
        value = data[name]
        if value is None or value == "":
            if field["required"]:
                problems.append(f"{name} is required")
        elif field["kind"] == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"{name} must be a number")
            elif not field["min"] <= value <= field["max"]:
                problems.append(f"{name} must be between {field['min']:g} and {field['max']:g}")
        elif value not in field["options"]:
            problems.append(f"{name} must be one of: {', '.join(field['options'])}")
    return problems


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES), task=TASK, model_version=BUNDLE["model_version"])


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400
    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return jsonify(error="Input schema mismatch", missing=missing, extra=extra), 400
    problems = validate(data)
    if problems:
        return jsonify(error="Invalid input", detail=problems), 400

    # Blank optional inputs become missing values, which the saved pipeline fills exactly as in training.
    frame = pd.DataFrame([{f: None if data[f] == "" else data[f] for f in FEATURES}], columns=FEATURES)
    frame[NUMERIC] = frame[NUMERIC].astype(float)
    try:
        probability = float(MODEL.predict_proba(frame)[0, 1])
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500

    prediction = int(probability >= THRESHOLD)  # threshold chosen on the validation set, not a fixed 0.5
    return jsonify(
        prediction=prediction,
        label="Higher default risk" if prediction else "Lower default risk",
        probability=round(probability, 4),
        probabilities=[round(1 - probability, 4), round(probability, 4)],
        threshold=round(THRESHOLD, 4),
        task=TASK,
        model_version=BUNDLE["model_version"],
        disclaimer="Research/teaching prototype only. Not a lending decision.",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
