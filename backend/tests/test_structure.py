"""Smoke test that needs no model: the project's main source files are in place."""
from pathlib import Path


def test_source_files_exist():
    root = Path(__file__).resolve().parents[2]
    assert (root / "backend/app.py").exists()
    assert (root / "backend/models/model_bundle.joblib").exists()
    assert (root / "frontend/package.json").exists()
    assert (root / "README.md").exists()
