"""Paths and settings shared by every script."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The CSVs sit in the project root by default; set CREDIT_DATA_DIR to read them from elsewhere (e.g. a Docker volume).
DATA_DIR = Path(os.environ.get("CREDIT_DATA_DIR", ROOT))
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

APPLICATION_FILE = DATA_DIR / "application_train.csv"
BUREAU_FILE = DATA_DIR / "bureau.csv"

ID_COL = "SK_ID_CURR"
TARGET = "TARGET"
RANDOM_STATE = 42
TEST_SIZE = 0.20  # same 80:20 train/test split as the original notebook
VAL_SIZE = 0.20  # share of the training part held back for choosing each model's decision threshold

# DAYS_EMPLOYED uses 365243 (~1000 years) as a placeholder rather than a real duration: the paper's "outlier".
DAYS_EMPLOYED_PLACEHOLDER = 365243
