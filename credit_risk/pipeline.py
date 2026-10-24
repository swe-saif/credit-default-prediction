"""The three models, each wrapped in one scikit-learn Pipeline with the preprocessing.

Everything that learns from the data (column means, category codes) lives inside the Pipeline. That way it
is fitted on the training split only, and it is saved together with the model, so the API can feed raw
applicant rows straight into the saved file. The preprocessing itself is in features.py.
"""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config
from .features import build_preprocessor


def build_models():
    """The original notebook's three models with the same settings, so any change in results comes from the data handling."""
    seed = config.RANDOM_STATE
    return {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "lightgbm": LGBMClassifier(n_estimators=100, random_state=seed, n_jobs=-1, verbose=-1),
        "xgboost": XGBClassifier(n_estimators=100, random_state=seed, n_jobs=-1, eval_metric="logloss"),
    }


def build_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
