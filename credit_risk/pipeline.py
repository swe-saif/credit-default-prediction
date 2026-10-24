"""Preprocessing and models.

Everything that learns from the data (column means, category codes) lives inside one scikit-learn
Pipeline. That way it is fitted on the training split only, and it is saved together with the model,
so a future API can feed raw applicant rows straight into the saved file.
"""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from xgboost import XGBClassifier

from . import config


def flag_days_employed(X):
    """Keep the fact that DAYS_EMPLOYED held the placeholder as a 0/1 column, then treat the value as missing."""
    X = X.copy()
    placeholder = X["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_PLACEHOLDER
    X["DAYS_EMPLOYED_ANOMALY"] = placeholder.astype(int)
    X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].mask(placeholder, np.nan)
    return X


def flag_days_employed_names(transformer, input_features):
    return [*input_features, "DAYS_EMPLOYED_ANOMALY"]


def build_preprocessor():
    """The paper's preprocessing steps, in a form that can be fitted on training data only."""
    numeric = SimpleImputer(strategy="mean")  # the paper: fill numbers with the column mean
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Data_Not_Available"),  # the paper's label for missing text
        # Same integer codes as LabelEncoder, but an unseen category becomes -1 instead of crashing.
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    )
    columns = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_exclude="number")),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return make_pipeline(
        FunctionTransformer(flag_days_employed, feature_names_out=flag_days_employed_names),
        columns,
    )


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
