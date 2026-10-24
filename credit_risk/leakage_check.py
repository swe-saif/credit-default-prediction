"""Re-run the original notebook's setup and test it for leakage.

The original joined every bureau record onto its application (1.5M rows for 307K applicants), preprocessed
everything, then split by row. This script repeats that exactly, then trains the same models a second time
with a split that keeps all of an applicant's rows on the same side. If the scores collapse, the original
results came from seeing the same applicants in training and testing.

Run from the project root:  python -m credit_risk.leakage_check
Outputs: reports/leakage_check.json, reports/figures/leakage_check.png
"""
import gc
import json
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config, data, pipeline, plots


def original_preprocessing(application, bureau):
    """The original notebook's steps, unchanged: row-level join, then fit means and codes on all rows."""
    df = application.merge(bureau, on=config.ID_COL, how="left", suffixes=("", "_BUREAU"))
    del application, bureau  # 1.5M rows is heavy; free the inputs early
    groups = df[config.ID_COL].to_numpy()
    df = df.drop(columns=["SK_ID_BUREAU", config.ID_COL])
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(config.DAYS_EMPLOYED_PLACEHOLDER, np.nan)

    text_cols = df.select_dtypes(exclude="number").columns
    for col in df.columns.difference(text_cols).drop(config.TARGET):
        df[col] = df[col].fillna(df[col].mean())
    for col in text_cols:
        df[col] = LabelEncoder().fit_transform(df[col].fillna("Data_Not_Available").astype(str))

    y = df.pop(config.TARGET).to_numpy()
    X = df.astype(np.float32)  # the tree models work in float32 internally, so this only saves memory
    return X, y, groups


def evaluate_split(X, y, train_idx, test_idx):
    results = {}
    for name, model in pipeline.build_models().items():
        t0 = time.perf_counter()
        model.fit(X.iloc[train_idx], y[train_idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)  # the original used predict(), i.e. a 0.5 threshold
        y_test = y[test_idx]
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "train_seconds": round(time.perf_counter() - t0, 1),
        }
        r = results[name]
        print(f"    {name:<14} accuracy {r['accuracy']:.4f} | recall {r['recall']:.4f} | ROC-AUC {r['roc_auc']:.4f} | {r['train_seconds']:.0f}s")
        del model
        gc.collect()  # the full-depth Random Forest alone is ~800 MB
    return results


def main():
    start = time.perf_counter()
    seed = config.RANDOM_STATE

    print("1/3 Rebuilding the original row-level dataset")
    X, y, groups = original_preprocessing(data.load_csv(config.APPLICATION_FILE), data.load_csv(config.BUREAU_FILE))
    gc.collect()
    print(f"  {len(X):,} rows x {X.shape[1]} features from {len(np.unique(groups)):,} applicants")

    row_train, row_test = train_test_split(np.arange(len(y)), test_size=0.20, random_state=seed, stratify=y)
    app_train, app_test = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed).split(X, y, groups))
    splits = {"row_split": (row_train, row_test), "applicant_split": (app_train, app_test)}

    report = {"rows": len(X), "applicants": int(len(np.unique(groups))), "splits": {}}
    for key, (train_idx, test_idx) in splits.items():
        seen = np.isin(groups[test_idx], groups[train_idx])
        report["splits"][key] = {"train_rows": len(train_idx), "test_rows": len(test_idx),
                                 "test_rows_whose_applicant_is_also_in_train": seen.mean()}

    results = {}
    for step, key in enumerate(splits, start=2):
        train_idx, test_idx = splits[key]
        print(f"{step}/3 {key}: {report['splits'][key]['test_rows_whose_applicant_is_also_in_train']:.1%} "
              "of test rows belong to applicants also in training")
        results[key] = evaluate_split(X, y, train_idx, test_idx)

    report.update(results)
    report["runtime_seconds"] = round(time.perf_counter() - start, 1)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "leakage_check.json").write_text(json.dumps(report, indent=2))
    print("  saved leakage_check.json")
    (config.REPORTS_DIR / "figures").mkdir(exist_ok=True)
    plots.leakage_bars(results, config.REPORTS_DIR / "figures" / "leakage_check.png")
    print(f"Done in {report['runtime_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
