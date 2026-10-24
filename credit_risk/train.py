"""Train and evaluate the three models on one row per applicant.

Run from the project root:  python -m credit_risk.train
Outputs: models/<name>.joblib, reports/metrics.json, reports/figures/*.png
"""
import json
import platform
import time

import joblib
import lightgbm
import pandas
import sklearn
import xgboost
from sklearn.model_selection import train_test_split

from . import config, data, evaluate, pipeline, plots


def main():
    start = time.perf_counter()
    seed = config.RANDOM_STATE

    print("1/4 Building the applicant table")
    application = data.load_csv(config.APPLICATION_FILE)
    bureau = data.load_csv(config.BUREAU_FILE)
    table = data.build_applicant_table(application, bureau)
    del application, bureau
    X = table.drop(columns=[config.ID_COL, config.TARGET])
    y = table[config.TARGET]
    print(f"  {len(table):,} applicants x {X.shape[1]} input columns, {y.mean():.2%} defaulted")

    print("2/4 Splitting first, so nothing is fitted on validation or test data")
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=config.VAL_SIZE, stratify=y_rest, random_state=seed
    )
    print(f"  train {len(X_train):,} | validation {len(X_val):,} | test {len(X_test):,}")

    print("3/4 Training")
    config.MODELS_DIR.mkdir(exist_ok=True)
    results, test_probs = {}, {}
    for name, model in pipeline.build_models().items():
        t0 = time.perf_counter()
        pipe = pipeline.build_pipeline(model).fit(X_train, y_train)
        threshold = evaluate.pick_threshold(y_val, pipe.predict_proba(X_val)[:, 1])
        test_prob = pipe.predict_proba(X_test)[:, 1]
        test_probs[name] = test_prob

        results[name] = {
            "threshold": threshold,
            "train_seconds": round(time.perf_counter() - t0, 1),
            "test": {
                **evaluate.ranking_scores(y_test, test_prob),
                "at_tuned_threshold": evaluate.score(y_test, test_prob, threshold),
                "at_0.5": evaluate.score(y_test, test_prob, 0.5),
            },
        }
        joblib.dump(
            {"pipeline": pipe, "threshold": threshold, "input_columns": list(X.columns)},
            config.MODELS_DIR / f"{name}.joblib",
            compress=3,
        )
        test = results[name]["test"]
        print(
            f"  {name:<14} ROC-AUC {test['roc_auc']:.4f} | recall {test['at_tuned_threshold']['recall']:.1%} "
            f"at threshold {threshold:.3f} | {results[name]['train_seconds']:.0f}s"
        )

    print("4/4 Writing reports")
    figures = config.REPORTS_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": {
            "applicants": len(table),
            "input_columns": X.shape[1],
            "default_rate": y.mean(),
            "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
            "random_state": seed,
        },
        # A model that always says "non-default" already scores this accuracy, which is why accuracy alone misleads.
        "baseline_accuracy_always_non_default": 1 - y_test.mean(),
        "models": results,
        "environment": {
            "python": platform.python_version(),
            "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__,
        },
        "runtime_seconds": round(time.perf_counter() - start, 1),
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print("  saved metrics.json")
    plots.metric_bars(results, figures / "model_comparison.png")
    plots.roc_curves(y_test, test_probs, figures / "roc_curves.png")
    plots.confusion_matrices(results, figures / "confusion_matrices.png")
    print(f"Done in {report['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
