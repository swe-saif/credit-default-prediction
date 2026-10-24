# Credit default prediction (Home Credit)

[![CI](https://github.com/swe-saif/credit-default-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/swe-saif/credit-default-prediction/actions/workflows/ci.yml)
[![Deploy to GitHub Pages](https://github.com/swe-saif/credit-default-prediction/actions/workflows/deploy.yml/badge.svg)](https://github.com/swe-saif/credit-default-prediction/actions/workflows/deploy.yml)

A replication of *Predicting Credit Paying Ability With Machine Learning Algorithms* (Febriyanti & Izzalqurny, 2023), extended with XGBoost, a data-leakage fix, and a full-stack application: a Flask API serving the model, an Angular form, and GitHub Actions for CI and deployment to GitHub Pages.

**Live demo (static):** https://swe-saif.github.io/credit-default-prediction/. GitHub Pages hosts only the Angular page; predictions need the Flask API running locally (see [Run the application](#run-the-application-locally)).

## Architecture

```
Browser: Angular form (http://localhost:4200)
   |  POST /api/predict (JSON)
   v
Angular dev proxy  ->  Flask API (http://127.0.0.1:5000)
                         validation -> saved preprocessing -> LightGBM
   <-  JSON: prediction, probability, threshold

git push -> GitHub Actions CI (API tests + Angular tests and build)
         -> on success: GitHub Pages deployment (static Angular only)
```

## Project layout

```
backend/                Flask API
  app.py                GET /health, POST /predict
  models/model_bundle.joblib   deployed model: pipeline + input schema + threshold (~150 KB)
  tests/                automated tests (pytest)
  make_payload.py, test_pipeline.py, test_api.py, test_payload.json   manual checks
  requirements.txt      pinned API dependencies
frontend/               Angular 22 app
  src/app/app.ts|html|css      the prediction form
  src/app/feature-schema.ts    input schema, generated from the model bundle
  src/app/prediction.service.ts
  proxy.conf.json       forwards /api/* to Flask during development
credit_risk/            research and model-building code
  data.py, features.py, pipeline.py, evaluate.py, plots.py
  train.py              research experiment (3 models, 186 inputs)
  leakage_check.py      re-runs the original 1.5M-row setup to measure the leakage
  export_model.py       trains and exports the deployed 15-input model
data/sample_inputs.csv  two synthetic applicants (not real people)
reports/                metrics.json, leakage_check.json, deploy_model.json, figures/
docs/environment.txt    tool and library versions
.github/workflows/      ci.yml (CI), deploy.yml (CD to GitHub Pages)
requirements.txt        pinned research/training dependencies
```

## Run the application locally

Needs Python 3.14 and Node.js 22+. Commands are for PowerShell in VS Code, run from the project root.

**One-time setup**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r backend/requirements.txt
cd frontend; npm ci; cd ..
```

**Terminal 1: Flask API** (keep running)

```powershell
.\.venv\Scripts\Activate.ps1
python backend\app.py
```

**Terminal 2: Angular** (keep running; the proxy is configured in `angular.json`)

```powershell
cd frontend
npm start
```

Open http://localhost:4200, click **Typical applicant** or **Higher-risk profile** to fill the form, then **Run local prediction**.

**Terminal 3: checks**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest                       # 9 API tests
python backend\test_pipeline.py        # model loads and reproduces the training-time answer
python backend\test_api.py             # live request to the running Flask server
curl.exe --noproxy "*" -i http://localhost:4200/api/health   # Angular proxy -> Flask
cd frontend; npx ng test --watch=false # 3 frontend tests
```

## The API

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok", "features": 15, "task": "classification", "model_version": "1.0"}` |
| `POST /predict` | JSON object with all 15 inputs (optional ones may be `null`) |

Example response:

```json
{"prediction": 1, "label": "Higher default risk", "probability": 0.1779, "probabilities": [0.8221, 0.1779],
 "threshold": 0.1378, "task": "classification", "model_version": "1.0",
 "disclaimer": "Research/teaching prototype only. Not a lending decision."}
```

Inputs are checked against the schema saved with the model: unknown or missing keys, blank required inputs, numbers outside the training range and unknown categories return HTTP 400 with the reason.

## The deployed model

The research model uses 186 inputs, too many for a form. `credit_risk/export_model.py` keeps the 15 most important inputs (by LightGBM gain), leaving out gender because it is a protected attribute in lending, and retrains the same pipeline on the same training split.

| Test set (61,503 applicants) | Deployed LightGBM, 15 inputs | Research LightGBM, 186 inputs |
|---|---|---|
| ROC-AUC | 0.753 | 0.764 |
| PR-AUC | 0.245 | 0.259 |
| Threshold (chosen on validation) | 0.14 | 0.17 |
| Precision | 22.8% | 27.0% |
| Recall | 45.1% | 38.1% |
| F1 | 30.3% | 31.6% |

8 inputs are required; 7 may be left blank because they are often missing in real applications (no car, no credit-bureau history, missing external scores), and the pipeline fills them as it did in training. The form's schema in `frontend/src/app/feature-schema.ts` is generated from the same bundle, so the API and the form cannot drift apart. Full numbers: [reports/deploy_model.json](reports/deploy_model.json).

To rebuild it (needs the Kaggle CSVs in the project root): `python -m credit_risk.export_model`, then `python backend/make_payload.py`.

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| [ci.yml](.github/workflows/ci.yml) | push / pull request to `main` | **Backend:** install pinned dependencies, compile-check Python, run the 9 API tests against the committed model. **Frontend:** `npm ci`, unit tests, production build, upload the build as an artifact |
| [deploy.yml](.github/workflows/deploy.yml) | CI finished successfully on a push to `main` | Builds the exact commit CI tested with `--base-href /<repo>/` and deploys it to GitHub Pages |

One-time GitHub setting: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

---

## Research: reproducing the experiments

Needs the two Kaggle files `application_train.csv` and `bureau.csv` in the project folder (or point `CREDIT_DATA_DIR` at another folder).

```powershell
python -m credit_risk.train            # ~1 minute: 3 models on 186 inputs
python -m credit_risk.leakage_check    # ~10 minutes, optional
python -m credit_risk.export_model     # deployed 15-input model
```

### What changed from the original notebook, and why

| Original | Problem | Now |
|---|---|---|
| Joined every bureau record onto its application: 1,509,345 rows for 307,511 applicants | Each applicant copied up to 116 times; copies of the same person in both train and test | Bureau records summarised per applicant (count, active/closed, totals, mean/max/min), then joined: one row per applicant |
| Means and category codes fitted on all data, then split | Test data influenced preprocessing | Split first; preprocessing lives inside a Pipeline fitted on training data only |
| `DAYS_EMPLOYED` placeholder 365243 replaced by the mean | Loses the information that the value was a placeholder | Extra 0/1 column `DAYS_EMPLOYED_ANOMALY`, then mean-filled as before |
| `LabelEncoder` | Crashes on a category it has not seen | `OrdinalEncoder` with unknown categories mapped to -1 |
| Fixed 0.5 decision threshold | With 8% defaulters almost nobody is flagged (LightGBM recall 3%) | Threshold with the best F1 picked on a validation set; test set used once |
| Chart numbers typed in by hand | Figures go stale when results change | Figures drawn from the computed results |
| Colab-only code, unpinned libraries, 807 MB model | Not reproducible outside Colab; too big for git | Plain scripts, pinned requirements, compressed model files |

Unchanged on purpose: the same two data files, the paper's mean / `Data_Not_Available` imputation, the same three models with the same settings (`n_estimators=100`, `random_state=42`), and the same 80:20 stratified test split. Differences in results therefore come from the data handling, not from new modelling choices.

### Results

307,511 applicants, 186 input columns, 8.07% defaulted. Split: 196,806 train / 49,202 validation / 61,503 test.

| Test set | Random Forest | LightGBM | XGBoost |
|---|---|---|---|
| ROC-AUC | 0.710 | **0.764** | 0.751 |
| PR-AUC | 0.204 | **0.259** | 0.232 |
| Threshold (chosen on validation) | 0.17 | 0.17 | 0.16 |
| Precision | 21.1% | **27.0%** | 24.1% |
| Recall | 38.4% | 38.1% | **39.1%** |
| F1 | 27.2% | **31.6%** | 29.8% |
| Accuracy | 83.5% | **86.7%** | 85.2% |
| Accuracy at 0.5 threshold | 91.9% | 92.0% | 91.9% |
| Recall at 0.5 threshold | 0.3% | 2.6% | 4.5% |

A model that always predicts "non-default" scores **91.9%** accuracy, so accuracy says almost nothing here; ROC-AUC and PR-AUC are the main measures.

- **LightGBM is the best model**, then XGBoost. Random Forest is clearly behind: with default settings its trees grow until they memorise the training rows.
- At the 0.5 threshold all three reach ~92% accuracy with very low recall, the same pattern the paper reports.
- At the chosen threshold the models catch about 38–39% of defaulters; in exchange, 9–13% of good applicants are flagged.

Figures: [model comparison](reports/figures/model_comparison.png), [ROC curves](reports/figures/roc_curves.png), [confusion matrices](reports/figures/confusion_matrices.png). Full numbers: [reports/metrics.json](reports/metrics.json).

### Leakage check (original 1.5M-row setup)

`leakage_check.py` rebuilds the original notebook's data exactly (row-level join, preprocessing on all rows, same models) and evaluates it with two splits of the same 1,509,345 rows:

- **Row split** (the original): 93.5% of test rows belong to applicants who also have rows in training.
- **Applicant split** (`GroupShuffleSplit` on `SK_ID_CURR`): 0% overlap. Every applicant is entirely in train or entirely in test.

| Model | ROC-AUC, row split | ROC-AUC, applicant split | Recall at 0.5, row split | Recall at 0.5, applicant split |
|---|---|---|---|---|
| Random Forest | 0.994 | 0.712 | 81.1% | 0.4% |
| LightGBM | 0.812 | 0.757 | 3.0% | 2.1% |
| XGBoost | 0.860 | 0.745 | 13.2% | 5.0% |

- The row split reproduces the Colab results: XGBoost's confusion matrix matches exactly, and Random Forest gets 98.51% accuracy and 0.994 ROC-AUC (Colab: 98.45%, 0.994). The small differences come from newer library versions.
- **The original 98% Random Forest result was leakage.** Once it cannot see the same applicants, its ROC-AUC falls from 0.994 to 0.712 and its recall from 81% to 0.4%. Fully grown trees had memorised individual applicants.
- The boosted models were inflated too (LightGBM 0.81 → 0.76, XGBoost 0.86 → 0.74), only less.
- The honest row-level scores are close to the one-row-per-applicant results above, so summarising bureau records loses nothing and gives one prediction per applicant.

Figure: [leakage check](reports/figures/leakage_check.png). Full numbers: [reports/leakage_check.json](reports/leakage_check.json).

### Comparison with the paper

The paper ([Febriyanti & Izzalqurny, 2023](https://jurnal.stiepas.ac.id/index.php/bistek/article/view/296)) evaluates at the default 0.5 threshold. Scored the same way, our fixed pipeline reproduces its results:

| Test set | Paper: Random Forest | Paper: LightGBM | Ours: Random Forest | Ours: LightGBM | Ours: XGBoost |
|---|---|---|---|---|---|
| Accuracy | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 |
| Recall | 0.00 | 0.00 | 0.00 | 0.03 | 0.04 |
| Precision | 0.73 | 0.73 | 0.61 | 0.60 | 0.48 |
| AUC from 0/1 predictions (the paper's method) | 0.50 | 0.50 | 0.50 | 0.51 | 0.52 |
| ROC-AUC from probabilities (correct method) | not reported | not reported | 0.710 | 0.764 | 0.751 |

Paper values are from its Table 2 and Figure 5; ours are at the 0.5 threshold.

**Errors in the paper's results**, found by recomputing its metrics from its own confusion matrices (Figure 5):

- **AUC was computed from 0/1 predictions instead of probabilities.** Every reported AUC equals (recall + specificity) / 2, e.g. Random Forest (0.0015 + 0.99995) / 2 = 0.50. A model that almost never predicts "default" always gets about 0.50 this way, however well it ranks applicants.
- **The F1 column is wrong.** It lists 0.02 for all four models; the confusion matrices give 0.02 for Logistic Regression, 0.15 for Decision Tree, and 0.003 for Random Forest and LightGBM.
- **Random Forest and LightGBM have identical confusion matrices**, `[[84750, 4], [7491, 11]]`. Two different algorithms making the same predictions on 92,256 rows is practically impossible, so the same output was most likely reported twice. The conclusion that both are equally best rests on this.
- **91.88% accuracy is presented as a good result**, but always predicting "non-default" scores about 91.9% on this data.

**Differences in data and setup:**

- **Bureau file cut off.** The paper lists 1,048,575 bureau rows, which is exactly Excel's row limit (1,048,576 minus the header row). The real file has 1,716,428, so the paper used about 61% of the bureau records.
- **Join not explained.** The paper reports 307,513 rows after a left join, about one per applicant, but the left join it describes produces 1,509,345 rows on the full files. Its 55,374 `DAYS_EMPLOYED` outliers also match one row per applicant (we count 55,374 in `application_train.csv` alone).
- **70:30 split.** Not stated in the paper, but its confusion matrices sum to 92,256 test rows, exactly 30% of 307,513. We use 80:20.
- **No hyperparameters or random seed reported**, so an exact numeric match is not possible; matching the pattern of results is.

**Conclusion:** under the paper's own evaluation, this replication reproduces its results (accuracy 0.92, recall near 0, label-based AUC about 0.50). Evaluated correctly, ROC-AUC is 0.71–0.76, LightGBM is the best model rather than tied with Random Forest, and lowering the threshold to about 0.17 catches 38–39% of defaulters instead of almost none.

## Limitations

- Uses 2 of Home Credit's 7 tables, as the paper did; previous applications and payment history would likely improve results.
- Model settings are the paper's defaults, not tuned.
- The threshold maximises F1. A lender would set it from the real cost of a missed defaulter versus a rejected good customer.
- No fairness or probability-calibration checks yet; not fit for real lending decisions.
- GitHub Pages is static: the public site cannot run predictions. A public demo would need the Flask API hosted separately behind HTTPS.
- The Kaggle data may not be redistributed, so it is not in this repository; the sample inputs are synthetic profiles.
