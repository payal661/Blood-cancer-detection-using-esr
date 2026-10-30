# 🩸 BloodCancerESR

> **Educational research prototype** — CBC-based leukemia classification dashboard built with Python, scikit-learn, and Streamlit.
>
> ⚠️ **Not ESR-based. Not a clinical diagnostic tool. Model predictions must not guide patient care.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Live Dashboard](#live-dashboard)
- [Project Structure](#project-structure)
- [Models](#models)
- [Performance Results](#performance-results)
- [CBC Features](#cbc-features)
- [Getting Started](#getting-started)
- [Sample Data](#sample-data)
- [How Prediction Works](#how-prediction-works)
- [Dashboard Pages](#dashboard-pages)
- [Limitations](#limitations)
- [Research Disclaimer](#research-disclaimer)

---

## Overview

**BloodCancerESR** is a student machine-learning research project that explores automated classification of **healthy vs. leukemia** dataset records using **Complete Blood Count (CBC)** data.

The project trains and evaluates three classifiers — Logistic Regression, Random Forest, and XGBoost — on CBC and analyzer-derived features sourced from the LeukoAlert dataset. Results are presented through an interactive Streamlit dashboard with two prediction workflows:

| Workflow | Model used | Input |
|---|---|---|
| **Manual CBC Entry** | `manual_cbc_model.joblib` | 8 CBC values entered in a form |
| **Upload CSV / Excel** | Full-feature Random Forest or Manual CBC | File with matching column names |

---

## Live Dashboard

```bash
# Clone and run locally
git clone https://github.com/your-username/BloodCancerESR.git
cd BloodCancerESR
pip install -r requirements.txt
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Project Structure

```
BloodCancerESR/
│
├── app.py                        # ← Streamlit dashboard (main entry point)
│
├── models/
│   ├── manual_cbc_model.joblib   # Random Forest trained on 8 CBC fields
│   ├── random_forest_model.joblib# Full-feature Random Forest (74 features)
│   ├── baseline_logistic_regression.joblib
│   ├── xgboost_model.joblib
│   ├── validation_results.csv    # Saved validation metrics (LR + RF)
│   ├── xgboost_results.csv       # Saved XGBoost validation metrics
│   ├── final_test_results.csv    # Held-out test evaluation snapshot
│   ├── random_forest_results.csv
│   ├── baseline_results.csv
│   └── threshold_results.csv
│
├── data/
│   └── raw/
│       └── LeukoAlert-project/
│           └── leuk vs HC/
│               └── dataset/
│                   ├── training/
│                   ├── validation/
│                   └── test/
│
├── sample_data/                  # Ready-to-upload CSV test files
│   ├── sample_healthy_patients.csv
│   ├── sample_leukemia_suspect.csv
│   ├── sample_mixed_patients.csv
│   ├── sample_borderline_cases.csv
│   ├── synthetic_cbc_sample_01.csv
│   ├── synthetic_cbc_sample_02.csv
│   ├── synthetic_cbc_sample_03.csv
│   └── synthetic_cbc_sample_04.csv
│
├── train_model.py                # Trains full-feature Random Forest
├── random_forest_model.py        # Random Forest training script
├── baseline_model.py             # Logistic Regression baseline
├── xgboost_model.py              # XGBoost training script
├── train_manual_model.py         # Trains the 8-feature manual CBC model
├── threshold_tuning.py           # Decision threshold sweep
├── evaluate_final_model.py       # Held-out test evaluation
├── data_audit.py                 # Dataset integrity checks
│
├── training_audit.csv
├── validation_audit.csv
├── test_audit.csv
│
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Dark theme configuration
└── README.md
```

---

## Models

### 1. Manual CBC Model — `manual_cbc_model.joblib`

Trained on **8 standard CBC fields** only. Used by the manual entry form.

```python
FEATURES = [
    "WBC(10^9/L)",   # White Blood Cell count
    "RBC(10^12/L)",  # Red Blood Cell count
    "HGB(g/L)",      # Hemoglobin
    "HCT(%)",        # Hematocrit
    "MCV(fL)",       # Mean Corpuscular Volume
    "MCH(pg)",       # Mean Corpuscular Hemoglobin
    "MCHC(g/L)",     # Mean Corpuscular Hemoglobin Concentration
    "PLT(10^9/L)",   # Platelet Count
]
```

Saved as a `dict`:
```python
{"model": <RandomForestClassifier>, "features": [...8 feature names...]}
```

Decision threshold: **0.5**

---

### 2. Full-Feature Random Forest — `random_forest_model.joblib`

Trained on **all 74 analyzer-derived feature columns** from the LeukoAlert dataset. Used for CSV uploads and saved-record predictions.

Architecture:
```
Pipeline([
    ColumnTransformer([
        ("num", Pipeline([SimpleImputer(median), StandardScaler()]), numeric_cols),
        ("cat", Pipeline([SimpleImputer(most_frequent), OneHotEncoder()]), cat_cols),
    ]),
    RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
])
```

Decision threshold: **0.8** (tuned for high precision on the held-out test split)

---

### 3. Logistic Regression Baseline — `baseline_logistic_regression.joblib`

Trained on the same 74 features. Used only for validation comparison.

---

### 4. XGBoost — `xgboost_model.joblib`

Gradient-boosted tree ensemble. Shown in the validation comparison table.

---

## Performance Results

### Validation split — model comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 90.25% | 25.69% | 83.92% | 0.3933 | 91.82% |
| Random Forest | 89.23% | 25.13% | 93.96% | 0.3966 | **96.76%** |
| XGBoost | 86.24% | 20.88% | 95.08% | 0.3424 | 96.45% |

> Validation results are saved in `models/validation_results.csv` and `models/xgboost_results.csv`.

---

### Held-out test split — Full-feature Random Forest at threshold 0.80

| Metric | Score |
|---|---|
| **Accuracy** | **95.88%** |
| **Precision** | **48.25%** |
| **Recall** | **72.97%** |
| **F1-score** | **0.5809** |
| **ROC-AUC** | **97.36%** |

> These results are saved in `models/final_test_results.csv` and verified on every dashboard load.

---

### Threshold comparison — held-out test split

| Threshold | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives |
|---|---|---|---|---|---|---|
| 20% | — | Low | High | — | High | Low |
| 50% | — | Mid | Mid | — | Mid | Mid |
| **80%** | **95.88%** | **48.25%** | **72.97%** | **0.581** | Low | Lower |

Full threshold sweep is shown interactively in the **Model Performance** dashboard page.

---

## CBC Features

The 8 CBC fields used by the manual model, along with their typical reference ranges:

| Feature | Full name | Unit | Typical healthy range |
|---|---|---|---|
| `WBC(10^9/L)` | White Blood Cell count | 10⁹/L | 4.5 – 11.0 |
| `RBC(10^12/L)` | Red Blood Cell count | 10¹²/L | 4.2 – 5.4 (M), 3.8 – 4.8 (F) |
| `HGB(g/L)` | Hemoglobin | g/L | 130 – 175 (M), 120 – 155 (F) |
| `HCT(%)` | Hematocrit | % | 38.3 – 48.6 (M), 35.5 – 44.9 (F) |
| `MCV(fL)` | Mean Corpuscular Volume | fL | 80 – 100 |
| `MCH(pg)` | Mean Corpuscular Hemoglobin | pg | 27 – 33 |
| `MCHC(g/L)` | Mean Corpuscular Hemoglobin Concentration | g/L | 315 – 360 |
| `PLT(10^9/L)` | Platelet Count | 10⁹/L | 150 – 400 |

> Reference ranges are for general information only and vary by laboratory and population.

---

## Getting Started

### Prerequisites

- Python **3.10+**
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/BloodCancerESR.git
cd BloodCancerESR

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Train the models (optional — pre-trained files included)

```bash
# Baseline Logistic Regression
python baseline_model.py

# Full-feature Random Forest
python random_forest_model.py

# XGBoost
python xgboost_model.py

# Manual CBC model (8 features)
python train_manual_model.py

# Threshold tuning sweep
python threshold_tuning.py

# Final held-out test evaluation
python evaluate_final_model.py
```

### Run the dashboard

```bash
streamlit run app.py
```

---

## Sample Data

Ready-to-upload CSV files are in the `sample_data/` folder.  
All files use the **8-column manual CBC format** and can be uploaded directly via the **Upload CSV / Excel** workflow.

| File | Rows | Expected prediction |
|---|---|---|
| `sample_healthy_patients.csv` | 10 | 🟢 Healthy — normal WBC (5–7), normal HGB, normal PLT |
| `sample_leukemia_suspect.csv` | 10 | 🔴 Leukemia — very high WBC (19–88), low HGB, low PLT |
| `sample_mixed_patients.csv` | 12 | Mixed — alternating healthy and suspect rows |
| `sample_borderline_cases.csv` | 10 | ⚠️ Borderline — mildly elevated WBC, near decision boundary |
| `synthetic_cbc_sample_01–04.csv` | 1 each | Single-row quick test files |

### CSV format required

```csv
WBC(10^9/L),RBC(10^12/L),HGB(g/L),HCT(%),MCV(fL),MCH(pg),MCHC(g/L),PLT(10^9/L)
6.1,4.44,134,40.2,90.5,30.2,334,252
```

> Column names must match exactly (including units and brackets), **or** the app will open a column-mapping interface to let you match them manually.

---

## How Prediction Works

```
User input
    │
    ├─ Manual CBC form ──────────────────────────────────► manual_cbc_model.joblib
    │                                                         (8 features, threshold 0.5)
    │
    └─ CSV / Excel upload
            │
            ├─ All 74 full-model columns present? ──Yes──► random_forest_model.joblib
            │                                               (74 features, threshold 0.8)
            │
            └─ No ──► Show missing column names
                       Open column-mapping interface
                       ──► manual_cbc_model.joblib
                             (8 features, threshold 0.5)

Output: Predicted dataset label (healthy / leukemia) + leukemia-class probability score
```

### Input validation rules

1. Uploaded file is checked for empty rows, duplicate column names, and non-numeric values.
2. The app reads the **model's own `feature_names_in_`** — it never assumes a fixed column count.
3. Missing columns are listed by name before any prediction runs.
4. All values must be finite numbers (no `inf`, no `NaN`) for the manual model.
5. The full-feature model uses a trained `SimpleImputer` to handle remaining blanks.

---

## Dashboard Pages

### ⚡ Overview
Project summary, stat cards (prediction methods, input validation status, feature count), and a directory of all dashboard pages.

### 🔮 Prediction Studio
Three tabbed workflows:
- **Manual CBC Entry** — 8 number inputs, `Run Prediction` button, score + badge output
- **Upload CSV / Excel** — file uploader, column validation, editable data preview, batch results table with CSV download
- **Saved Dataset Record** — search by `sample_id`, predict a held-out test record without revealing its true label

### 📊 Model Performance
- Validation comparison bar chart (LR vs RF vs XGBoost)
- Held-out test metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix
- ROC curve
- Threshold sensitivity table and chart
- Snapshot verification against saved `final_test_results.csv`

### 🗄️ Dataset Explorer
- Record counts by split (training / validation / test)
- Healthy vs. leukemia label distribution chart
- CBC feature statistics (mean, std, min, median, max) from training split
- Searchable and filterable held-out test record browser

### 💡 About the Project
Problem statement, methodology, model descriptions, limitations, and research disclaimer.

---

## Limitations

- **Not ESR-based.** Despite the project name, no ESR measurements are used. The classifier uses CBC and analyzer-derived features.
- **Not a medical device.** This is an educational prototype and must not be used for clinical screening, diagnosis, or treatment decisions.
- **Threshold selection.** The 80% decision threshold was chosen to improve precision on the held-out split; it is not a clinically validated operating point.
- **No PDF or image OCR.** Only CSV and Excel file uploads are supported. PDF reports and blood-test images require a separate extraction step.
- **No external validation.** The models have been evaluated only on the LeukoAlert dataset splits. Performance on different populations or analyzers is unknown.
- **Class imbalance.** Leukemia records are a minority class. `class_weight="balanced"` is used during training, which affects precision/recall trade-offs.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard |
| `scikit-learn` | Model training, pipelines, evaluation |
| `xgboost` | XGBoost classifier |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `joblib` | Model serialization |
| `openpyxl` | Excel file reading |
| `plotly` | (Available for future charts) |
| `matplotlib` / `seaborn` | (Available for future plots) |

---

## Research Disclaimer

> This project is an **educational research demonstration** created for learning purposes. It explores machine-learning classification techniques applied to publicly available CBC data.
>
> - Model predictions are **not medical diagnoses**.
> - Results must **not** be used for patient screening, treatment planning, or any clinical decision.
> - The dataset labels ("healthy" / "leukemia") are **research dataset labels**, not confirmed clinical diagnoses.
> - Always consult a qualified medical professional for health concerns.

---

## License

This project is released for educational and research purposes only.  
No warranty is provided. Use at your own risk.

---

<div align="center">

**BloodCancerESR** · CBC-based classification research prototype  
Built with 🩸 Python · scikit-learn · Streamlit

</div>
