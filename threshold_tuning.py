
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    ROOT / "data" / "raw" / "LeukoAlert-project"
    / "leuk vs HC" / "dataset"
)

MODEL_PATH = ROOT / "models" / "random_forest_model.joblib"
VALID_DIR = DATA_DIR / "validation"

model = joblib.load(MODEL_PATH)

files = list(VALID_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError(f"No validation CSV files found in {VALID_DIR}")

val_df = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True
)

TARGET = "leukemia_label"

val_df = val_df.dropna(subset=[TARGET]).copy()

X_val = val_df.drop(
    columns=[TARGET, "sample_id"],
    errors="ignore"
)
y_val = val_df[TARGET]

# Convert labels if stored as text
if not pd.api.types.is_numeric_dtype(y_val):
    mapping = {
        "healthy": 0,
        "hc": 0,
        "leukemia": 1,
        "leukaemia": 1,
        "leuk": 1
    }
    y_val = y_val.astype(str).str.strip().str.lower().map(mapping)

if y_val.isna().any():
    raise ValueError("Some target labels could not be converted.")

y_val = y_val.astype(int)

probabilities = model.predict_proba(X_val)[:, 1]

results = []

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    predictions = (probabilities >= threshold).astype(int)

    results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, predictions, zero_division=0
        ),
        "false_positives": int(
            confusion_matrix(
                y_val, predictions, labels=[0, 1]
            )[0, 1]
        ),
        "false_negatives": int(
            confusion_matrix(
                y_val, predictions, labels=[0, 1]
            )[1, 0]
        )
    })

results_df = pd.DataFrame(results)

print("\nRandom Forest threshold comparison:")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    ROOT / "models" / "threshold_results.csv",
    index=False
)

print("\nSaved results to models/threshold_results.csv")
