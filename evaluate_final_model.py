
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    ROOT / "data" / "raw" / "LeukoAlert-project"
    / "leuk vs HC" / "dataset"
)

TEST_FILE = DATA_DIR / "test" / "test_site_true_world.csv"
MODEL_FILE = ROOT / "models" / "random_forest_model.joblib"
OUTPUT_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.8
TARGET = "leukemia_label"

print("Loading saved Random Forest model...")
model = joblib.load(MODEL_FILE)

print("Loading untouched test dataset...")
test_df = pd.read_csv(TEST_FILE)
test_df = test_df.dropna(subset=[TARGET]).copy()

X_test = test_df.drop(
    columns=[TARGET, "sample_id"],
    errors="ignore"
)
y_test = test_df[TARGET]

# Convert labels to 0 = healthy, 1 = leukemia
if not pd.api.types.is_numeric_dtype(y_test):
    mapping = {
        "healthy": 0,
        "hc": 0,
        "leukemia": 1,
        "leukaemia": 1,
        "leuk": 1
    }
    y_test = y_test.astype(str).str.strip().str.lower().map(mapping)

if y_test.isna().any():
    raise ValueError("Unknown target labels found in test data.")

y_test = y_test.astype(int)

print("Generating predictions...")
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

print(f"\nTest results at threshold {THRESHOLD}:")

results = {
    "model": "Random Forest",
    "threshold": THRESHOLD,
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "f1_score": f1_score(
        y_test, predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_test, probabilities)
}

for metric, value in results.items():
    if isinstance(value, (int, float, np.integer, np.floating)):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

print("\nConfusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["healthy", "leukemia"],
    zero_division=0
))

pd.DataFrame([results]).to_csv(
    OUTPUT_DIR / "final_test_results.csv",
    index=False
)

print("\nSaved results to models/final_test_results.csv")
