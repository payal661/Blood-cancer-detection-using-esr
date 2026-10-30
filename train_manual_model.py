
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer

# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "LeukoAlert-project" / "leuk vs HC" / "dataset"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "WBC(10^9/L)",
    "RBC(10^12/L)",
    "HGB(g/L)",
    "HCT(%)",
    "MCV(fL)",
    "MCH(pg)",
    "MCHC(g/L)",
    "PLT(10^9/L)",
]
TARGET = "leukemia_label"
label_map = {
    "healthy": 0,
    "leukemia": 1,
}

# Load the three training files
train_dir = DATA_DIR / "training"
train_files = sorted(train_dir.glob("*.csv"))

if not train_files:
    raise FileNotFoundError(f"No training CSV files found in: {train_dir}")

train_df = pd.concat(
    [pd.read_csv(file) for file in train_files],
    ignore_index=True,
)

required = FEATURES + [TARGET]
missing = [column for column in required if column not in train_df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

X_train = train_df[FEATURES].apply(pd.to_numeric, errors="coerce")
y_train = (
    train_df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(label_map)
)

if y_train.isna().any():
    unknown_labels = train_df.loc[y_train.isna(), TARGET].unique()
    raise ValueError(f"Unexpected labels found: {unknown_labels}")

y_train = y_train.astype(int)

# Train a separate model specifically for manual CBC inputs
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)

# Save the model and its required input fields
model_path = MODEL_DIR / "manual_cbc_model.joblib"
joblib.dump(
    {"model": model, "features": FEATURES},
    model_path,
)

print(f"Model saved to: {model_path}")
print(f"Training rows: {len(train_df):,}")
print(f"Features used: {len(FEATURES)}")

# Evaluate on the separate test dataset
test_file = DATA_DIR / "test" / "test_site_true_world.csv"

if test_file.exists():
    test_df = pd.read_csv(test_file)
    X_test = test_df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y_test = (
        test_df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(label_map)
    )

    if y_test.isna().any():
        unknown_labels = test_df.loc[y_test.isna(), TARGET].unique()
        raise ValueError(f"Unexpected test labels found: {unknown_labels}")

    y_test = y_test.astype(int)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\nTest-set results at threshold 0.5")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

    if y_test.nunique() == 2:
        print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
else:
    print(f"Test file not found: {test_file}")
