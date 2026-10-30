
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier


# Project folders
ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    ROOT / "data" / "raw" / "LeukoAlert-project"
    / "leuk vs HC" / "dataset"
)

TRAIN_DIR = DATA_DIR / "training"
VALID_DIR = DATA_DIR / "validation"

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# Load CSV files
def load_csv_folder(folder):
    files = list(folder.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {folder}"
        )

    dataframes = [pd.read_csv(file) for file in files]
    return pd.concat(dataframes, ignore_index=True)


print("Loading data...")

train_df = load_csv_folder(TRAIN_DIR)
val_df = load_csv_folder(VALID_DIR)

TARGET = "leukemia_label"

if TARGET not in train_df.columns or TARGET not in val_df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

# Remove rows without a known target
train_df = train_df.dropna(subset=[TARGET]).copy()
val_df = val_df.dropna(subset=[TARGET]).copy()

# Separate features and target
X_train = train_df.drop(columns=[TARGET, "sample_id"], errors="ignore")
X_val = val_df.drop(columns=[TARGET, "sample_id"], errors="ignore")

y_train = train_df[TARGET]
y_val = val_df[TARGET]

# Convert labels to 0 and 1 if needed
def encode_target(series):
    if pd.api.types.is_numeric_dtype(series):
        values = set(series.dropna().unique())
        if values.issubset({0, 1}):
            return series.astype(int)

    mapping = {
        "healthy": 0,
        "hc": 0,
        "leukemia": 1,
        "leukaemia": 1,
        "leuk": 1
    }

    encoded = series.astype(str).str.strip().str.lower().map(mapping)

    if encoded.isna().any():
        unknown = series[encoded.isna()].unique()
        raise ValueError(f"Unrecognized target labels: {unknown}")

    return encoded.astype(int)


y_train = encode_target(y_train)
y_val = encode_target(y_val)

# Identify numeric and categorical columns
numeric_cols = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Preprocess missing values
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# Calculate positive-class weight from training data
positive = int((y_train == 1).sum())
negative = int((y_train == 0).sum())

if positive == 0:
    raise ValueError("No positive leukemia samples found.")

scale_weight = negative / positive

# Build XGBoost model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    ))
])

print("Training XGBoost...")
model.fit(X_train, y_train)

# Evaluate on validation data
predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, 1]

results = {
    "model": "XGBoost",
    "accuracy": accuracy_score(y_val, predictions),
    "precision": precision_score(
        y_val, predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, predictions, zero_division=0
    ),
    "f1_score": f1_score(
        y_val, predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_val, probabilities)
}

print("\nXGBoost validation results:")

for metric, value in results.items():
    if isinstance(value, (int, float, np.integer, np.floating)):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

print("\nConfusion matrix:")
print(confusion_matrix(y_val, predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=["healthy", "leukemia"],
        zero_division=0
    )
)

# Save model and results
joblib.dump(model, MODEL_DIR / "xgboost_model.joblib")

pd.DataFrame([results]).to_csv(
    MODEL_DIR / "xgboost_results.csv",
    index=False
)

print("\nSaved model to models/xgboost_model.joblib")
print("Saved metrics to models/xgboost_results.csv")

