
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.base import clone
import joblib

# Project paths
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "raw" / "LeukoAlert-project" / "leuk vs HC" / "dataset"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "leukemia_label"
ID_COLUMN = "sample_id"

def load_split(split_name):
    folder = DATA / split_name
    files = sorted(folder.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    frames = [pd.read_csv(file) for file in files]
    return pd.concat(frames, ignore_index=True)

# Load training and validation data only
train_df = load_split("training")
val_df = load_split("validation")

if TARGET not in train_df.columns or TARGET not in val_df.columns:
    raise ValueError(f"Expected target column: {TARGET}")

# Remove missing target labels
train_df = train_df.dropna(subset=[TARGET]).copy()
val_df = val_df.dropna(subset=[TARGET]).copy()

# Separate labels and exclude identifiers
X_train = train_df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
y_train = train_df[TARGET].astype(str)

X_val = val_df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
y_val = val_df[TARGET].astype(str)

# Use the same feature columns in both splits
X_val = X_val.reindex(columns=X_train.columns)

# Identify numeric and categorical columns from training data
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [
    col for col in X_train.columns if col not in numeric_cols
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_f1 = -1
best_pipeline = None
best_name = None

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    classes = list(pipeline.named_steps["model"].classes_)
    positive_index = classes.index("leukemia")
    probabilities = pipeline.predict_proba(X_val)[:, positive_index]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, pos_label="leukemia", zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, pos_label="leukemia", zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, pos_label="leukemia", zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            (y_val == "leukemia").astype(int), probabilities
        )
    }

    results.append(metrics)

    print(pd.Series(metrics).to_string())
    print("\nClassification report:")
    print(classification_report(y_val, predictions, zero_division=0))

    if metrics["F1"] > best_f1:
        best_f1 = metrics["F1"]
        best_pipeline = pipeline
        best_name = name

results_df = pd.DataFrame(results)
results_df.to_csv(MODEL_DIR / "validation_results.csv", index=False)

joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib")

print("\nValidation comparison:")
print(results_df.to_string(index=False))
print(f"\nSaved best validation-F1 model: {best_name}")
print(f"Saved model to: {MODEL_DIR / 'best_model.joblib'}")
print("Test data has not been used.")
