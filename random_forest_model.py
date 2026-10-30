
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "raw" / "LeukoAlert-project" / "leuk vs HC" / "dataset"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "leukemia_label"
ID = "sample_id"


def load_split(name):
    files = sorted((DATA / name).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {DATA / name}")
    return pd.concat(
        [pd.read_csv(f) for f in files],
        ignore_index=True
    )


train = load_split("training").dropna(subset=[TARGET]).copy()
valid = load_split("validation").dropna(subset=[TARGET]).copy()

X_train = train.drop(columns=[TARGET, ID], errors="ignore")
y_train = train[TARGET].astype(str)

X_valid = valid.drop(columns=[TARGET, ID], errors="ignore")
y_valid = valid[TARGET].astype(str)

X_valid = X_valid.reindex(columns=X_train.columns)

numeric = X_train.select_dtypes(include=np.number).columns.tolist()
categorical = [c for c in X_train.columns if c not in numeric]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

print("Training Random Forest...")
model.fit(X_train, y_train)

pred = model.predict(X_valid)
classes = list(model.named_steps["classifier"].classes_)
prob = model.predict_proba(X_valid)[:, classes.index("leukemia")]
actual = (y_valid == "leukemia").astype(int)

results = {
    "model": "Random Forest",
    "accuracy": accuracy_score(y_valid, pred),
    "precision": precision_score(
        y_valid, pred, pos_label="leukemia", zero_division=0
    ),
    "recall": recall_score(
        y_valid, pred, pos_label="leukemia", zero_division=0
    ),
    "f1_score": f1_score(
        y_valid, pred, pos_label="leukemia", zero_division=0
    ),
    "roc_auc": roc_auc_score(actual, prob),
}

print("\nRandom Forest validation results:")
for key, value in results.items():
    if isinstance(value, (int, float, np.number)):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\nConfusion matrix (actual rows, predicted columns):")
print(confusion_matrix(
    y_valid, pred, labels=["healthy", "leukemia"]
))

joblib.dump(model, MODEL_DIR / "random_forest_model.joblib")
pd.DataFrame([results]).to_csv(
    MODEL_DIR / "random_forest_results.csv", index=False
)

print("\nModel and results saved in:", MODEL_DIR)
