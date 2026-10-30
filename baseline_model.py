
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Locate the dataset
BASE = Path(__file__).resolve().parent

DATA = (
    BASE
    / "data"
    / "raw"
    / "LeukoAlert-project"
    / "leuk vs HC"
    / "dataset"
)

TARGET = "leukemia_label"
ID_COLUMN = "sample_id"


# 2. Load CSV files from a split
def load_data(split_name):
    folder = DATA / split_name
    files = sorted(folder.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {folder}"
        )

    print(f"\nLoading {split_name} data...")
    print("Files:", len(files))

    frames = [pd.read_csv(file) for file in files]
    data = pd.concat(frames, ignore_index=True)

    print("Dataset shape:", data.shape)
    return data


# 3. Load training and validation data
train_df = load_data("training")
val_df = load_data("validation")

if TARGET not in train_df.columns or TARGET not in val_df.columns:
    raise ValueError(f"Missing target column: {TARGET}")

train_df = train_df.dropna(subset=[TARGET]).copy()
val_df = val_df.dropna(subset=[TARGET]).copy()

# Remove identifiers and labels from model inputs
X_train = train_df.drop(
    columns=[TARGET, ID_COLUMN], errors="ignore"
)
y_train = train_df[TARGET].astype(str)

X_val = val_df.drop(
    columns=[TARGET, ID_COLUMN], errors="ignore"
)
y_val = val_df[TARGET].astype(str)

# Match validation features to training features
X_val = X_val.reindex(columns=X_train.columns)

# 4. Identify numeric and categorical features
numeric_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = [
    column
    for column in X_train.columns
    if column not in numeric_features
]

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))
print("\nTraining labels:")
print(y_train.value_counts())

# 5. Prepare the data safely
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 6. Create the baseline model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )),
])

# 7. Train
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# 8. Evaluate on validation data
predictions = model.predict(X_val)
classes = list(model.named_steps["classifier"].classes_)

if "leukemia" not in classes:
    raise ValueError("The training data has no leukemia class.")

positive_index = classes.index("leukemia")
probabilities = model.predict_proba(X_val)[:, positive_index]

actual_binary = (y_val == "leukemia").astype(int)

print("\n===== BASELINE VALIDATION RESULTS =====")
print(f"Accuracy:  {accuracy_score(y_val, predictions):.4f}")
print(f"Precision: {precision_score(y_val, predictions, pos_label='leukemia', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val, predictions, pos_label='leukemia', zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_val, predictions, pos_label='leukemia', zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(actual_binary, probabilities):.4f}")

print("\nClassification report:")
print(classification_report(
    y_val, predictions, zero_division=0
))

print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(
    y_val, predictions, labels=["healthy", "leukemia"]
))

# 9. Save the baseline model and results
import joblib

model_dir = BASE / "models"
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / "baseline_logistic_regression.joblib")

results = pd.DataFrame([{
    "model": "Logistic Regression",
    "accuracy": accuracy_score(y_val, predictions),
    "precision": precision_score(
        y_val, predictions, pos_label="leukemia", zero_division=0
    ),
    "recall": recall_score(
        y_val, predictions, pos_label="leukemia", zero_division=0
    ),
    "f1_score": f1_score(
        y_val, predictions, pos_label="leukemia", zero_division=0
    ),
    "roc_auc": roc_auc_score(actual_binary, probabilities),
}])

results.to_csv(model_dir / "baseline_results.csv", index=False)

print("\nSaved baseline model and results in:", model_dir)
