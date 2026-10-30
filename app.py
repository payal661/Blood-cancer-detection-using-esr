from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import re
import zipfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BloodCancerESR",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS / Design System ───────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1828 0%, #0f2035 100%) !important;
        border-right: 1px solid #1e3a52;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.9rem !important;
        color: #8fa8c0 !important;
        padding: 6px 10px !important;
        border-radius: 8px;
        transition: all .2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #38c9b5 !important;
        background: rgba(56,201,181,.08) !important;
    }
    /* Active nav item */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked + div {
        border-color: #38c9b5 !important;
    }

    /* ── Main container ── */
    .main .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }

    /* ── Stat card ── */
    .stat-card {
        background: #122136;
        border: 1px solid #1e3a52;
        border-radius: 14px;
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: #38c9b5;
        border-radius: 3px 0 0 3px;
    }
    .stat-card .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #8fa8c0;
        margin-bottom: 6px;
    }
    .stat-card .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #eaf2f8;
        line-height: 1.1;
    }
    .stat-card .stat-sub {
        font-size: 0.78rem;
        color: #38c9b5;
        margin-top: 4px;
        font-weight: 500;
    }

    /* ── Method card ── */
    .method-card {
        background: #122136;
        border: 1px solid #1e3a52;
        border-radius: 14px;
        padding: 22px 24px;
        height: 100%;
        transition: border-color .25s, box-shadow .25s;
        cursor: default;
    }
    .method-card:hover {
        border-color: #38c9b5;
        box-shadow: 0 0 0 1px #38c9b566, 0 4px 24px rgba(56,201,181,.12);
    }
    .method-card .mc-icon {
        font-size: 1.6rem;
        margin-bottom: 10px;
        display: block;
    }
    .method-card .mc-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #eaf2f8;
        margin-bottom: 6px;
    }
    .method-card .mc-desc {
        font-size: 0.82rem;
        color: #8fa8c0;
        line-height: 1.5;
    }

    /* ── Page header strip ── */
    .page-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .page-header-icon {
        width: 44px; height: 44px;
        background: rgba(56,201,181,.15);
        border: 1px solid rgba(56,201,181,.3);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
    }
    .page-header-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #eaf2f8;
        margin: 0;
    }
    .page-header-sub {
        font-size: 0.82rem;
        color: #8fa8c0;
        margin: 2px 0 0;
    }

    /* ── Section divider ── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, #1e3a52 0%, transparent 100%);
        margin: 24px 0;
    }

    /* ── Prediction result badge ── */
    .result-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: .03em;
    }
    .result-badge.leukemia {
        background: rgba(239,68,68,.15);
        color: #f87171;
        border: 1px solid rgba(239,68,68,.3);
    }
    .result-badge.healthy {
        background: rgba(56,201,181,.15);
        color: #38c9b5;
        border: 1px solid rgba(56,201,181,.3);
    }

    /* ── Research notice ── */
    .research-notice {
        background: rgba(245,158,11,.07);
        border: 1px solid rgba(245,158,11,.25);
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 0.8rem;
        color: #fcd34d;
        margin-bottom: 20px;
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }

    /* ── Preview table ── */
    .preview-panel {
        background: #122136;
        border: 1px solid #1e3a52;
        border-radius: 14px;
        padding: 20px;
    }
    .preview-panel .pp-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #38c9b5;
        margin-bottom: 4px;
    }
    .preview-panel .pp-sub {
        font-size: 0.75rem;
        color: #8fa8c0;
        margin-bottom: 14px;
    }

    /* ── Workflow tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        font-weight: 600;
        color: #8fa8c0;
        border-radius: 10px 10px 0 0;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #38c9b5 !important;
        border-bottom-color: #38c9b5 !important;
    }

    /* ── Metric tiles ── */
    [data-testid="stMetric"] {
        background: #122136;
        border: 1px solid #1e3a52;
        border-radius: 12px;
        padding: 16px 18px !important;
    }
    [data-testid="stMetricLabel"] { color: #8fa8c0 !important; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { color: #eaf2f8 !important; }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #38c9b5, #2ba89a) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: .02em !important;
        color: #0c1828 !important;
        box-shadow: 0 2px 14px rgba(56,201,181,.35) !important;
        transition: opacity .2s !important;
    }
    .stButton > button[kind="primary"]:hover { opacity: .88 !important; }

    /* ── Download button ── */
    .stDownloadButton > button {
        border-color: #1e3a52 !important;
        color: #38c9b5 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        background: transparent !important;
    }

    /* ── Form inputs ── */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] > div > div {
        background: #0e1d30 !important;
        border-color: #1e3a52 !important;
        color: #eaf2f8 !important;
        border-radius: 8px !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #0e1d30 !important;
        border: 2px dashed #1e3a52 !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #38c9b5 !important;
    }

    /* ── Sidebar logo area ── */
    .sidebar-logo {
        display: flex; align-items: center; gap: 12px;
        padding: 8px 0 20px;
        border-bottom: 1px solid #1e3a52;
        margin-bottom: 20px;
    }
    .sidebar-logo-icon {
        width: 40px; height: 40px;
        background: rgba(56,201,181,.15);
        border: 1px solid rgba(56,201,181,.3);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
    }
    .sidebar-logo-text .name { font-size: 1rem; font-weight: 700; color: #eaf2f8; }
    .sidebar-logo-text .sub  { font-size: 0.72rem; color: #8fa8c0; }

    /* ── Nav badge ── */
    .nav-badge {
        display: inline-block;
        background: rgba(56,201,181,.15);
        color: #38c9b5;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 999px;
        border: 1px solid rgba(56,201,181,.25);
        vertical-align: middle;
        margin-left: 6px;
    }

    /* ── Overview page hero ── */
    .overview-hero {
        background: linear-gradient(135deg, #122136 0%, #0c2035 100%);
        border: 1px solid #1e3a52;
        border-radius: 18px;
        padding: 32px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .overview-hero::after {
        content: '🩸';
        position: absolute;
        right: 36px; top: 24px;
        font-size: 5rem;
        opacity: .12;
    }

    /* ── Missing feature chip ── */
    .missing-chip {
        display: inline-block;
        background: rgba(239,68,68,.12);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,.25);
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 10px;
        margin: 3px 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = (
    ROOT / "data" / "raw" / "LeukoAlert-project" / "leuk vs HC" / "dataset"
)

MANUAL_MODEL_PATH  = MODEL_DIR / "manual_cbc_model.joblib"
FULL_MODEL_PATH    = MODEL_DIR / "random_forest_model.joblib"
TEST_DATA_PATH     = DATA_DIR / "test" / "test_site_true_world.csv"
VALIDATION_RESULTS_PATH = MODEL_DIR / "validation_results.csv"
XGBOOST_RESULTS_PATH    = MODEL_DIR / "xgboost_results.csv"
FINAL_RESULTS_PATH      = MODEL_DIR / "final_test_results.csv"

TARGET      = "leukemia_label"
ID_COLUMN   = "sample_id"
MANUAL_THRESHOLD     = 0.5
FULL_MODEL_THRESHOLD = 0.8
MAX_RECORD_MATCHES   = 50
MAX_UPLOAD_ROWS      = 1_000

MANUAL_FIELDS = [
    ("WBC(10^9/L)",  "WBC",  "White Blood Cells",                         7.0),
    ("RBC(10^12/L)", "RBC",  "Red Blood Cells",                           4.0),
    ("HGB(g/L)",     "HGB",  "Hemoglobin",                              120.0),
    ("HCT(%)",       "HCT",  "Hematocrit",                               36.0),
    ("MCV(fL)",      "MCV",  "Mean Corpuscular Volume",                  90.0),
    ("MCH(pg)",      "MCH",  "Mean Corpuscular Hemoglobin",              30.0),
    ("MCHC(g/L)",    "MCHC", "Mean Corpuscular Hemoglobin Concentration",330.0),
    ("PLT(10^9/L)",  "PLT",  "Platelet Count",                          200.0),
]
MANUAL_FEATURES = [f for f, _, _, _ in MANUAL_FIELDS]

MANUAL_ALIASES = {
    "WBC(10^9/L)":  ("wbc", "white blood cell count"),
    "RBC(10^12/L)": ("rbc", "red blood cell count"),
    "HGB(g/L)":     ("hgb", "hemoglobin", "haemoglobin"),
    "HCT(%)":       ("hct", "hematocrit", "haematocrit"),
    "MCV(fL)":      ("mcv", "mean corpuscular volume"),
    "MCH(pg)":      ("mch", "mean corpuscular hemoglobin"),
    "MCHC(g/L)":    ("mchc", "mean corpuscular hemoglobin concentration"),
    "PLT(10^9/L)":  ("plt", "platelet count", "platelets"),
}
LABEL_ALIASES = {"healthy": 0, "hc": 0, "leukemia": 1, "leukaemia": 1, "leuk": 1}

FIELD_UNITS = {
    "WBC(10^9/L)":  "10⁹/L",
    "RBC(10^12/L)": "10¹²/L",
    "HGB(g/L)":     "g/L",
    "HCT(%)":       "%",
    "MCV(fL)":      "fL",
    "MCH(pg)":      "pg",
    "MCHC(g/L)":    "g/L",
    "PLT(10^9/L)":  "10⁹/L",
}


# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_manual_model(path: str) -> tuple[object, list[str]]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Manual model file must be a dict with a 'model' key.")
    model    = artifact["model"]
    features = artifact.get("features")
    if features != MANUAL_FEATURES:
        raise ValueError("Manual model feature list mismatch.")
    validate_prediction_model(model)
    return model, list(features)


@st.cache_resource
def load_full_model(path: str) -> tuple[object, list[str]]:
    model    = joblib.load(path)
    features = list(getattr(model, "feature_names_in_", []))
    if not features:
        raise ValueError("Full model does not expose feature_names_in_.")
    if not set(MANUAL_FEATURES).issubset(features):
        raise ValueError("Full model missing expected CBC fields.")
    validate_prediction_model(model)
    return model, features


def validate_prediction_model(model: object) -> None:
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("Model must support predict_proba and expose classes_.")
    if positive_class_index(model) is None:
        raise ValueError("Model does not contain a leukemia class.")


def positive_class_index(model: object) -> int | None:
    classes = list(model.classes_)
    for label in ("leukemia", "leukaemia", "leuk", 1):
        if label in classes:
            return classes.index(label)
    return None


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_test_records(path: str, features: tuple[str, ...]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=[ID_COLUMN, *features])


@st.cache_data(show_spinner=False)
def load_uploaded_data(contents: bytes, suffix: str) -> pd.DataFrame:
    source = BytesIO(contents)
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix == ".xlsx":
        frame = pd.read_excel(source, engine="openpyxl")
    else:
        raise ValueError("Upload a CSV (.csv) or Excel (.xlsx) file.")
    if frame.empty:
        raise ValueError("The uploaded file contains no data rows.")
    frame.columns = [str(c).strip() for c in frame.columns]
    dupes = frame.columns[frame.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate column names after trimming: {dupes}")
    return frame


@st.cache_data(show_spinner=False)
def load_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_dataset_summary(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root      = Path(data_dir)
    split_rows: list[dict] = []
    label_rows: list[dict] = []
    for split in ("training", "validation", "test"):
        files  = sorted((root / split).glob("*.csv"))
        counts: dict[str, int] = {}
        total  = 0
        for f in files:
            labels = pd.read_csv(f, usecols=[TARGET])[TARGET]
            total += len(labels)
            for lbl, cnt in labels.fillna("Missing").value_counts().items():
                counts[str(lbl)] = counts.get(str(lbl), 0) + int(cnt)
        split_rows.append({"Dataset split": split.title(), "Files": len(files), "Records": total})
        for lbl, cnt in counts.items():
            label_rows.append({"Dataset split": split.title(), "Dataset label": lbl, "Records": cnt})
    return pd.DataFrame(split_rows), pd.DataFrame(label_rows)


@st.cache_data(show_spinner=False)
def load_training_feature_stats(data_dir: str) -> pd.DataFrame:
    training_dir = Path(data_dir) / "training"
    files = sorted(training_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No training CSVs in {training_dir}")
    frames = [pd.read_csv(f, usecols=MANUAL_FEATURES) for f in files]
    features = pd.concat(frames, ignore_index=True)
    numeric  = features.apply(pd.to_numeric, errors="coerce")
    stats    = numeric.describe().T
    return stats.loc[:, ["count", "mean", "std", "min", "50%", "max"]].rename(
        columns={"count": "Count", "mean": "Mean", "std": "Std. dev.",
                 "min": "Minimum", "50%": "Median", "max": "Maximum"}
    )


@st.cache_data(show_spinner=False)
def load_explorer_records(path: str) -> pd.DataFrame:
    return pd.read_csv(path, usecols=[ID_COLUMN, TARGET, *MANUAL_FEATURES])


@st.cache_data(show_spinner=False)
def calculate_test_evaluation(
    model_path: str,
    test_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model, features = load_full_model(model_path)
    test_data = pd.read_csv(test_path, usecols=[*features, TARGET])
    labels    = (
        test_data[TARGET].astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    )
    if labels.isna().any():
        unknown = test_data.loc[labels.isna(), TARGET].unique().tolist()
        raise ValueError(f"Unrecognized labels: {unknown}")

    pos_idx      = positive_class_index(model)
    probabilities = model.predict_proba(test_data.loc[:, features])[:, pos_idx]
    actual       = labels.to_numpy(dtype=int)
    predictions  = (probabilities >= FULL_MODEL_THRESHOLD).astype(int)

    metrics = pd.DataFrame([{
        "Dataset split":   "Held-out test",
        "Model":           "Full-feature Random Forest",
        "Threshold":       FULL_MODEL_THRESHOLD,
        "Accuracy":        accuracy_score(actual, predictions),
        "Precision":       precision_score(actual, predictions, zero_division=0),
        "Recall":          recall_score(actual, predictions, zero_division=0),
        "F1-score":        f1_score(actual, predictions, zero_division=0),
        "ROC-AUC":         roc_auc_score(actual, probabilities),
    }])

    matrix    = confusion_matrix(actual, predictions, labels=[0, 1])
    confusion = pd.DataFrame(
        matrix,
        index=["Actual healthy", "Actual leukemia"],
        columns=["Predicted healthy", "Predicted leukemia"],
    )
    fpr, tpr, _ = roc_curve(actual, probabilities)
    fpr_grid    = np.linspace(0.0, 1.0, 101)
    roc_data    = pd.DataFrame({
        "False positive rate":        fpr_grid,
        "Random baseline":            fpr_grid,
        "Full-feature Random Forest": np.interp(fpr_grid, fpr, tpr),
    })

    threshold_rows = []
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        preds = (probabilities >= t).astype(int)
        cm    = confusion_matrix(actual, preds, labels=[0, 1])
        threshold_rows.append({
            "Threshold":       t,
            "Accuracy":        accuracy_score(actual, preds),
            "Precision":       precision_score(actual, preds, zero_division=0),
            "Recall":          recall_score(actual, preds, zero_division=0),
            "F1-score":        f1_score(actual, preds, zero_division=0),
            "False positives": int(cm[0, 1]),
            "False negatives": int(cm[1, 0]),
        })
    return metrics, confusion, roc_data, pd.DataFrame(threshold_rows)


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def guess_manual_column(columns: list[str], feature: str) -> str | None:
    normalized = {normalize_header(c): c for c in columns}
    aliases    = (feature, *MANUAL_ALIASES.get(feature, ()))
    for alias in aliases:
        match = normalized.get(normalize_header(alias))
        if match is not None:
            return match
    return None


def build_manual_input_frame(source: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {feat: pd.to_numeric(source[col], errors="coerce") for feat, col in mapping.items()},
        index=source.index,
    ).loc[:, MANUAL_FEATURES]


def predict_rows(
    model,
    model_features: list[str],
    input_values: pd.DataFrame,
    threshold: float,
    require_complete_numeric: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    missing = [f for f in model_features if f not in input_values.columns]
    if missing:
        raise ValueError(f"Missing required model fields: {missing}")
    features = input_values.loc[:, model_features]
    numeric  = features.apply(pd.to_numeric, errors="coerce")
    if require_complete_numeric and numeric.isna().any().any():
        bad = (np.flatnonzero(numeric.isna().any(axis=1).to_numpy()) + 1).tolist()
        raise ValueError(
            f"Missing/non-numeric values in row(s): {bad[:10]}"
            + (" ..." if len(bad) > 10 else "")
        )
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("CBC input values cannot be infinite.")
    pos_idx       = positive_class_index(model)
    probabilities = model.predict_proba(numeric)[:, pos_idx]
    predictions   = (probabilities >= threshold).astype(int)
    return probabilities, predictions


# ── Shared UI helpers ─────────────────────────────────────────────────────────
def page_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-icon">{icon}</div>
            <div>
                <div class="page-header-title">{title}</div>
                <div class="page-header-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def research_notice() -> None:
    st.markdown(
        """
        <div class="research-notice">
            <span>⚠️</span>
            <span>
                <strong>Educational research prototype only.</strong>
                CBC-based model — not ESR-based. Model predictions are
                <strong>not medical diagnoses</strong> and must not guide
                patient care or clinical decisions.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {"<div class='stat-sub'>" + sub + "</div>" if sub else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_prediction_results(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    threshold: float,
    record_ids: list[str] | None = None,
) -> None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if len(predictions) == 1:
        prob  = float(probabilities[0])
        label = "leukemia" if predictions[0] == 1 else "healthy"
        badge_cls = "leukemia" if label == "leukemia" else "healthy"
        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            st.metric("Model Score", f"{prob:.1%}")
        with col_b:
            st.markdown(
                f"<div style='margin-top:24px'>"
                f"<span class='result-badge {badge_cls}'>"
                f"{'🔴 Leukemia' if label == 'leukemia' else '🟢 Healthy'}"
                f"</span></div>",
                unsafe_allow_html=True,
            )
        with col_c:
            st.progress(min(prob, 1.0))
            st.caption(
                f"Threshold: {threshold:.0%} · Model: {model_name}. "
                "Score is not a clinical probability."
            )
    else:
        st.caption(f"Model: **{model_name}** · Threshold: {threshold:.0%}")

    output = pd.DataFrame({
        "Record": (
            record_ids if record_ids is not None
            else [f"Row {i}" for i in range(1, len(predictions) + 1)]
        ),
        "Predicted label": [
            "leukemia" if p == 1 else "healthy" for p in predictions
        ],
        "Leukemia-class score": probabilities,
        "Threshold": threshold,
    })
    st.dataframe(
        output,
        hide_index=True,
        column_config={
            "Leukemia-class score": st.column_config.ProgressColumn(
                "Leukemia-class score",
                format="percent",
                min_value=0.0,
                max_value=1.0,
            )
        },
        use_container_width=True,
    )
    st.download_button(
        "⬇ Download results as CSV",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="bloodcanceresr_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "⚠️ Experimental predictions of dataset labels — not diagnoses. "
        "Must not be used to guide patient care."
    )


# ── Page: Overview ────────────────────────────────────────────────────────────
def show_overview() -> None:
    page_header("⚡", "Overview", "BloodCancerESR · CBC Analysis & Research Dashboard")

    st.markdown(
        """
        <div class="overview-hero">
            <div style="font-size:.75rem;font-weight:700;letter-spacing:.1em;
                        text-transform:uppercase;color:#38c9b5;margin-bottom:8px;">
                RESEARCH PROTOTYPE
            </div>
            <div style="font-size:1.45rem;font-weight:700;color:#eaf2f8;
                        max-width:520px;line-height:1.35;margin-bottom:12px;">
                Machine-learning leukemia classification<br>using Complete Blood Count data
            </div>
            <div style="font-size:.85rem;color:#8fa8c0;max-width:460px;line-height:1.6;">
                BloodCancerESR trains and evaluates Random Forest, Logistic Regression, and
                XGBoost models on CBC features to explore automated classification of healthy
                versus leukemia dataset records.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    research_notice()

    # ── Quick-stat cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Prediction methods", "02", "Manual + CSV upload")
    with c2:
        stat_card("Input validation", "On", "Feature checks before prediction")
    with c3:
        stat_card("CBC features (manual)", "08", "WBC · RBC · HGB · HCT · MCV · MCH · MCHC · PLT")
    with c4:
        stat_card("Model format", ".joblib", "scikit-learn compatible")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Sidebar page guide ────────────────────────────────────────────────────
    st.markdown("#### Pages included in this dashboard")
    pages = [
        ("🔮", "Prediction Studio",
         "Manual CBC entry form and CSV upload with data preview, "
         "column validation, and downloadable results."),
        ("📊", "Model Performance",
         "Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, "
         "and threshold comparison across all saved models."),
        ("🗄️", "Dataset Explorer",
         "Dataset split record counts, healthy/leukemia label distribution, "
         "CBC feature statistics, and a searchable test-record browser."),
        ("💡", "About the Project",
         "Problem statement, methodology, model details, limitations, "
         "and research disclaimer."),
    ]
    for icon, title, desc in pages:
        with st.container(border=True):
            st.markdown(
                f"**{icon} {title}**  \n"
                f"<span style='font-size:.82rem;color:#8fa8c0'>{desc}</span>",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.75rem;color:#8fa8c0;text-align:center;padding:8px 0'>"
        "BloodCancerESR · Educational CBC-based research prototype · "
        "Not ESR-based · Not for clinical diagnosis"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Page: Prediction Studio ───────────────────────────────────────────────────
def show_prediction_studio() -> None:
    page_header("🔮", "Prediction Studio", "Choose how you want to provide CBC data")
    research_notice()

    # ── Method selector cards (visual) ────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="method-card">
                <span class="mc-icon">📋</span>
                <div class="mc-title">Manual CBC Entry</div>
                <div class="mc-desc">
                    Enter eight blood-test parameters directly using the form below.
                    Uses <code>manual_cbc_model.joblib</code> — expects exactly 8 features.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="method-card">
                <span class="mc-icon">📂</span>
                <div class="mc-title">Upload Patient Record</div>
                <div class="mc-desc">
                    Upload a CSV or Excel file with CBC data.
                    The app validates column names, shows a preview,
                    and lets you correct values before prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Workflow tabs ─────────────────────────────────────────────────────────
    tab_manual, tab_upload, tab_saved = st.tabs([
        "  📋  Manual CBC Entry  ",
        "  📂  Upload CSV / Excel  ",
        "  🗂️  Saved Dataset Record  ",
    ])

    with tab_manual:
        _show_manual_entry()

    with tab_upload:
        _show_file_upload()

    with tab_saved:
        _show_saved_dataset_prediction()


def _show_manual_entry() -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "Enter the eight CBC values below and click **Run Prediction**. "
        "Pre-filled numbers are example defaults — replace them with actual values."
    )

    if not MANUAL_MODEL_PATH.exists():
        st.error(
            f"Manual CBC model not found at `{MANUAL_MODEL_PATH}`. "
            "Run `train_manual_model.py` to create it."
        )
        return
    try:
        model, features = load_manual_model(str(MANUAL_MODEL_PATH))
    except (OSError, ValueError, KeyError) as e:
        st.error(f"Manual CBC model could not be loaded: {e}")
        return

    if st.button("↩ Restore example defaults", key="restore_manual"):
        for feat, _, _, default in MANUAL_FIELDS:
            st.session_state[f"manual_{feat}"] = default

    with st.form("manual_cbc_form"):
        cols_a = st.columns(4)
        cols_b = st.columns(4)
        col_groups = [cols_a, cols_b]
        entered: dict[str, float] = {}

        for idx, (feat, short, long_name, default) in enumerate(MANUAL_FIELDS):
            row    = idx // 4
            col_i  = idx % 4
            unit   = FIELD_UNITS.get(feat, "")
            with col_groups[row][col_i]:
                entered[feat] = st.number_input(
                    f"{short}",
                    min_value=0.0,
                    max_value=1_000_000.0,
                    value=default,
                    step=0.1,
                    format="%.2f",
                    help=f"{long_name} ({unit}) · model key: {feat}",
                    key=f"manual_{feat}",
                )
                st.caption(unit)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔬 Run Prediction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        values = pd.DataFrame([entered], columns=features)
        try:
            probs, preds = predict_rows(model, features, values, MANUAL_THRESHOLD)
        except (TypeError, ValueError) as e:
            st.error(f"Prediction failed: {e}")
            return
        show_prediction_results(probs, preds, "Manual CBC Random Forest", MANUAL_THRESHOLD)
        with st.expander("🔍 Review the eight entered values"):
            st.dataframe(
                values.T.rename(columns={values.index[0]: "Entered value"}),
                use_container_width=True,
            )


def _show_file_upload() -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "Upload a **CSV or Excel** file with one patient per row. "
        "The app inspects headers, lets you confirm column mapping, "
        "and runs the best compatible model automatically."
    )

    # ── Upload info panel ─────────────────────────────────────────────────────
    with st.expander("ℹ️ How CSV upload works", expanded=False):
        st.markdown(
            """
            | Upload / input | Model | Will it work? |
            |---|---|---|
            | Manual entry of 8 CBC values | `manual_cbc_model.joblib` | ✅ Yes — expects 8 features |
            | CSV with all required full-model columns | Full-feature Random Forest | ✅ Yes — if column names match |
            | CSV with some columns missing | Full-feature Random Forest | ❌ App shows which columns are missing |

            **The app never assumes** any fixed number of columns is sufficient.
            It reads the model's own `feature_names_in_` and compares them to your file.
            """
        )

    uploaded = st.file_uploader(
        "Choose a CBC data file (.csv or .xlsx, max 10 MB, max 1 000 rows)",
        type=["csv", "xlsx"],
        help="One patient per row. Column names should match CBC feature names.",
    )

    if uploaded is None:
        # Sample layout preview
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="preview-panel">
                <div class="pp-title">📊 Uploaded file preview</div>
                <div class="pp-sub">Example layout — actual columns depend on the uploaded file.</div>
            """,
            unsafe_allow_html=True,
        )
        sample = pd.DataFrame({
            "WBC(10^9/L)":  [7.2, 18.5],
            "RBC(10^12/L)": [4.6,  3.4],
            "HGB(g/L)":     [135,   92],
            "HCT(%)":       [40.1,  27.3],
            "MCV(fL)":      [87.2,  79.5],
            "MCH(pg)":      [29.3,  26.1],
            "MCHC(g/L)":    [336,  328],
            "PLT(10^9/L)":  [250,  110],
        })
        st.dataframe(sample, hide_index=True, use_container_width=True)
        st.markdown(
            """
            <div style="font-size:.78rem;color:#38c9b5;margin-top:10px;">
                ✅ Columns are validated against the model's expected feature list before prediction.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Parse file ────────────────────────────────────────────────────────────
    contents = uploaded.getvalue()
    suffix   = Path(uploaded.name).suffix.lower()
    try:
        source = load_uploaded_data(contents, suffix)
    except (ImportError, OSError, ValueError, UnicodeDecodeError,
            zipfile.BadZipFile, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        st.error(f"File could not be read: {e}")
        return

    if len(source) > MAX_UPLOAD_ROWS:
        st.error(
            f"File has {len(source):,} rows. "
            f"This demo accepts at most {MAX_UPLOAD_ROWS:,} rows at a time."
        )
        return

    st.success(
        f"✅ File read: **{len(source):,} record(s)** · **{len(source.columns):,} column(s)**"
    )

    # ── Determine model ───────────────────────────────────────────────────────
    full_model:    object | None    = None
    full_features: list[str]        = []
    full_ok                         = False

    if FULL_MODEL_PATH.exists():
        try:
            full_model, full_features = load_full_model(str(FULL_MODEL_PATH))
            missing_for_full = [f for f in full_features if f not in source.columns]
            full_ok          = len(missing_for_full) == 0
            if not full_ok:
                st.warning(
                    f"**Full-feature model** requires {len(full_features)} columns. "
                    f"**{len(missing_for_full)} are missing** from your file."
                )
                st.markdown(
                    "Missing columns: " + " ".join(
                        f'<span class="missing-chip">{c}</span>'
                        for c in missing_for_full[:30]
                    ) + ("  ..." if len(missing_for_full) > 30 else ""),
                    unsafe_allow_html=True,
                )
        except (OSError, ValueError, KeyError) as e:
            st.warning(f"Full-feature model unavailable: {e}")

    if full_ok:
        route = st.selectbox(
            "Compatible model detected — choose which one to use:",
            [
                f"Full-feature Random Forest ({len(full_features)} fields)",
                "Manual CBC model (8 fields)",
            ],
        )
        use_full = route.startswith("Full-feature")
    else:
        use_full = False
        st.info(
            "File does not contain all full-model columns. "
            "Falling back to the **manual CBC model** (8 fields)."
        )

    # ── Prepare input values ──────────────────────────────────────────────────
    if use_full and full_model is not None:
        model              = full_model
        model_features     = full_features
        model_name         = "Full-feature Random Forest"
        threshold          = FULL_MODEL_THRESHOLD
        input_values       = source.loc[:, model_features].apply(
            pd.to_numeric, errors="coerce"
        )
        require_complete   = False
        if input_values.isna().any().any():
            st.warning("Some values are blank or non-numeric — shown as empty cells.")
    else:
        if not MANUAL_MODEL_PATH.exists():
            st.error(
                f"Manual CBC model not found: `{MANUAL_MODEL_PATH}`. "
                "Run `train_manual_model.py`."
            )
            return
        try:
            model, model_features = load_manual_model(str(MANUAL_MODEL_PATH))
        except (OSError, ValueError, KeyError) as e:
            st.error(f"Manual CBC model could not be loaded: {e}")
            return

        model_name       = "Manual CBC Random Forest"
        threshold        = MANUAL_THRESHOLD
        require_complete = True

        # Column mapping
        columns  = source.columns.tolist()
        digest   = hashlib.sha256(contents).hexdigest()[:12]
        mapping: dict[str, str] = {}

        with st.expander("🗂️ Confirm or map CBC columns", expanded=True):
            st.caption(
                "The app auto-matched column names. "
                "Correct any mismatches using the dropdowns."
            )
            map_cols = st.columns(2)
            for idx, (feat, short, long_name, _) in enumerate(MANUAL_FIELDS):
                guessed  = guess_manual_column(columns, feat)
                options  = ["— Select a column —", *columns]
                sel_idx  = options.index(guessed) if guessed in options else 0
                with map_cols[idx % 2]:
                    selected = st.selectbox(
                        f"{short} — {long_name}",
                        options,
                        index=sel_idx,
                        key=f"upload_map_{digest}_{feat}",
                    )
                if selected != "— Select a column —":
                    mapping[feat] = selected

        if len(mapping) != len(MANUAL_FEATURES):
            st.warning("Map a source column to each of the 8 CBC fields before running.")
            return
        if len(set(mapping.values())) != len(mapping):
            st.error("Each CBC field must map to a different source column.")
            return

        input_values = build_manual_input_frame(source, mapping)
        if input_values.isna().any().any():
            st.warning("Blank or non-numeric values appear as empty cells — correct before prediction.")

    # ── Data preview + edit ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="preview-panel">
            <div class="pp-title">📊 Uploaded file preview</div>
            <div class="pp-sub">
                {model_name} · {len(model_features)} required fields ·
                {len(input_values):,} record(s). Edit cells to correct values.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    dig2 = hashlib.sha256(contents).hexdigest()[:12]
    edited = st.data_editor(
        input_values,
        key=f"upload_preview_{dig2}_{'full' if use_full else 'manual'}",
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        height=min(420, 38 + 35 * min(len(input_values), 10)),
    )

    if st.button(
        "✅ Confirm values and run prediction",
        type="primary",
        use_container_width=True,
    ):
        try:
            probs, preds = predict_rows(
                model, model_features, edited, threshold,
                require_complete_numeric=require_complete,
            )
        except (TypeError, ValueError) as e:
            st.error(f"Prediction failed: {e}")
            return
        show_prediction_results(probs, preds, model_name, threshold)


def _show_saved_dataset_prediction() -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "Search the held-out test dataset by record ID. "
        "The full-feature Random Forest uses all 74 model fields. "
        "The record's true label is withheld."
    )

    if not FULL_MODEL_PATH.exists():
        st.error(f"Full-feature model not found: `{FULL_MODEL_PATH}`")
        return
    if not TEST_DATA_PATH.exists():
        st.error(f"Test dataset not found: `{TEST_DATA_PATH}`")
        return
    try:
        model, features = load_full_model(str(FULL_MODEL_PATH))
        records         = load_test_records(str(TEST_DATA_PATH), tuple(features))
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as e:
        st.error(f"Records could not be loaded: {e}")
        return

    search_term = st.text_input(
        "🔍 Search saved record ID",
        placeholder="e.g. patient_388183  (at least 2 characters)",
    ).strip()

    if len(search_term) < 2:
        st.info("Enter at least 2 characters to find a record.")
        return

    matches = records.loc[
        records[ID_COLUMN].astype(str).str.contains(
            search_term, case=False, regex=False, na=False
        )
    ]
    if matches.empty:
        st.warning("No records match that ID. Try a different search term.")
        return
    if len(matches) > MAX_RECORD_MATCHES:
        st.info("More than 50 results — please refine the search.")
    matches     = matches.head(MAX_RECORD_MATCHES)
    selected_id = st.selectbox(
        "Select a record",
        options=matches[ID_COLUMN].astype(str).tolist(),
    )
    st.caption(f"{len(matches):,} matching record(s) shown.")

    selected = matches.loc[
        matches[ID_COLUMN].astype(str) == selected_id, features
    ]
    if st.button("🔬 Run prediction for selected record", type="primary", use_container_width=True):
        try:
            probs, preds = predict_rows(
                model, features, selected, FULL_MODEL_THRESHOLD,
                require_complete_numeric=False,
            )
        except (TypeError, ValueError) as e:
            st.error(f"Prediction failed: {e}")
            return
        st.caption(f"Record: **{selected_id}**")
        show_prediction_results(
            probs, preds, "Full-feature Random Forest", FULL_MODEL_THRESHOLD
        )
        with st.expander("🔍 CBC values for selected record"):
            st.dataframe(
                selected.loc[:, MANUAL_FEATURES].T.rename(
                    columns={selected.index[0]: "Value"}
                ),
                use_container_width=True,
            )


# ── Page: Model Performance ───────────────────────────────────────────────────
def show_model_performance() -> None:
    page_header("📊", "Model Performance", "Validation and held-out test evaluation results")
    research_notice()

    # ── Validation comparison ─────────────────────────────────────────────────
    if VALIDATION_RESULTS_PATH.exists() and XGBOOST_RESULTS_PATH.exists():
        try:
            val = load_results(str(VALIDATION_RESULTS_PATH)).rename(
                columns={"F1": "F1-score", "ROC_AUC": "ROC-AUC"}
            )
            xgb = load_results(str(XGBOOST_RESULTS_PATH)).rename(
                columns={"model": "Model", "accuracy": "Accuracy",
                         "precision": "Precision", "recall": "Recall",
                         "f1_score": "F1-score", "roc_auc": "ROC-AUC"}
            )
            comparison = pd.concat([val, xgb], ignore_index=True)
        except (OSError, ValueError, pd.errors.ParserError) as e:
            st.error(f"Validation results could not be read: {e}")
            comparison = pd.DataFrame()

        if not comparison.empty:
            st.subheader("Validation model comparison")
            st.caption(
                "Validation split · saved results · default thresholds. "
                "These are NOT held-out test results."
            )
            chart_metrics = [
                m for m in ("Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC")
                if m in comparison.columns
            ]
            if chart_metrics:
                st.bar_chart(
                    comparison.set_index("Model")[chart_metrics],
                    use_container_width=True,
                )
            st.dataframe(
                comparison.style.format({
                    "Accuracy": "{:.2%}", "Precision": "{:.2%}",
                    "Recall": "{:.2%}", "F1-score": "{:.4f}",
                    "ROC-AUC": "{:.2%}",
                }),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.warning(
            "Saved validation results unavailable. "
            "Check `validation_results.csv` and `xgboost_results.csv` in the models folder."
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Held-out test evaluation ──────────────────────────────────────────────
    st.subheader("Held-out test evaluation")
    st.caption(
        "Full-feature Random Forest · untouched test split · "
        f"{FULL_MODEL_THRESHOLD:.0%} decision threshold. "
        "ROC-AUC is threshold-independent."
    )

    if not FULL_MODEL_PATH.exists() or not TEST_DATA_PATH.exists():
        st.error("Saved full-feature model or test dataset missing — metrics unavailable.")
        return

    try:
        metrics, confusion, roc_data, threshold_data = calculate_test_evaluation(
            str(FULL_MODEL_PATH), str(TEST_DATA_PATH)
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as e:
        st.error(f"Held-out test metrics could not be calculated: {e}")
        return

    result = metrics.iloc[0]
    cols   = st.columns(5)
    cols[0].metric("Accuracy",  f"{result['Accuracy']:.2%}")
    cols[1].metric("Precision", f"{result['Precision']:.2%}")
    cols[2].metric("Recall",    f"{result['Recall']:.2%}")
    cols[3].metric("F1-score",  f"{result['F1-score']:.4f}")
    cols[4].metric("ROC-AUC",   f"{result['ROC-AUC']:.2%}")
    st.caption(f"Evaluated **{int(confusion.to_numpy().sum()):,}** held-out test records.")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown("#### Confusion matrix")
        st.caption("Rows: actual labels · Columns: predicted labels.")
        st.dataframe(confusion, use_container_width=True)
    with col_roc:
        st.markdown("#### ROC curve")
        st.caption("Sensitivity vs. false-positive rate.")
        st.line_chart(roc_data.set_index("False positive rate"), use_container_width=True)

    st.markdown("#### Threshold comparison — held-out test split")
    st.caption(
        "All rows use the same test data and saved Random Forest. "
        "Changing the threshold shifts the precision/recall trade-off."
    )
    st.line_chart(
        threshold_data.set_index("Threshold")[["Precision", "Recall", "F1-score"]],
        use_container_width=True,
    )
    st.dataframe(
        threshold_data.style.format({
            "Threshold": "{:.0%}", "Accuracy": "{:.2%}",
            "Precision": "{:.2%}", "Recall": "{:.2%}",
            "F1-score": "{:.4f}",
            "False positives": "{:,.0f}", "False negatives": "{:,.0f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    if FINAL_RESULTS_PATH.exists():
        try:
            saved = load_results(str(FINAL_RESULTS_PATH))
            if not saved.empty:
                saved_r = saved.iloc[0]
                pairs   = {
                    "accuracy": "Accuracy", "precision": "Precision",
                    "recall": "Recall", "f1_score": "F1-score",
                    "roc_auc": "ROC-AUC",
                }
                match = all(
                    np.isclose(float(saved_r[s]), float(result[d]), rtol=1e-4, atol=1e-6)
                    for s, d in pairs.items()
                )
                st.caption(
                    f"Saved snapshot threshold: {float(saved_r['threshold']):.0%}."
                )
                if match:
                    st.success("✅ Recomputed metrics match the saved evaluation snapshot.")
                else:
                    st.warning(
                        "⚠️ Recomputed metrics differ from the saved snapshot. "
                        "Ensure the model and test data are from the same project run."
                    )
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as e:
            st.warning(f"Saved snapshot could not be read: {e}")


# ── Page: Dataset Explorer ────────────────────────────────────────────────────
def show_dataset_explorer() -> None:
    page_header("🗄️", "Dataset Explorer",
                "Dataset splits, class balance, CBC feature ranges, and record search")
    research_notice()

    if not DATA_DIR.exists():
        st.error(f"Dataset folder not found: `{DATA_DIR}`")
        return
    try:
        split_summary, label_counts = load_dataset_summary(str(DATA_DIR))
        feature_stats               = load_training_feature_stats(str(DATA_DIR))
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as e:
        st.error(f"Dataset information could not be loaded: {e}")
        return

    if not split_summary.empty:
        cols = st.columns(3)
        for col, row in zip(cols, split_summary.to_dict("records")):
            col.metric(
                f"{row['Dataset split']} records",
                f"{int(row['Records']):,}",
                delta=f"{int(row['Files']):,} CSV files",
                delta_color="off",
            )
        st.subheader("Dataset split summary")
        st.dataframe(split_summary, hide_index=True, use_container_width=True)

    if not label_counts.empty:
        st.subheader("Healthy vs. leukemia label distribution")
        label_chart = label_counts.pivot(
            index="Dataset split", columns="Dataset label", values="Records"
        ).fillna(0)
        st.bar_chart(label_chart, use_container_width=True)
        st.dataframe(label_counts, hide_index=True, use_container_width=True)

    st.subheader("CBC feature statistics")
    st.caption("Calculated from the training split only.")
    st.dataframe(
        feature_stats.style.format("{:,.2f}"),
        use_container_width=True,
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Search held-out test records")
    st.caption(
        "Shows record ID, dataset label, and eight CBC fields. "
        "Use de-identified data for demonstrations."
    )

    if not TEST_DATA_PATH.exists():
        st.warning(f"Test records not found: `{TEST_DATA_PATH}`")
        return
    try:
        records = load_explorer_records(str(TEST_DATA_PATH))
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as e:
        st.error(f"Test records could not be loaded: {e}")
        return

    filters = st.columns([2, 1])
    search  = filters[0].text_input(
        "Search by record ID", placeholder="Type at least 2 characters"
    ).strip()
    label_filter = filters[1].selectbox(
        "Filter by label", ["All labels", "healthy", "leukemia"]
    )

    visible = records
    if search:
        visible = visible.loc[
            visible[ID_COLUMN].astype(str).str.contains(
                search, case=False, regex=False, na=False
            )
        ]
    if label_filter != "All labels":
        visible = visible.loc[visible[TARGET].astype(str).str.lower() == label_filter]

    st.caption(f"{len(visible):,} matching record(s) · showing up to 100.")
    st.dataframe(visible.head(100), hide_index=True, use_container_width=True)


# ── Page: About the Project ───────────────────────────────────────────────────
def show_about_project() -> None:
    page_header("💡", "About the Project",
                "Problem statement, methodology, model details, and disclaimer")
    research_notice()

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("Project objective")
            st.write(
                "Demonstrate a reproducible data workflow: prepare CBC features, "
                "train classification models, compare saved evaluation results, "
                "and present an experimental model output."
            )
    with c2:
        with st.container(border=True):
            st.subheader("Dataset and target")
            st.write(
                "The project dataset contains CBC and analyzer-derived features "
                "with healthy and leukemia labels. It does not contain ESR "
                "measurements for this prediction workflow."
            )

    st.subheader("How the prototype works")
    st.markdown(
        """
        1. **Prepare data:** load training, validation, and test splits.
        2. **Train and save models:** use the project scripts and saved model files.
        3. **Run an experiment:** enter eight CBC values, upload a compatible CSV or Excel file,
           or select a saved test record.
        4. **Review evaluation:** inspect validation comparisons and separately labeled
           held-out test metrics on **Model Performance**.
        """
    )

    st.subheader("Available models")
    st.markdown(
        """
        | Model | Fields | Used for |
        |---|---|---|
        | **Manual CBC Random Forest** | 8 CBC fields | Manual entry workflow |
        | **Full-feature Random Forest** | 74 feature columns | CSV upload & saved records |
        | **Logistic Regression** | 74 columns | Validation comparison only |
        | **XGBoost** | 74 columns | Validation comparison only |
        """
    )

    st.subheader("Limitations and future work")
    st.markdown(
        """
        - This is an **educational research prototype**, not a medical device or clinical diagnostic tool.
        - The model is CBC-based — **not ESR-based**. A prediction score is not a clinical probability.
        - A CSV or Excel upload must include the exact full-model feature columns or be mapped to the 8 manual CBC fields.
        - PDF and image report extraction is **not implemented**. OCR and human verification would be required.
        - Further work includes external validation, safer data handling, and domain-expert review.
        """
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🩸</div>
            <div class="sidebar-logo-text">
                <div class="name">BloodCancerESR</div>
                <div class="sub">CBC Analysis & Research Dashboard</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:.68rem;font-weight:700;letter-spacing:.1em;'
        'text-transform:uppercase;color:#8fa8c0;margin-bottom:10px;">'
        'Navigation</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "⚡  Overview",
            "🔮  Prediction Studio",
            "📊  Model Performance",
            "🗄️  Dataset Explorer",
            "💡  About the Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Quick model status
    manual_ok = MANUAL_MODEL_PATH.exists()
    full_ok   = FULL_MODEL_PATH.exists()
    st.markdown(
        f"""
        <div style="font-size:.75rem;color:#8fa8c0;margin-bottom:6px;
                    font-weight:700;letter-spacing:.08em;text-transform:uppercase;">
            Model status
        </div>
        <div style="font-size:.8rem;color:#{'38c9b5' if manual_ok else 'f87171'};
                    margin-bottom:4px;">
            {'✅' if manual_ok else '❌'} manual_cbc_model.joblib
        </div>
        <div style="font-size:.8rem;color:#{'38c9b5' if full_ok else 'f87171'};">
            {'✅' if full_ok else '❌'} random_forest_model.joblib
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:.72rem;color:#8fa8c0;line-height:1.6;">'
        'Research prototype · Not ESR-based<br>'
        'Not for clinical diagnosis'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Routing ───────────────────────────────────────────────────────────────────
if "Overview" in page:
    show_overview()
elif "Prediction Studio" in page:
    show_prediction_studio()
elif "Model Performance" in page:
    show_model_performance()
elif "Dataset Explorer" in page:
    show_dataset_explorer()
else:
    show_about_project()
