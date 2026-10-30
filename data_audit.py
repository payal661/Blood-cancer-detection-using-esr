"""Utilities for auditing project data."""

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "LeukoAlert-project"
    / "leuk vs HC"
    / "dataset"
)

SPLITS = ["training", "validation", "test"]


def audit_dataset():
    if not DATASET_DIR.exists():
        print(f"Dataset folder not found:\n{DATASET_DIR}")
        print("Check where you extracted LeukoAlert-project.")
        return

    for split in SPLITS:
        folder = DATASET_DIR / split
        files = sorted(folder.glob("*.csv"))

        if not files:
            print(f"\nNo CSV files found in {folder}")
            continue

        frames = []

        print(f"\n{'=' * 55}")
        print(f"{split.upper()} DATA")
        print("=" * 55)

        for file in files:
            df = pd.read_csv(file)
            frames.append(df)

            esr_columns = [
                col for col in df.columns
                if "esr" in col.lower()
                or "sedimentation" in col.lower()
            ]

            print(f"\nFile: {file.name}")
            print(f"Rows: {len(df):,}")
            print(f"Columns: {len(df.columns)}")
            print(f"Missing cells: {df.isna().sum().sum():,}")
            print(f"Duplicate rows: {df.duplicated().sum():,}")

            if "sample_id" in df.columns:
                print(
                    "Duplicate sample IDs:",
                    df["sample_id"].duplicated().sum()
                )

            if "leukemia_label" in df.columns:
                print("Label distribution:")
                print(df["leukemia_label"].value_counts())

            print("ESR columns:", esr_columns or "None found")

        combined = pd.concat(frames, ignore_index=True)

        print(f"\nCombined {split} shape: {combined.shape}")

        if "leukemia_label" in combined.columns:
            print("\nCombined label distribution:")
            print(combined["leukemia_label"].value_counts())

        # Save a report, not a merged training dataset.
        report = pd.DataFrame({
            "column": combined.columns,
            "missing_values": [
                combined[col].isna().sum()
                for col in combined.columns
            ],
            "data_type": [
                str(combined[col].dtype)
                for col in combined.columns
            ],
            "unique_values": [
                combined[col].nunique(dropna=False)
                for col in combined.columns
            ],
        })

        output = PROJECT_DIR / f"{split}_audit.csv"
        report.to_csv(output, index=False)
        print(f"\nAudit report saved: {output.name}")


if __name__ == "__main__":
    audit_dataset()
