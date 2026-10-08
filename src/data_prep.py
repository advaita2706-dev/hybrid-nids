"""
Data Preparation Pipeline for CICIDS2017 Dataset
Author: Advait (25BCY1014)
Role: Network & Data Lead

Cleans, merges and preprocesses all 8 CICIDS2017 CSVs into a single
analysis-ready DataFrame.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── paths ──────────────────────────────────────────────────────────────
RAW_DIR = os.environ.get("CICIDS_RAW_DIR", "data/raw")
OUT_DIR = os.environ.get("CICIDS_OUT_DIR", "data/processed")

CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]


def load_and_merge(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Load all CSVs and concatenate into one DataFrame."""
    frames = []
    for fname in CSV_FILES:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found")
            continue
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)
        print(f"  [loaded] {fname}: {len(df):,} rows")
    if not frames:
        raise FileNotFoundError(f"No CICIDS2017 CSVs found in {raw_dir}")
    merged = pd.concat(frames, ignore_index=True)
    print(f"\nTotal raw rows: {len(merged):,}")
    return merged


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove infinities, NaNs, duplicates and negative values."""
    initial = len(df)

    # Replace inf with NaN then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop rows with negative duration or byte counts
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if "duration" in col.lower() or "bytes" in col.lower() or "length" in col.lower():
            df = df[df[col] >= 0]

    print(f"Cleaned: {initial:,} → {len(df):,} rows ({initial - len(df):,} removed)")
    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """Encode the Label column and print class distribution."""
    df[label_col] = df[label_col].str.strip()
    print("\nClass distribution:")
    dist = df[label_col].value_counts()
    for cls, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {cls:30s} {count:>10,}  ({pct:5.2f}%)")

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df[label_col])
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("CICIDS2017 Data Preparation Pipeline")
    print("=" * 60)

    print("\n1. Loading CSVs...")
    df = load_and_merge()

    print("\n2. Cleaning...")
    df = clean(df)

    print("\n3. Encoding labels...")
    df = encode_labels(df)

    # Save processed data
    out_path = os.path.join(OUT_DIR, "cicids2017_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"\n4. Saved to {out_path}")
    print(f"   Final shape: {df.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
