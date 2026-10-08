"""
Random Forest Baseline Training for CICIDS2017
Author: Advait (25BCY1014)
Role: Network & Data Lead

Trains a Random Forest classifier on the cleaned CICIDS2017 dataset,
saves the model, feature importances, and per-class metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get("CICIDS_CLEAN", "data/processed/cicids2017_clean.csv")
MODEL_DIR = "models"

# ── hyperparameters ────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("Random Forest Baseline — CICIDS2017")
    print("=" * 60)

    # ── load ───────────────────────────────────────────────────────────
    print("\n1. Loading data...")
    df = pd.read_csv(DATA_PATH)
    label_col = "Label"
    drop_cols = [label_col, "label_encoded"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[label_col].values
    print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]:,}")

    # ── split ──────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"   Train: {len(X_train):,}  Test: {len(X_test):,}")

    # ── scale ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── train ──────────────────────────────────────────────────────────
    print("\n2. Training Random Forest...")
    print(f"   Params: {json.dumps(RF_PARAMS, indent=2)}")
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)

    # ── evaluate ───────────────────────────────────────────────────────
    print("\n3. Evaluating...")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    # ── feature importance ─────────────────────────────────────────────
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n4. Top 10 features:")
    for rank, idx in enumerate(indices[:10], 1):
        print(f"   {rank:2d}. {feature_cols[idx]:35s}  {importances[idx]:.4f}")

    # ── save ───────────────────────────────────────────────────────────
    joblib.dump(clf, os.path.join(MODEL_DIR, "rf_baseline.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    metrics_path = os.path.join(MODEL_DIR, "baseline_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Random Forest Baseline — CICIDS2017\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Hyperparameters:\n{json.dumps(RF_PARAMS, indent=2)}\n\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write("\nTop 10 Features:\n")
        for rank, idx in enumerate(indices[:10], 1):
            f.write(f"  {rank:2d}. {feature_cols[idx]:35s}  {importances[idx]:.4f}\n")

    feat_path = os.path.join(MODEL_DIR, "feature_importances.json")
    with open(feat_path, "w") as f:
        json.dump(
            {feature_cols[i]: float(importances[i]) for i in indices},
            f, indent=2,
        )

    print(f"\n5. Saved:")
    print(f"   Model     → {MODEL_DIR}/rf_baseline.pkl")
    print(f"   Scaler    → {MODEL_DIR}/scaler.pkl")
    print(f"   Metrics   → {metrics_path}")
    print(f"   Features  → {feat_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()
