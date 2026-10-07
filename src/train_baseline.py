#!/usr/bin/env python3
"""
Random Forest Baseline Model Training
Trains on CICIDS2017 dataset with balanced class weights
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_baseline_model(train_path, test_path, model_path):
    """Train Random Forest baseline model"""

    print("\n" + "="*80)
    print("RANDOM FOREST BASELINE TRAINING")
    print("="*80)

    # Load data
    print("\n📂 Loading training data...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Prepare features and labels
    X_train = train_df.drop('Label', axis=1)
    y_train = train_df['Label']
    X_test = test_df.drop('Label', axis=1)
    y_test = test_df['Label']

    print(f"   Training set: {len(X_train)} samples, {len(X_train.columns)} features")
    print(f"   Test set:     {len(X_test)} samples")
    print(f"   Classes:      {sorted(y_train.unique())}")

    # Train Random Forest
    print("\n🤖 Training Random Forest (this takes 5-10 minutes)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
        verbose=1
    )
    rf.fit(X_train, y_train)

    # Make predictions
    print("\n📊 Generating predictions on test set...")
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)

    # Evaluate
    print("\n" + "="*80)
    print("PER-CLASS METRICS (THIS IS WHAT YOU SHOW TOMORROW)")
    print("="*80)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=sorted(rf.classes_)
    )

    print(f"\n{'Attack Type':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
    print("-" * 85)

    for label, p, r, f, s in zip(sorted(rf.classes_), precision, recall, f1, support):
        print(f"{label:<30} | {p:.3f}       | {r:.3f}      | {f:.3f}       | {int(s):<10}")

    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)

    print(f"Macro-Average Precision: {avg_precision:.3f}")
    print(f"Macro-Average Recall:    {avg_recall:.3f}")
    print(f"Macro-Average F1-Score:  {avg_f1:.3f}")

    # Feature importance (top 10)
    print("\n📊 TOP 10 MOST IMPORTANT FEATURES:")
    print("-" * 50)
    feature_names = X_train.columns
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(top_indices, 1):
        print(f"   {rank:>2}. {feature_names[idx]:<30} {importances[idx]:.4f}")

    # Save model
    from pathlib import Path
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, model_path)
    print(f"\n💾 Model saved to: {model_path}")

    # Save metrics to file
    metrics_path = str(Path(model_path).parent / 'baseline_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("RANDOM FOREST BASELINE METRICS\n")
        f.write(f"Macro F1: {avg_f1:.4f}\n\n")
        f.write(f"{'Attack Type':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
        f.write("-" * 80 + "\n")
        for label, p, r, fi, s in zip(sorted(rf.classes_), precision, recall, f1, support):
            f.write(f"{label:<30} {p:>10.3f} {r:>10.3f} {fi:>10.3f} {int(s):>10}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Macro Average':<30} {avg_precision:>10.3f} {avg_recall:>10.3f} {avg_f1:>10.3f} {int(np.sum(support)):>10}\n")
    print(f"📄 Metrics saved to: {metrics_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE ✅")
    print(f"Macro-Average F1-Score: {avg_f1:.4f}")
    print("="*80 + "\n")

    return rf

if __name__ == '__main__':
    train_baseline_model(
        'data/processed/train.parquet',
        'data/processed/test.parquet',
        'models/random_forest_baseline.pkl'
    )
