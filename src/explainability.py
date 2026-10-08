"""
SHAP Explainability Module for Hybrid NIDS
==========================================
Uses TreeExplainer to provide per-prediction feature explanations
for the Random Forest classifier.
"""

import numpy as np
import joblib
import shap


class SHAPExplainer:
    """Wraps shap.TreeExplainer for the RF model."""

    def __init__(self, model_path: str = "models/random_forest.pkl"):
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.classes = list(self.model.classes_)

    def explain(self, X: np.ndarray, feature_names: list[str], top_n: int = 5) -> dict:
        """
        Explain a single prediction.

        Parameters
        ----------
        X : np.ndarray of shape (1, n_features)
        feature_names : list of feature column names
        top_n : number of top contributing features to return

        Returns
        -------
        dict with prediction, confidence, anomaly info placeholder,
        and top SHAP feature contributions.
        """
        X = np.asarray(X).reshape(1, -1)
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        confidence = float(proba.max())
        pred_idx = self.classes.index(prediction)

        # shap_values is a list (one array per class) for multi-class
        shap_values = self.explainer.shap_values(X)
        # Handle both list-of-arrays (multi-class) and single array formats
        if isinstance(shap_values, list) and pred_idx < len(shap_values):
            sample_shap = shap_values[pred_idx][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sample_shap = shap_values[0, :, pred_idx] if pred_idx < shap_values.shape[2] else shap_values[0, :, 0]
        elif isinstance(shap_values, np.ndarray):
            sample_shap = shap_values[0]
        else:
            sample_shap = np.zeros(X.shape[1])

        top_indices = np.argsort(np.abs(sample_shap))[-top_n:][::-1]

        top_features = []
        for i in top_indices:
            fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            top_features.append({
                "feature": fname,
                "value": float(X[0, i]),
                "shap_value": float(sample_shap[i]),
                "direction": "increases" if sample_shap[i] > 0 else "decreases",
            })

        return {
            "prediction": str(prediction),
            "confidence": confidence,
            "class_probabilities": {
                cls: float(p) for cls, p in zip(self.classes, proba)
            },
            "top_features": top_features,
        }

    def explain_batch(self, X: np.ndarray, feature_names: list[str], top_n: int = 5) -> list[dict]:
        """Explain a batch of predictions."""
        results = []
        for i in range(X.shape[0]):
            results.append(self.explain(X[i:i+1], feature_names, top_n))
        return results
