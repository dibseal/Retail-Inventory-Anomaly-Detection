from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .feature_engineering import build_transaction_features


def train_and_score_transactions(
    config: dict[str, Any], transactions: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train Isolation Forest on earlier transactions and evaluate on later ones.

    Ground-truth labels are used only after predictions are produced, for metrics.
    The fitted model is trained on the first 80% of rows by timestamp.
    """
    if len(transactions) < 20:
        raise ValueError("At least 20 transactions are required to train and evaluate the model")
    if "is_anomaly" not in transactions.columns:
        raise ValueError("Transactions must include is_anomaly for holdout evaluation")

    ordered = transactions.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise")
    ordered = ordered.sort_values(["timestamp", "transaction_id"] if "transaction_id" in ordered.columns else ["timestamp"]).reset_index(drop=True)
    features = build_transaction_features(ordered).reset_index(drop=True)
    labels = ordered["is_anomaly"].astype(bool).to_numpy()

    split_at = int(len(ordered) * 0.8)
    split_at = min(max(split_at, 1), len(ordered) - 1)
    train_x, test_x = features.iloc[:split_at], features.iloc[split_at:]
    test_y = labels[split_at:]

    contamination = float(config["generation"].get("target_anomaly_rate_pct", 2.0)) / 100.0
    contamination = min(max(contamination, 0.001), 0.5)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=int(config["project"]["seed"]),
        n_jobs=-1,
    )
    model.fit(train_x)

    # sklearn returns -1 for an outlier and +1 for an inlier.
    test_pred = model.predict(test_x) == -1
    all_pred = model.predict(features) == -1
    anomaly_score = -model.score_samples(features)  # larger means more anomalous

    tn, fp, fn, tp = confusion_matrix(test_y, test_pred, labels=[False, True]).ravel()
    metrics: dict[str, Any] = {
        "model": "IsolationForest",
        "split_strategy": "chronological_80_20",
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "feature_count": int(features.shape[1]),
        "features": list(features.columns),
        "train_anomaly_rate_pct": round(float(labels[:split_at].mean() * 100), 4),
        "test_anomaly_rate_pct": round(float(test_y.mean() * 100), 4),
        "predicted_test_anomaly_rate_pct": round(float(test_pred.mean() * 100), 4),
        "precision": round(float(precision_score(test_y, test_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(test_y, test_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(test_y, test_pred, zero_division=0)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
        "note": "Synthetic labels are evaluation targets only; they are excluded from model features.",
    }

    scored = ordered.copy()
    scored["anomaly_score"] = anomaly_score
    scored["predicted_is_anomaly"] = all_pred
    scored["evaluation_split"] = np.where(np.arange(len(scored)) < split_at, "train", "test")
    return scored, metrics
