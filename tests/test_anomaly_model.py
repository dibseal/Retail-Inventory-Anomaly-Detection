from pathlib import Path

from inventory_anomaly.anomaly_model import train_and_score_transactions
from inventory_anomaly.config import load_config
from inventory_anomaly.feature_engineering import build_transaction_features
from inventory_anomaly.generators import generate_master_data, generate_transactions


def small_config():
    config = load_config(Path(__file__).parents[1] / "config" / "inventory_config.yaml")
    config["generation"].update({
        "products": 30,
        "stores": 5,
        "warehouses": 2,
        "history_days": 7,
        "target_transactions": 500,
        "start_date": "2026-01-01",
        "target_anomaly_rate_pct": 2,
    })
    return config


def test_features_exclude_ground_truth_and_direct_label_leakage():
    config = small_config()
    master = generate_master_data(config)
    transactions = generate_transactions(config, master)

    features = build_transaction_features(transactions)

    forbidden = {"is_anomaly", "anomaly_type", "severity", "source_system", "transaction_id", "event_id"}
    assert forbidden.isdisjoint(set(features.columns))
    assert len(features) == len(transactions)
    assert features.notna().all().all()


def test_isolation_forest_returns_scores_and_holdout_metrics():
    config = small_config()
    master = generate_master_data(config)
    transactions = generate_transactions(config, master)

    scored, metrics = train_and_score_transactions(config, transactions)

    assert len(scored) == len(transactions)
    assert {"anomaly_score", "predicted_is_anomaly", "evaluation_split"}.issubset(scored.columns)
    assert set(scored["evaluation_split"]) == {"train", "test"}
    assert metrics["model"] == "IsolationForest"
    assert metrics["train_rows"] + metrics["test_rows"] == len(transactions)
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
