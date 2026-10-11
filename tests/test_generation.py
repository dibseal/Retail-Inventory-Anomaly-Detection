from pathlib import Path
from inventory_anomaly.config import load_config
from inventory_anomaly.generators import generate_master_data, generate_transactions, build_ground_truth
from inventory_anomaly.validation import validate_datasets
from inventory_anomaly.generators import derive_inventory_snapshots


def small_config():
    config = load_config(Path(__file__).parents[1] / "config" / "inventory_config.yaml")
    config["generation"].update({"products": 30, "stores": 5, "warehouses": 2, "history_days": 7,
                                 "target_transactions": 1000, "start_date": "2026-01-01",
                                 "target_anomaly_rate_pct": 2})
    return config


def test_generation_relationships_and_labels():
    config = small_config()
    master = generate_master_data(config)
    transactions = generate_transactions(config, master)
    snapshots = derive_inventory_snapshots(config, master, transactions)
    truth = build_ground_truth(transactions)
    assert len(master["products"]) == 30
    assert len(transactions) == 1000
    assert len(truth) == 20
    assert transactions["transaction_id"].is_unique
    assert set(transactions["product_id"]).issubset(set(master["products"]["product_id"]))
    assert validate_datasets(config, master, transactions, snapshots, truth) == []


def test_seed_reproducibility():
    config = small_config()
    master1 = generate_master_data(config)
    master2 = generate_master_data(config)
    tx1 = generate_transactions(config, master1)
    tx2 = generate_transactions(config, master2)
    assert tx1["transaction_id"].tolist() == tx2["transaction_id"].tolist()
    assert tx1["quantity"].tolist() == tx2["quantity"].tolist()
