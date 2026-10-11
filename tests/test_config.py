from pathlib import Path
import pytest
from inventory_anomaly.config import load_config, validate_config


def test_default_config_loads():
    config = load_config(Path(__file__).parents[1] / "config" / "inventory_config.yaml")
    assert config["generation"]["products"] == 1000
    assert config["generation"]["target_transactions"] == 100000


def test_invalid_anomaly_rate_rejected():
    config = {"generation": {"products": 1, "stores": 1, "warehouses": 1, "history_days": 1,
                             "target_transactions": 1, "target_anomaly_rate_pct": 101, "output_formats": ["csv"]}}
    with pytest.raises(ValueError):
        validate_config(config)
