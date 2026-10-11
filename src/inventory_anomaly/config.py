from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    generation = config.get("generation", {})
    for key in ("products", "stores", "warehouses", "history_days", "target_transactions"):
        if int(generation.get(key, 0)) <= 0:
            raise ValueError(f"generation.{key} must be greater than zero")
    rate = float(generation.get("target_anomaly_rate_pct", -1))
    if not 0 <= rate <= 100:
        raise ValueError("generation.target_anomaly_rate_pct must be between 0 and 100")
    formats = set(generation.get("output_formats", []))
    if not formats.issubset({"csv", "parquet"}):
        raise ValueError("output_formats supports only csv and parquet")
