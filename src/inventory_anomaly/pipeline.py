from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from .config import load_config
from .anomaly_model import train_and_score_transactions
from .generators import (
    build_ground_truth, derive_inventory_snapshots, generate_master_data, generate_transactions
)
from .validation import quality_report, validate_datasets


def write_table(df: pd.DataFrame, base: Path, formats: list[str]) -> None:
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "parquet":
            df.to_parquet(path, index=False, compression="snappy")


def run_pipeline(config_path: str, output_dir: str) -> dict:
    config = load_config(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    formats = config["generation"]["output_formats"]

    master = generate_master_data(config)
    transactions = generate_transactions(config, master)
    snapshots = derive_inventory_snapshots(config, master, transactions)
    ground_truth = build_ground_truth(transactions)
    errors = validate_datasets(config, master, transactions, snapshots, ground_truth)
    report = quality_report(master, transactions, snapshots, ground_truth)
    if errors:
        raise ValueError("Data validation failed: " + "; ".join(errors))

    scored_transactions, model_metrics = train_and_score_transactions(config, transactions)

    for name, df in master.items():
        write_table(df, out / name, formats)
    write_table(transactions, out / "inventory_transactions", formats)
    write_table(snapshots, out / "inventory_snapshots", formats)
    write_table(ground_truth, out / "anomaly_ground_truth", formats)
    write_table(scored_transactions, out / "scored_transactions", formats)
    (out / "model_metrics.json").write_text(json.dumps(model_metrics, indent=2), encoding="utf-8")

    manifest = {
        "project": config["project"]["name"],
        "schema_version": config["project"]["schema_version"],
        "seed": config["project"]["seed"],
        "config": config,
        "tables": report,
        "validation": {"status": "PASSED", "errors": []},
        "anomaly_rate_actual_pct": round(100 * len(ground_truth) / len(transactions), 4),
        "model_metrics": model_metrics,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic retail inventory datasets.")
    parser.add_argument("--config", default="config/inventory_config.yaml")
    parser.add_argument("--output", default="data/output")
    args = parser.parse_args()
    manifest = run_pipeline(args.config, args.output)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
