from __future__ import annotations

import pandas as pd


def validate_datasets(config: dict, master: dict[str, pd.DataFrame], transactions: pd.DataFrame,
                      snapshots: pd.DataFrame, ground_truth: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if transactions["transaction_id"].duplicated().any():
        errors.append("transaction_id values are not unique")
    if transactions["event_id"].duplicated().any():
        errors.append("event_id values are not unique")
    if not set(transactions["product_id"]).issubset(set(master["products"]["product_id"])):
        errors.append("transactions contain unknown product_id")
    if not set(transactions["store_id"]).issubset(set(master["stores"]["store_id"])):
        errors.append("transactions contain unknown store_id")
    if not set(transactions["warehouse_id"]).issubset(set(master["warehouses"]["warehouse_id"])):
        errors.append("transactions contain unknown warehouse_id")
    if transactions[["transaction_id", "timestamp", "store_id", "product_id", "transaction_type", "quantity"]].isna().any().any():
        errors.append("required transaction fields contain nulls")
    expected_count = round(len(transactions) * float(config["generation"]["target_anomaly_rate_pct"]) / 100)
    if len(ground_truth) != expected_count:
        errors.append(f"ground truth count {len(ground_truth)} does not match expected {expected_count}")
    if not set(ground_truth["event_id"]).issubset(set(transactions["event_id"])):
        errors.append("ground truth contains unknown event_id")
    if snapshots["variance_quantity"].isna().any():
        errors.append("snapshot variance contains nulls")
    return errors


def quality_report(master: dict[str, pd.DataFrame], transactions: pd.DataFrame,
                   snapshots: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    tables = {**master, "transactions": transactions, "inventory_snapshots": snapshots, "anomaly_ground_truth": ground_truth}
    return {
        name: {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "null_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }
        for name, df in tables.items()
    }
