from __future__ import annotations

import numpy as np
import pandas as pd


def generate_master_data(config: dict) -> dict[str, pd.DataFrame]:
    gen = config["generation"]
    business = config["business"]
    rng = np.random.default_rng(int(config["project"]["seed"]))
    categories = business["categories"]
    regions = business["regions"]

    products = pd.DataFrame({
        "product_id": [f"SKU{i:06d}" for i in range(1, int(gen["products"]) + 1)],
        "category": rng.choice(categories, int(gen["products"])),
        "unit_cost": np.round(rng.lognormal(mean=3.1, sigma=0.65, size=int(gen["products"])), 2),
        "unit_price": 0.0,
        "supplier_id": [f"SUP{(i % max(20, int(gen['products']) // 20)) + 1:05d}" for i in range(int(gen["products"]))],
        "active": True,
    })
    products["unit_price"] = np.round(products["unit_cost"] * rng.uniform(1.3, 2.6, len(products)), 2)

    warehouse_count = int(gen["warehouses"])
    stores = pd.DataFrame({
        "store_id": [f"STR{i:04d}" for i in range(1, int(gen["stores"]) + 1)],
        "region": rng.choice(regions, int(gen["stores"])),
        "store_type": rng.choice(["Flagship", "Standard", "Outlet"], int(gen["stores"]), p=[0.1, 0.7, 0.2]),
        "square_feet": rng.integers(3000, 30000, int(gen["stores"])),
        "warehouse_id": [f"WH{(i % warehouse_count) + 1:03d}" for i in range(int(gen["stores"]))],
    })
    warehouses = pd.DataFrame({
        "warehouse_id": [f"WH{i:03d}" for i in range(1, int(gen["warehouses"]) + 1)],
        "region": regions[:int(gen["warehouses"])],
        "capacity_units": rng.integers(100000, 1000000, int(gen["warehouses"])),
    })
    return {"products": products, "stores": stores, "warehouses": warehouses}


def generate_transactions(config: dict, master: dict[str, pd.DataFrame]) -> pd.DataFrame:
    gen, seed = config["generation"], int(config["project"]["seed"])
    rng = np.random.default_rng(seed + 1)
    n = int(gen["target_transactions"])
    start = pd.Timestamp(gen["start_date"])
    dates = start + pd.to_timedelta(rng.integers(0, int(gen["history_days"]), n), unit="D")
    products = master["products"]
    stores = master["stores"]
    product_idx = rng.integers(0, len(products), n)
    store_idx = rng.integers(0, len(stores), n)
    types = rng.choice(
        ["RECEIPT", "SALE", "TRANSFER_OUT", "TRANSFER_IN", "RETURN", "ADJUSTMENT"],
        n, p=[0.12, 0.58, 0.08, 0.08, 0.06, 0.08],
    )
    quantities = np.maximum(1, rng.negative_binomial(2, 0.35, n) + 1)
    # Signed quantity conventions: stock-in positive; stock-out negative.
    signed = quantities.astype(int)
    signed[np.isin(types, ["SALE", "TRANSFER_OUT"])] *= -1
    signed[types == "ADJUSTMENT"] *= rng.choice([-1, 1], (types == "ADJUSTMENT").sum())
    # Returns and receipts remain positive.
    ts = pd.to_datetime(dates) + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
    chosen_store_ids = stores["store_id"].to_numpy()[store_idx]
    store_to_warehouse = stores.set_index("store_id")["warehouse_id"].to_dict()

    txn = pd.DataFrame({
        "transaction_id": [f"TXN{i:09d}" for i in range(1, n + 1)],
        "event_id": [f"EVT{i:09d}" for i in range(1, n + 1)],
        "transfer_id": None,
        "timestamp": ts,
        "store_id": chosen_store_ids,
        "product_id": products["product_id"].to_numpy()[product_idx],
        "warehouse_id": [store_to_warehouse[s] for s in chosen_store_ids],
        "transaction_type": types,
        "quantity": signed,
        "unit_cost": products["unit_cost"].to_numpy()[product_idx],
        "source_system": rng.choice(["POS", "WMS", "ERP", "MOBILE_SCANNER"], n, p=[0.55, 0.2, 0.2, 0.05]),
        "is_anomaly": False,
        "anomaly_type": "NONE",
        "severity": "NONE",
    })
    # Inject anomalies by selecting a deterministic set of events. Labels are retained
    # in a separate table and are not intended as model features.
    anomaly_count = round(n * float(gen["target_anomaly_rate_pct"]) / 100)
    if anomaly_count:
        selected = rng.choice(n, anomaly_count, replace=False)
        anomaly_types = np.array(["UNEXPLAINED_SHRINKAGE", "UNUSUAL_VELOCITY", "DUPLICATE_TRANSACTION", "COUNTING_ERROR"])
        for j, idx in enumerate(selected):
            kind = anomaly_types[j % len(anomaly_types)]
            txn.loc[idx, "is_anomaly"] = True
            txn.loc[idx, "anomaly_type"] = kind
            txn.loc[idx, "severity"] = rng.choice(["LOW", "MEDIUM", "HIGH"], p=[0.25, 0.5, 0.25])
            if kind == "UNEXPLAINED_SHRINKAGE":
                txn.loc[idx, "transaction_type"] = "ADJUSTMENT"
                txn.loc[idx, "quantity"] = -int(rng.integers(20, 150))
            elif kind == "UNUSUAL_VELOCITY":
                txn.loc[idx, "quantity"] = int(txn.loc[idx, "quantity"]) * int(rng.integers(8, 25))
            elif kind == "DUPLICATE_TRANSACTION":
                txn.loc[idx, "source_system"] = "DUPLICATE_FEED"
            elif kind == "COUNTING_ERROR":
                txn.loc[idx, "transaction_type"] = "ADJUSTMENT"
                txn.loc[idx, "quantity"] = int(rng.choice([-1, 1])) * int(rng.integers(50, 250))
    return txn.sort_values("timestamp").reset_index(drop=True)


def derive_inventory_snapshots(config: dict, master: dict[str, pd.DataFrame], transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Derive daily stock snapshots by replaying signed transaction quantities.

    The expected balance is the ledger-derived stock position. The observed
    balance is normally identical, except for explicitly injected COUNTING_ERROR
    events, which create a measurable observed-vs-expected discrepancy.
    """
    start = pd.Timestamp(config["generation"]["start_date"]).normalize()
    days = int(config["generation"]["history_days"])
    seed = int(config["project"]["seed"])

    # Build the snapshot universe from actual transaction pairs, then sample
    # deterministically if the requested snapshot size would be too large.
    pairs = transactions[["store_id", "product_id"]].drop_duplicates().reset_index(drop=True)
    max_pairs = min(
        len(master["stores"]) * len(master["products"]),
        max(5000, len(transactions) // 3),
    )
    if len(pairs) > max_pairs:
        pairs = pairs.sample(n=max_pairs, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed + 2)
    pairs["opening_quantity"] = rng.integers(5, 250, len(pairs))

    tx = transactions.copy()
    tx["snapshot_date"] = pd.to_datetime(tx["timestamp"]).dt.normalize()
    movement = tx.groupby(
        ["snapshot_date", "store_id", "product_id"], as_index=False
    )["quantity"].sum()
    movement_map = movement.set_index(
        ["snapshot_date", "store_id", "product_id"]
    )["quantity"].to_dict()

    current_balance = {
        (row.store_id, row.product_id): int(row.opening_quantity)
        for row in pairs.itertuples(index=False)
    }
    daily_rows = []

    # Index counting errors by date/store/product so they affect observed stock,
    # not the expected ledger balance.
    counting_errors = transactions[
        transactions["anomaly_type"] == "COUNTING_ERROR"
    ].copy()
    if not counting_errors.empty:
        counting_errors["snapshot_date"] = pd.to_datetime(
            counting_errors["timestamp"]
        ).dt.normalize()
        count_error_map = counting_errors.groupby(
            ["snapshot_date", "store_id", "product_id"]
        )["quantity"].sum().to_dict()
    else:
        count_error_map = {}

    for day_offset in range(days):
        day = start + pd.Timedelta(days=day_offset)
        for pair in pairs.itertuples(index=False):
            key = (pair.store_id, pair.product_id)
            opening = current_balance[key]
            net_movement = int(movement_map.get(
                (day, pair.store_id, pair.product_id), 0
            ))
            calculated_closing = opening + net_movement
            expected_closing = max(0, calculated_closing)

            # A physical count can disagree with the ledger. It is kept separate
            # from the expected balance to make the discrepancy measurable.
            count_delta = int(count_error_map.get(
                (day, pair.store_id, pair.product_id), 0
            ))
            observed_closing = max(0, expected_closing + count_delta)
            variance = observed_closing - expected_closing

            daily_rows.append({
                "snapshot_date": day,
                "store_id": pair.store_id,
                "product_id": pair.product_id,
                "opening_quantity": opening,
                "net_movement": net_movement,
                "calculated_closing_quantity": calculated_closing,
                "expected_closing_quantity": expected_closing,
                "observed_closing_quantity": observed_closing,
                "variance_quantity": variance,
                "is_anomaly": variance != 0,
            })
            # Carry forward the ledger's expected balance, not the observed
            # count, so a count discrepancy does not silently corrupt the ledger.
            current_balance[key] = expected_closing

    return pd.DataFrame(daily_rows)


def build_ground_truth(transactions: pd.DataFrame) -> pd.DataFrame:
    cols = ["event_id", "transaction_id", "timestamp", "store_id", "product_id", "is_anomaly", "anomaly_type", "severity"]
    return transactions.loc[transactions["is_anomaly"], cols].copy().reset_index(drop=True)
