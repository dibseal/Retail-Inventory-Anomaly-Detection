from __future__ import annotations

import numpy as np
import pandas as pd


def build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build model inputs without using ground-truth labels or label-derived fields.

    Features are calculated from transaction attributes and prior history only.
    In particular, source_system is deliberately excluded because the synthetic
    DUPLICATE_FEED value would reveal one injected anomaly class directly.
    """
    required = {"timestamp", "store_id", "product_id", "transaction_type", "quantity", "unit_cost"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Transactions missing required feature columns: {sorted(missing)}")

    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="raise")
    tx = tx.sort_values(["timestamp", "transaction_id"] if "transaction_id" in tx.columns else ["timestamp"]).copy()
    qty = pd.to_numeric(tx["quantity"], errors="coerce").fillna(0).astype(float)
    abs_qty = qty.abs()

    features = pd.DataFrame(index=tx.index)
    features["quantity_abs_log"] = np.log1p(abs_qty)
    features["quantity_sign"] = np.sign(qty)
    features["unit_cost_log"] = np.log1p(pd.to_numeric(tx["unit_cost"], errors="coerce").fillna(0).clip(lower=0))
    features["transaction_hour"] = tx["timestamp"].dt.hour.astype(float)
    features["day_of_week"] = tx["timestamp"].dt.dayofweek.astype(float)
    features["is_weekend"] = tx["timestamp"].dt.dayofweek.isin([5, 6]).astype(float)

    # Prior-only history features for each store/product pair. Current transaction
    # quantity is excluded from its own historical statistics.
    keys = [tx["store_id"].astype(str), tx["product_id"].astype(str)]
    group_key = keys[0] + "::" + keys[1]
    grouped_abs = abs_qty.groupby(group_key, sort=False)
    prior_count = grouped_abs.cumcount().astype(float)
    cumulative_sum = grouped_abs.cumsum() - abs_qty
    cumulative_sq_sum = (abs_qty.pow(2).groupby(group_key, sort=False).cumsum() - abs_qty.pow(2))
    prior_mean = cumulative_sum.div(prior_count.replace(0, np.nan))
    prior_var = cumulative_sq_sum.div(prior_count.replace(0, np.nan)) - prior_mean.pow(2)

    features["store_product_prior_count"] = prior_count
    features["store_product_prior_mean_abs_quantity"] = prior_mean.fillna(0).clip(lower=0)
    features["store_product_prior_std_abs_quantity"] = np.sqrt(prior_var.clip(lower=0).fillna(0))
    features["quantity_vs_prior_mean"] = abs_qty.div(prior_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(upper=1000)

    # Transaction type is a legitimate operational attribute, not the injected label.
    type_dummies = pd.get_dummies(tx["transaction_type"].fillna("UNKNOWN"), prefix="type", dtype=float)
    type_dummies.index = tx.index
    features = pd.concat([features, type_dummies], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    return features
