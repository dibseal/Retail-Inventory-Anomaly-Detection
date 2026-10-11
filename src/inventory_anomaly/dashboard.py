from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# dashboard.py lives in <project-root>/src/inventory_anomaly/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


st.set_page_config(
    page_title="Retail Inventory Intelligence",
    page_icon="📦",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_table(output_dir: str, stem: str) -> pd.DataFrame:
    """Load a generated table, preferring CSV and falling back to Parquet."""
    folder = Path(output_dir)
    csv_path = folder / f"{stem}.csv"
    parquet_path = folder / f"{stem}.parquet"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def safe_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def main() -> None:
    st.title("📦 Retail Inventory Intelligence")
    st.caption("Inventory anomaly detection, transaction investigation, and model evaluation")

    with st.sidebar:
        st.header("Dashboard settings")
        output_dir_text = st.text_input(
            "Output directory",
            value=str(DEFAULT_OUTPUT_DIR),
            help="Folder containing the generated CSV/Parquet datasets and model_metrics.json.",
        )
        st.caption("Run the data pipeline before opening this dashboard.")

    output_dir = Path(output_dir_text).expanduser()
    metrics = load_json(str(output_dir / "model_metrics.json"))
    scored = load_table(str(output_dir), "scored_transactions")
    snapshots = load_table(str(output_dir), "inventory_snapshots")
    products = load_table(str(output_dir), "products")
    stores = load_table(str(output_dir), "stores")
    manifest = load_json(str(output_dir / "manifest.json"))

    if not output_dir.exists():
        st.error(f"Output directory does not exist: {output_dir}")
        st.info("Run the pipeline from the project root, then refresh this page.")
        st.code(
            r".\.venv\Scripts\python.exe -m inventory_anomaly.pipeline "
            r"--config config/inventory_config.yaml --output data/output"
        )
        return

    if scored.empty:
        st.error("Could not find scored_transactions.csv or scored_transactions.parquet.")
        st.info("Run the pipeline successfully before starting the dashboard.")
        st.code(
            r".\.venv\Scripts\python.exe -m inventory_anomaly.pipeline "
            r"--config config/inventory_config.yaml --output data/output"
        )
        return

    scored = safe_datetime(scored, "timestamp")
    snapshots = safe_datetime(snapshots, "snapshot_date")

    if "predicted_is_anomaly" in scored.columns:
        scored["predicted_is_anomaly"] = (
            scored["predicted_is_anomaly"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        )
    else:
        scored["predicted_is_anomaly"] = False

    # Sidebar filters are shared by transaction charts and the investigation table.
    with st.sidebar:
        st.divider()
        st.subheader("Transaction filters")

        if "store_id" in scored.columns:
            store_values = sorted(scored["store_id"].dropna().astype(str).unique().tolist())
            selected_stores = st.multiselect("Stores", store_values, default=[])
        else:
            selected_stores = []

        if "transaction_type" in scored.columns:
            type_values = sorted(scored["transaction_type"].dropna().astype(str).unique().tolist())
            selected_types = st.multiselect("Transaction types", type_values, default=[])
        else:
            selected_types = []

        anomaly_filter = st.selectbox(
            "Prediction status",
            ["All transactions", "Predicted anomalies only", "Predicted normal only"],
        )

        if "timestamp" in scored.columns and scored["timestamp"].notna().any():
            min_ts = scored["timestamp"].min().date()
            max_ts = scored["timestamp"].max().date()
            date_range = st.date_input(
                "Transaction date range",
                value=(min_ts, max_ts),
                min_value=min_ts,
                max_value=max_ts,
            )
        else:
            date_range = None

    filtered = scored.copy()
    if selected_stores and "store_id" in filtered.columns:
        filtered = filtered[filtered["store_id"].astype(str).isin(selected_stores)]
    if selected_types and "transaction_type" in filtered.columns:
        filtered = filtered[filtered["transaction_type"].astype(str).isin(selected_types)]
    if anomaly_filter == "Predicted anomalies only":
        filtered = filtered[filtered["predicted_is_anomaly"]]
    elif anomaly_filter == "Predicted normal only":
        filtered = filtered[~filtered["predicted_is_anomaly"]]
    if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2 and "timestamp" in filtered.columns:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        filtered = filtered[filtered["timestamp"].between(start_date, end_date)]

    predicted_count = int(scored["predicted_is_anomaly"].sum())
    total_count = int(len(scored))
    predicted_rate = predicted_count / total_count if total_count else 0.0

    st.subheader("Executive overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions scored", f"{total_count:,}")
    k2.metric("Predicted anomalies", f"{predicted_count:,}")
    k3.metric("Predicted anomaly rate", f"{predicted_rate:.2%}")
    k4.metric("Inventory snapshots", f"{len(snapshots):,}")

    # Model metrics are holdout metrics from model_metrics.json.
    st.subheader("Model evaluation — chronological holdout")
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", format_pct(metrics.get("precision")))
        m2.metric("Recall", format_pct(metrics.get("recall")))
        m3.metric("F1-score", format_pct(metrics.get("f1_score")))
        m4.metric("False-positive rate", format_pct(metrics.get("false_positive_rate")))

        st.caption(
            f"Model: {metrics.get('model', 'Unknown')} · "
            f"Split: {metrics.get('split_strategy', 'Not specified')} · "
            f"Train rows: {metrics.get('train_rows', 'N/A'):,} · "
            f"Test rows: {metrics.get('test_rows', 'N/A'):,}"
        )
        st.warning(
            "This is a synthetic-data POC. Holdout metrics measure agreement with injected labels, "
            "not verified real-world inventory fraud or loss."
        )
    else:
        st.info("model_metrics.json was not found or could not be read.")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Predicted anomalies by transaction type")
        if "transaction_type" in filtered.columns and not filtered.empty:
            chart_data = (
                filtered[filtered["predicted_is_anomaly"]]
                .groupby("transaction_type", dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            if chart_data.empty:
                st.info("No predicted anomalies match the current filters.")
            else:
                fig = px.bar(
                    chart_data,
                    x="transaction_type",
                    y="count",
                    labels={"transaction_type": "Transaction type", "count": "Predicted anomalies"},
                )
                fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Transaction type data is unavailable.")

    with right:
        st.markdown("#### Anomaly score distribution")
        if "anomaly_score" in filtered.columns and not filtered.empty:
            fig = px.histogram(
                filtered,
                x="anomaly_score",
                color="predicted_is_anomaly",
                nbins=40,
                barmode="overlay",
                labels={"anomaly_score": "Anomaly score", "predicted_is_anomaly": "Predicted anomaly"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Anomaly score data is unavailable.")

    st.subheader("Anomaly trend")
    if "timestamp" in filtered.columns and filtered["timestamp"].notna().any():
        trend_source = filtered[filtered["predicted_is_anomaly"]].copy()
        if not trend_source.empty:
            trend_source["date"] = trend_source["timestamp"].dt.date
            trend = trend_source.groupby("date").size().reset_index(name="predicted_anomalies")
            fig = px.line(
                trend,
                x="date",
                y="predicted_anomalies",
                markers=True,
                labels={"date": "Date", "predicted_anomalies": "Predicted anomalies"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predicted anomalies match the current filters.")
    else:
        st.info("Timestamps are unavailable for the anomaly trend.")

    st.subheader("Transaction investigation")
    st.caption(f"Showing {len(filtered):,} of {len(scored):,} transactions after filters.")

    if "anomaly_score" in filtered.columns:
        table = filtered.sort_values("anomaly_score", ascending=False)
    else:
        table = filtered.copy()

    preferred_columns = [
        "transaction_id",
        "timestamp",
        "store_id",
        "product_id",
        "transaction_type",
        "quantity",
        "source_system",
        "anomaly_score",
        "predicted_is_anomaly",
        "is_anomaly",
        "anomaly_type",
        "severity",
        "evaluation_split",
    ]
    display_columns = [column for column in preferred_columns if column in table.columns]
    if not display_columns:
        display_columns = table.columns.tolist()

    st.dataframe(
        table[display_columns].head(1000),
        use_container_width=True,
        hide_index=True,
    )

    if not table.empty:
        csv_bytes = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered transactions as CSV",
            data=csv_bytes,
            file_name="filtered_scored_transactions.csv",
            mime="text/csv",
        )

    st.subheader("Inventory reconciliation")
    if snapshots.empty:
        st.info("Inventory snapshots were not found.")
    elif {"variance_quantity", "snapshot_date"}.issubset(snapshots.columns):
        variance = pd.to_numeric(snapshots["variance_quantity"], errors="coerce").fillna(0)
        anomalous_snapshots = snapshots.loc[variance.ne(0)].copy()
        a, b, c = st.columns(3)
        a.metric("Snapshots with variance", f"{len(anomalous_snapshots):,}")
        b.metric("Absolute variance units", f"{variance.abs().sum():,.0f}")
        c.metric("Stores represented", f"{snapshots['store_id'].nunique():,}" if "store_id" in snapshots.columns else "N/A")

        if not anomalous_snapshots.empty:
            anomalous_snapshots["absolute_variance"] = pd.to_numeric(
                anomalous_snapshots["variance_quantity"], errors="coerce"
            ).abs()
            top_variance = anomalous_snapshots.nlargest(20, "absolute_variance")
            labels = [
                f"{row.get('store_id', 'Store')} / {row.get('product_id', 'Product')}"
                for _, row in top_variance.iterrows()
            ]
            fig = px.bar(
                top_variance.assign(label=labels),
                x="label",
                y="variance_quantity",
                labels={"label": "Store / product", "variance_quantity": "Variance units"},
                title="Largest snapshot variances",
            )
            fig.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                top_variance.drop(columns=["absolute_variance"], errors="ignore"),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No non-zero inventory snapshot variances were found.")
    else:
        st.info("Snapshot data does not contain the expected variance/date columns.")

    with st.expander("Technical details"):
        st.write("**Output directory:**", str(output_dir.resolve()))
        if manifest:
            st.write("**Project:**", manifest.get("project", "Unknown"))
            st.write("**Schema version:**", manifest.get("schema_version", "Unknown"))
            st.write("**Data validation:**", manifest.get("validation", {}).get("status", "Unknown"))
        if metrics.get("features"):
            st.write("**Model features:**", metrics["features"])
        if metrics.get("confusion_matrix"):
            st.json(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
