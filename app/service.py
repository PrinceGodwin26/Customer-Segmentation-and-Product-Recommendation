"""Orchestrates the full pipeline and holds the latest run's results in memory."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from . import config
from .clustering import (
    apply_pca,
    cluster_centroids_standardized,
    detect_outliers,
    evaluate_clustering,
    run_kmeans,
    scale_features,
)
from .pipeline import build_customer_features, clean_transactions, load_raw_data
from .recommender import generate_recommendations


class PipelineNotReadyError(RuntimeError):
    pass


@dataclass
class PipelineState:
    ready: bool = False
    last_run_at: Optional[str] = None
    data_path: Optional[str] = None
    raw_row_count: int = 0
    data_quality: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    cluster_summary: list = field(default_factory=list)
    customer_data_cleaned: Optional[pd.DataFrame] = None
    customer_data_pca: Optional[pd.DataFrame] = None
    outliers_data: Optional[pd.DataFrame] = None
    recommendations_df: Optional[pd.DataFrame] = None
    top_products_per_cluster: Optional[pd.DataFrame] = None
    error: Optional[str] = None


STATE = PipelineState()


def _build_cluster_summary(customer_data_cleaned: pd.DataFrame) -> list:
    counts = customer_data_cleaned["cluster"].value_counts().sort_index()
    percentages = customer_data_cleaned["cluster"].value_counts(normalize=True).sort_index() * 100
    centroids = cluster_centroids_standardized(customer_data_cleaned)

    summary = []
    for cluster_id in counts.index:
        summary.append(
            {
                "cluster": int(cluster_id),
                "count": int(counts.loc[cluster_id]),
                "percentage": round(float(percentages.loc[cluster_id]), 2),
                "centroid": {
                    str(k): float(v) for k, v in centroids.loc[cluster_id].to_dict().items()
                },
            }
        )
    return summary


def run_pipeline(data_path: Optional[str] = None) -> PipelineState:
    path = data_path or config.DATA_PATH

    raw_df = load_raw_data(path)
    cleaned_df, data_quality = clean_transactions(raw_df)
    customer_data = build_customer_features(cleaned_df)

    customer_data_cleaned, outliers_data = detect_outliers(
        customer_data,
        contamination=config.ISOLATION_FOREST_CONTAMINATION,
        random_state=config.RANDOM_STATE,
    )

    customer_data_scaled, _scaler = scale_features(customer_data_cleaned)
    customer_data_pca, _pca = apply_pca(
        customer_data_scaled, n_components=config.PCA_N_COMPONENTS
    )
    # Keep CustomerID numeric everywhere downstream (it matches the raw
    # transaction data and the recommendations table), instead of the string
    # dtype build_customer_features() uses internally.
    customer_data_pca.index = customer_data_pca.index.astype(float)

    labels, _kmeans = run_kmeans(
        customer_data_pca, n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_STATE
    )
    customer_data_cleaned = customer_data_cleaned.copy()
    customer_data_cleaned["cluster"] = labels
    customer_data_pca = customer_data_pca.copy()
    customer_data_pca["cluster"] = labels

    metrics = evaluate_clustering(customer_data_pca.drop(columns=["cluster"]), labels)
    cluster_summary = _build_cluster_summary(customer_data_cleaned)

    # Recommendations need numeric CustomerIDs (matches the raw transaction data)
    # and must exclude outlier customers' transactions.
    outlier_ids = outliers_data["CustomerID"].astype(float).unique()

    cleaned_for_join = customer_data_cleaned.copy()
    cleaned_for_join["CustomerID"] = cleaned_for_join["CustomerID"].astype(float)

    df_filtered = cleaned_df.copy()
    df_filtered["CustomerID"] = df_filtered["CustomerID"].astype(float)
    df_filtered = df_filtered[~df_filtered["CustomerID"].isin(outlier_ids)]

    recommendations_df, top_products_per_cluster = generate_recommendations(
        df_filtered,
        cleaned_for_join,
        top_n_products=config.TOP_N_PRODUCTS_PER_CLUSTER,
        top_n_recommendations=config.TOP_N_RECOMMENDATIONS,
    )

    STATE.ready = True
    STATE.error = None
    STATE.last_run_at = datetime.now(timezone.utc).isoformat()
    STATE.data_path = str(path)
    STATE.raw_row_count = int(len(raw_df))
    STATE.data_quality = data_quality
    STATE.metrics = metrics
    STATE.cluster_summary = cluster_summary
    STATE.customer_data_cleaned = cleaned_for_join
    STATE.customer_data_pca = customer_data_pca
    STATE.outliers_data = outliers_data
    STATE.recommendations_df = recommendations_df
    STATE.top_products_per_cluster = top_products_per_cluster
    return STATE


def get_state() -> PipelineState:
    return STATE


def ensure_ready() -> PipelineState:
    if not STATE.ready:
        raise PipelineNotReadyError(
            "Pipeline has not run yet. Call POST /pipeline/run first."
        )
    return STATE
