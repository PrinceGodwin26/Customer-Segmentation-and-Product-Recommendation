"""Outlier detection, scaling, PCA and KMeans clustering (notebook Step 5-10)."""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def detect_outliers(
    customer_data: pd.DataFrame, contamination: float = 0.05, random_state: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag outlier customers with Isolation Forest and split them out."""
    customer_data = customer_data.copy()
    model = IsolationForest(contamination=contamination, random_state=random_state)
    scores = model.fit_predict(customer_data.iloc[:, 1:].to_numpy())
    customer_data["Outlier_Scores"] = scores
    customer_data["Is_Outlier"] = [1 if x == -1 else 0 for x in scores]

    outliers_data = customer_data[customer_data["Is_Outlier"] == 1].reset_index(drop=True)
    cleaned = (
        customer_data[customer_data["Is_Outlier"] == 0]
        .drop(columns=["Outlier_Scores", "Is_Outlier"])
        .reset_index(drop=True)
    )
    return cleaned, outliers_data


def scale_features(customer_data_cleaned: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    columns_to_exclude = ["CustomerID", "Is_UK", "Day_Of_Week"]
    columns_to_scale = customer_data_cleaned.columns.difference(columns_to_exclude)

    scaled = customer_data_cleaned.copy()
    scaled[columns_to_scale] = scaler.fit_transform(scaled[columns_to_scale].astype(float))
    return scaled, scaler


def apply_pca(customer_data_scaled: pd.DataFrame, n_components: int = 6) -> tuple[pd.DataFrame, PCA]:
    scaled_indexed = customer_data_scaled.set_index("CustomerID")
    n_components = max(1, min(n_components, scaled_indexed.shape[0], scaled_indexed.shape[1]))

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(scaled_indexed.astype(float))
    pca_df = pd.DataFrame(
        transformed,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
        index=scaled_indexed.index,
    )
    return pca_df, pca


def run_kmeans(
    customer_data_pca: pd.DataFrame, n_clusters: int = 3, random_state: int = 0
) -> tuple[np.ndarray, KMeans]:
    n_clusters = max(1, min(n_clusters, len(customer_data_pca)))
    kmeans = KMeans(
        n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=100, random_state=random_state
    )
    kmeans.fit(customer_data_pca)

    # Relabel clusters by descending size so "Cluster 0" is always the
    # largest segment, regardless of KMeans' arbitrary internal label order.
    frequencies = Counter(kmeans.labels_)
    order = [label for label, _ in frequencies.most_common()]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(order)}
    new_labels = np.array([label_mapping[label] for label in kmeans.labels_])
    return new_labels, kmeans


def evaluate_clustering(X: pd.DataFrame, labels: np.ndarray) -> dict:
    return {
        "n_observations": int(len(X)),
        "silhouette_score": float(silhouette_score(X, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
    }


def cluster_centroids_standardized(customer_data_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Standardized per-cluster feature means, for radar-chart style profiling."""
    df_customer = customer_data_cleaned.set_index("CustomerID")
    feature_cols = [c for c in df_customer.columns if c != "cluster"]

    scaler = StandardScaler()
    standardized = scaler.fit_transform(df_customer[feature_cols].astype(float))
    standardized_df = pd.DataFrame(standardized, columns=feature_cols, index=df_customer.index)
    standardized_df["cluster"] = df_customer["cluster"].values
    return standardized_df.groupby("cluster").mean()
