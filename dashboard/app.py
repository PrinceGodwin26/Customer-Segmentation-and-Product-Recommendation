"""Streamlit dashboard for the customer segmentation & recommendation API.

Run with:
    streamlit run dashboard/app.py

Configure the API location via the sidebar or the API_BASE_URL env var
(defaults to http://localhost:8000).
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Customer Segmentation & Recommendations", layout="wide")

CLUSTER_COLORS = {0: "#e8000b", 1: "#1ac938", 2: "#023eff"}
DEFAULT_COLOR = "#ff6200"


def cluster_color(cluster_id) -> str:
    return CLUSTER_COLORS.get(int(cluster_id), DEFAULT_COLOR)


@st.cache_data(ttl=10, show_spinner=False)
def api_get(base_url: str, path: str, params: dict | None = None):
    resp = requests.get(f"{base_url}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(base_url: str, path: str, json: dict | None = None):
    resp = requests.post(f"{base_url}{path}", json=json or {}, timeout=120)
    resp.raise_for_status()
    return resp.json()


# --- Sidebar ---------------------------------------------------------------
st.sidebar.title("Pipeline Control")
default_base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
base_url = st.sidebar.text_input("API base URL", value=default_base_url).rstrip("/")

try:
    status = api_get(base_url, "/status")
except requests.exceptions.RequestException as exc:
    st.sidebar.error(f"Cannot reach API at {base_url}\n\n{exc}")
    st.stop()

st.sidebar.write("**Ready:**", status["ready"])
st.sidebar.write("**Data path:**", status["data_path"])
st.sidebar.write("**Raw rows:**", status["raw_row_count"])
st.sidebar.write("**Last run:**", status["last_run_at"])
if status.get("error"):
    st.sidebar.warning(status["error"])

data_path_override = st.sidebar.text_input(
    "Data path override (optional)",
    value="",
    help="Leave blank to use the server's configured DATA_PATH (the placeholder file until you swap in the real dataset).",
)

if st.sidebar.button("Run / Refresh pipeline", type="primary"):
    with st.spinner("Running pipeline..."):
        try:
            run_result = api_post(
                base_url, "/pipeline/run", {"data_path": data_path_override or None}
            )
            st.cache_data.clear()
            st.sidebar.success(
                f"Done: {run_result['n_customers']} customers, {run_result['n_outliers']} outliers excluded."
            )
        except requests.exceptions.RequestException as exc:
            st.sidebar.error(f"Pipeline run failed: {exc}")

if not status["ready"]:
    st.warning("Pipeline has not produced any results yet. Click **Run / Refresh pipeline** in the sidebar.")
    st.stop()

st.title("Customer Segmentation & Recommendation Dashboard")

tab_overview, tab_clusters, tab_profiles, tab_recs = st.tabs(
    ["Data Overview", "Cluster Visualization", "Cluster Profiles", "Recommendations"]
)

# --- Data Overview -----------------------------------------------------
with tab_overview:
    quality = api_get(base_url, "/data/quality")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw rows", quality["raw_row_count"])
    col2.metric("Rows after cleaning", quality["final_rows"])
    col3.metric("Duplicate rows removed", quality["duplicate_rows"])
    col4.metric("Cancelled transactions", f"{quality['cancelled_pct']}%")

    st.subheader("Data quality issues found during cleaning")
    quality_rows = [
        {"Issue": "Cancelled transactions", "Percentage": quality["cancelled_pct"]},
        {"Issue": "Anomalous stock codes", "Percentage": quality["anomalous_stockcode_pct"]},
        {"Issue": "Service-related descriptions", "Percentage": quality["service_related_pct"]},
    ]
    for col, pct in quality["missing_value_pct"].items():
        quality_rows.append({"Issue": f"Missing {col}", "Percentage": pct})
    quality_df = pd.DataFrame(quality_rows).sort_values("Percentage", ascending=True)

    fig = px.bar(quality_df, x="Percentage", y="Issue", orientation="h", color_discrete_sequence=[DEFAULT_COLOR])
    fig.update_layout(xaxis_title="Percentage of rows (%)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# --- Cluster Visualization -----------------------------------------------
with tab_clusters:
    metrics = api_get(base_url, "/metrics")
    summary = api_get(base_url, "/clusters/summary")
    pca_records = api_get(base_url, "/clusters/pca")
    pca_df = pd.DataFrame(pca_records)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers (post outlier removal)", metrics["n_observations"])
    col2.metric("Silhouette score", round(metrics["silhouette_score"], 3))
    col3.metric("Calinski-Harabasz score", round(metrics["calinski_harabasz_score"], 1))
    col4.metric("Davies-Bouldin score", round(metrics["davies_bouldin_score"], 3))

    st.subheader("Cluster distribution")
    dist_df = pd.DataFrame(summary)[["cluster", "count", "percentage"]].sort_values("cluster")
    dist_df["cluster_label"] = "Cluster " + dist_df["cluster"].astype(str)
    fig_dist = px.bar(
        dist_df,
        x="percentage",
        y="cluster_label",
        orientation="h",
        text="percentage",
        color="cluster_label",
        color_discrete_map={f"Cluster {c}": cluster_color(c) for c in dist_df["cluster"]},
    )
    fig_dist.update_traces(texttemplate="%{text:.2f}%")
    fig_dist.update_layout(xaxis_title="Percentage of customers (%)", yaxis_title="Cluster", showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("3D PCA visualization of customer clusters")
    if {"PC1", "PC2", "PC3"}.issubset(pca_df.columns):
        fig_3d = go.Figure()
        for cluster_id in sorted(pca_df["cluster"].unique()):
            subset = pca_df[pca_df["cluster"] == cluster_id]
            fig_3d.add_trace(
                go.Scatter3d(
                    x=subset["PC1"],
                    y=subset["PC2"],
                    z=subset["PC3"],
                    mode="markers",
                    marker=dict(color=cluster_color(cluster_id), size=5, opacity=0.6),
                    name=f"Cluster {cluster_id}",
                )
            )
        fig_3d.update_layout(
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            height=700,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("Not enough principal components to render a 3D plot.")

# --- Cluster Profiles ------------------------------------------------------
with tab_profiles:
    summary = api_get(base_url, "/clusters/summary")
    features_df = pd.DataFrame(api_get(base_url, "/clusters/features"))

    st.subheader("Radar chart: standardized feature means per cluster")
    centroid_features = list(summary[0]["centroid"].keys())
    fig_radar = go.Figure()
    for row in sorted(summary, key=lambda r: r["cluster"]):
        values = [row["centroid"][f] for f in centroid_features]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=centroid_features + [centroid_features[0]],
                fill="toself",
                name=f"Cluster {row['cluster']}",
                line_color=cluster_color(row["cluster"]),
                opacity=0.6,
            )
        )
    fig_radar.update_layout(height=650, showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Feature distributions by cluster")
    feature_cols = [c for c in features_df.columns if c not in ("CustomerID", "cluster")]
    selected_feature = st.selectbox("Feature", feature_cols)
    features_df["cluster_label"] = "Cluster " + features_df["cluster"].astype(str)
    fig_hist = px.histogram(
        features_df,
        x=selected_feature,
        color="cluster_label",
        barmode="overlay",
        opacity=0.6,
        color_discrete_map={f"Cluster {c}": cluster_color(c) for c in features_df["cluster"].unique()},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Top-selling products per cluster")
    cluster_pick = st.selectbox(
        "Cluster", sorted(features_df["cluster"].unique()), key="top_products_cluster"
    )
    top_products = api_get(base_url, "/clusters/top-products", {"cluster": int(cluster_pick)})
    st.dataframe(pd.DataFrame(top_products), use_container_width=True, hide_index=True)

# --- Recommendations ---------------------------------------------------
with tab_recs:
    st.subheader("Look up a customer's recommendations")
    customer_id_input = st.text_input("Customer ID", placeholder="e.g. 17850")
    if customer_id_input:
        try:
            rec = api_get(base_url, f"/recommendations/{float(customer_id_input)}")
            st.json(rec)
        except requests.exceptions.HTTPError:
            st.error("No recommendations found for that customer ID.")
        except ValueError:
            st.error("Customer ID must be numeric.")

    st.subheader("Browse all recommendations")
    cluster_options = ["All"] + sorted(features_df["cluster"].unique().tolist())
    cluster_filter = st.selectbox("Filter by cluster", cluster_options, key="recs_cluster_filter")
    params = {"limit": 200}
    if cluster_filter != "All":
        params["cluster"] = int(cluster_filter)
    recs_page = api_get(base_url, "/recommendations", params)
    st.caption(f"Showing {len(recs_page['items'])} of {recs_page['total']} customers")
    st.dataframe(pd.DataFrame(recs_page["items"]), use_container_width=True, hide_index=True)
