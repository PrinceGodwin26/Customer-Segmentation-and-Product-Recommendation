"""FastAPI service exposing the customer segmentation & recommendation pipeline.

Run with:
    uvicorn api.main:app --reload --port 8000

Swap in the real dataset by setting DATA_PATH before starting the server, or
by pointing POST /pipeline/run at a different path at runtime, e.g.:
    export DATA_PATH=/path/to/real_data.csv
"""

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import config
from app.service import PipelineNotReadyError, ensure_ready, get_state, run_pipeline
from app.utils import df_to_records

app = FastAPI(
    title="Customer Segmentation & Recommendation API",
    version="1.0.0",
    description="Runs the RFM/behavioral clustering pipeline and serves cluster & recommendation data.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    data_path: Optional[str] = None


@app.on_event("startup")
def _run_on_startup() -> None:
    try:
        run_pipeline()
    except Exception as exc:  # noqa: BLE001 - surfaced via /status instead of crashing the app
        get_state().error = str(exc)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    state = get_state()
    return {
        "ready": state.ready,
        "last_run_at": state.last_run_at,
        "data_path": state.data_path,
        "raw_row_count": state.raw_row_count,
        "error": state.error,
    }


@app.post("/pipeline/run")
def run(request: RunRequest):
    try:
        state = run_pipeline(request.data_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Data file not found: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        get_state().error = str(exc)
        raise HTTPException(status_code=500, detail=f"Pipeline run failed: {exc}") from exc

    return {
        "ready": state.ready,
        "last_run_at": state.last_run_at,
        "data_path": state.data_path,
        "raw_row_count": state.raw_row_count,
        "n_customers": len(state.customer_data_cleaned),
        "n_outliers": len(state.outliers_data),
        "metrics": state.metrics,
    }


@app.get("/data/quality")
def data_quality():
    state = ensure_ready_or_404()
    return {
        "raw_row_count": state.raw_row_count,
        **state.data_quality,
    }


@app.get("/metrics")
def metrics():
    state = ensure_ready_or_404()
    return state.metrics


@app.get("/clusters/summary")
def clusters_summary():
    state = ensure_ready_or_404()
    return state.cluster_summary


@app.get("/clusters/pca")
def clusters_pca():
    state = ensure_ready_or_404()
    df = state.customer_data_pca.reset_index().rename(columns={"index": "CustomerID"})
    return df_to_records(df)


@app.get("/clusters/features")
def clusters_features():
    state = ensure_ready_or_404()
    return df_to_records(state.customer_data_cleaned)


@app.get("/clusters/top-products")
def clusters_top_products(cluster: Optional[int] = None):
    state = ensure_ready_or_404()
    df = state.top_products_per_cluster
    if cluster is not None:
        df = df[df["cluster"] == cluster]
    return df_to_records(df)


@app.get("/recommendations")
def recommendations(
    cluster: Optional[int] = None,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    state = ensure_ready_or_404()
    df = state.recommendations_df
    if cluster is not None:
        df = df[df["cluster"] == cluster]
    total = len(df)
    page = df.iloc[offset : offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "items": df_to_records(page)}


@app.get("/recommendations/{customer_id}")
def recommendation_for_customer(customer_id: float):
    state = ensure_ready_or_404()
    df = state.recommendations_df
    match = df[df["CustomerID"] == customer_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"No recommendations for customer {customer_id}")
    return df_to_records(match)[0]


def ensure_ready_or_404():
    try:
        return ensure_ready()
    except PipelineNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
