"""FastAPI app served on Cloud Run.

uvicorn churn.serve.app:app --port 8080
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from churn.features import add_derived_features
from churn.serve.model_store import load_model
from churn.serve.prediction_log import BigQuerySink, NullSink, to_rows
from churn.serve.schemas import (
    HealthResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
)

log = logging.getLogger("churn.serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

HIGH_RISK = 0.75


def _band(p: float, threshold: float) -> str:
    if p >= HIGH_RISK:
        return "high"
    return "medium" if p >= threshold else "low"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, meta = load_model()
    app.state.model = model
    app.state.meta = meta

    table = os.environ.get("PREDICTION_LOG_TABLE")
    app.state.sink = BigQuerySink(table) if table else NullSink()
    yield
    app.state.sink.close()


app = FastAPI(title="churn-service", version="0.4.0", lifespan=lifespan)


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    # Structured enough for Cloud Logging to pick up as jsonPayload fields.
    log.info(
        "request",
        extra={"path": request.url.path, "status": response.status_code, "latency_ms": elapsed_ms},
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    meta = getattr(request.app.state, "meta", None)
    if meta is None:
        return HealthResponse(status="loading")
    return HealthResponse(status="ok", model_version=meta.model_version)


@app.get("/metadata")
def metadata(request: Request) -> dict:
    meta = request.app.state.meta
    return {
        "model_version": meta.model_version,
        "trained_at": meta.trained_at,
        "decision_threshold": meta.decision_threshold,
        "metrics": meta.metrics,
        "features": meta.features,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    model = request.app.state.model
    meta = request.app.state.meta

    raw = [c.model_dump() for c in body.instances]
    df = add_derived_features(pd.DataFrame(raw))
    try:
        probs = model.predict_proba(df[meta.features])[:, 1].tolist()
    except Exception as exc:  # noqa: BLE001
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail="prediction failed") from exc

    request.app.state.sink.write(to_rows(raw, probs, meta.model_version))

    t = meta.decision_threshold
    preds = [
        Prediction(
            customer_id=c.customer_id,
            churn_probability=round(p, 5),
            will_churn=p >= t,
            risk_band=_band(p, t),
        )
        for c, p in zip(body.instances, probs, strict=True)
    ]
    return PredictResponse(model_version=meta.model_version, predictions=preds)
