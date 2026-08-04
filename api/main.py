"""
FastAPI serving layer.
Exposes:
  GET /clients/{client_id}/churn-risk  -> probability + top churn reason
  GET /at-risk?threshold=0.5            -> ranked list of at-risk clients

Run: uvicorn api.main:app --reload
"""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from etl.load import read_table
from ml.train_model import MODEL_PATH, CATEGORICAL_COLS

app = FastAPI(title="Client Churn Intelligence API")

_bundle = None


def get_model_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    bundle = get_model_bundle()
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    for col in bundle["feature_cols"]:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    return df_encoded[bundle["feature_cols"]]


@app.get("/clients/{client_id}/churn-risk")
def get_churn_risk(client_id: str):
    features_df = read_table("client_features")
    row = features_df[features_df["client_id"] == client_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="client_id not found")

    bundle = get_model_bundle()
    X = _prepare_features(row)
    proba = float(bundle["model"].predict_proba(X)[0, 1])

    try:
        enriched = read_table("feedback_enriched")
        client_fb = enriched[enriched["client_id"] == client_id]
        top_reason = (
            client_fb["churn_reason_category"].value_counts().idxmax()
            if not client_fb.empty else "unknown"
        )
    except Exception:
        top_reason = "unknown"

    return {
        "client_id": client_id,
        "churn_probability": round(proba, 3),
        "risk_level": "high" if proba > 0.6 else "medium" if proba > 0.3 else "low",
        "top_feedback_reason": top_reason,
    }


@app.get("/at-risk")
def get_at_risk_clients(threshold: float = 0.5, limit: int = 20):
    features_df = read_table("client_features")
    bundle = get_model_bundle()
    X = _prepare_features(features_df)
    probas = bundle["model"].predict_proba(X)[:, 1]
    features_df["churn_probability"] = probas

    at_risk = features_df[features_df["churn_probability"] >= threshold] \
        .sort_values("churn_probability", ascending=False) \
        .head(limit)

    return at_risk[["client_id", "plan_type", "monthly_revenue", "churn_probability"]] \
        .to_dict(orient="records")


@app.get("/health")
def health():
    return {"status": "ok"}