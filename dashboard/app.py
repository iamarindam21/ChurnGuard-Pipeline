"""
Streamlit dashboard - the business-facing view on top of the model and
API. Run: streamlit run dashboard/app.py
"""
import joblib
import pandas as pd
import streamlit as st

from etl.load import read_table
from ml.train_model import MODEL_PATH, CATEGORICAL_COLS

st.set_page_config(page_title="ChurnGuard", layout="wide")
st.title("ChurnGuard — Client Churn Intelligence Dashboard")

df = read_table("client_features")
bundle = joblib.load(MODEL_PATH)

enc = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
for c in bundle["feature_cols"]:
    if c not in enc.columns:
        enc[c] = 0
X = enc[bundle["feature_cols"]]
df["churn_probability"] = bundle["model"].predict_proba(X)[:, 1]

col1, col2, col3 = st.columns(3)
col1.metric("Total clients", len(df))
col2.metric("At-risk clients (>50%)", int((df["churn_probability"] > 0.5).sum()))
col3.metric("Revenue at risk", f"${df.loc[df['churn_probability'] > 0.5, 'monthly_revenue'].sum():,.0f}/mo")

st.subheader("At-risk clients")
threshold = st.slider("Churn probability threshold", 0.0, 1.0, 0.5)
at_risk = df[df["churn_probability"] >= threshold].sort_values("churn_probability", ascending=False)
st.dataframe(at_risk[["client_id", "plan_type", "industry", "monthly_revenue", "churn_probability"]])

st.subheader("Churn probability distribution")
st.bar_chart(df["churn_probability"].value_counts(bins=10).sort_index())

try:
    enriched = read_table("feedback_enriched")
    st.subheader("Top churn reasons (from AI-enriched feedback)")
    st.bar_chart(enriched["churn_reason_category"].value_counts())
except Exception:
    st.info("Run ai/feedback_enrichment.py first to see feedback themes.")