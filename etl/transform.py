"""
Transform layer - cleans raw data and builds the model-ready feature table.
This is where data quality checks and feature engineering live.
"""
import pandas as pd


def _data_quality_checks(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Drops exact duplicate rows and asserts no null client_id.
    In production this would raise an alert and fail the pipeline task
    instead of silently dropping rows - kept simple here for the demo."""
    before = len(df)
    df = df.drop_duplicates()
    assert df["client_id"].notna().all(), f"{name}: found null client_id"
    dropped = before - len(df)
    if dropped:
        print(f"[DQ] {name}: dropped {dropped} duplicate rows")
    return df


def build_client_features(clients: pd.DataFrame, usage: pd.DataFrame,
                           feedback: pd.DataFrame, labels: pd.DataFrame,
                           as_of: pd.Timestamp = None) -> pd.DataFrame:
    clients = _data_quality_checks(clients, "clients")
    usage = _data_quality_checks(usage, "usage")
    feedback = _data_quality_checks(feedback, "feedback")

    if as_of is None:
        as_of = usage["month"].max()

    # usage aggregation - averages plus trend (early vs late window)
    usage_sorted = usage.sort_values(["client_id", "month"])
    agg = usage_sorted.groupby("client_id").agg(
        avg_logins=("logins", "mean"),
        avg_feature_score=("feature_usage_score", "mean"),
        total_support_tickets=("support_tickets", "sum"),
    ).reset_index()

    def trend(group, col):
        half = len(group) // 2
        if half == 0:
            return 0.0
        early = group[col].iloc[:half].mean()
        late = group[col].iloc[half:].mean()
        if early == 0:
            return 0.0
        return (late - early) / early

    trends = usage_sorted.groupby("client_id").apply(
        lambda g: pd.Series({
            "login_trend": trend(g, "logins"),
            "feature_score_trend": trend(g, "feature_usage_score"),
        }), include_groups=False
    ).reset_index()

    usage_features = agg.merge(trends, on="client_id")

    # tenure feature
    clients_features = clients.copy()
    clients_features["tenure_days"] = (as_of - clients_features["signup_date"]).dt.days

    # feedback volume + a simple keyword-based negative ratio -
    # placeholder until the LLM enrichment step replaces this with
    # real sentiment classification
    negative_kw = ["slow", "bug", "competitor", "complex", "high", "missing", "frustrat"]
    feedback["is_negative_kw"] = feedback["feedback_text"].str.lower().apply(
        lambda t: any(k in t for k in negative_kw)
    )
    feedback_agg = feedback.groupby("client_id").agg(
        feedback_count=("feedback_text", "count"),
        negative_feedback_ratio=("is_negative_kw", "mean"),
    ).reset_index()

    df = clients_features.merge(usage_features, on="client_id", how="left")
    df = df.merge(feedback_agg, on="client_id", how="left")
    df = df.merge(labels, on="client_id", how="left")

    df["negative_feedback_ratio"] = df["negative_feedback_ratio"].fillna(0.0)
    df["feedback_count"] = df["feedback_count"].fillna(0)

    return df