"""
Trains a churn classifier on the feature table produced by the ETL layer.

Model choice: RandomForestClassifier.
- Handles mixed numeric + categorical (one-hot) features without scaling.
- Gives feature_importances_ for free, which answers "why is this client
  at risk", not just "is this client at risk".
- Robust on a modest dataset (300 clients) without heavy tuning.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from etl.load import read_table

MODEL_PATH = "ml/churn_model.joblib"

FEATURE_COLS = [
    "monthly_revenue", "tenure_days", "avg_logins", "avg_feature_score",
    "total_support_tickets", "login_trend", "feature_score_trend",
    "feedback_count", "negative_feedback_ratio",
]
CATEGORICAL_COLS = ["plan_type", "industry", "region"]


def train_and_evaluate():
    df = read_table("client_features")
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    dummy_cols = [c for c in df.columns if any(c.startswith(cat + "_") for cat in CATEGORICAL_COLS)]
    X = df[FEATURE_COLS + dummy_cols]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # class_weight="balanced" matters here since churn is a minority
    # class (~28%) - without it the model could just predict "not
    # churned" for everyone and still look accurate.
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1": round(f1_score(y_test, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
    }

    joblib.dump({"model": model, "feature_cols": X.columns.tolist()}, MODEL_PATH)

    importances = sorted(
        zip(X.columns, model.feature_importances_), key=lambda x: -x[1]
    )[:5]

    print("Metrics:", metrics)
    print("Top 5 features driving churn:")
    for feat, imp in importances:
        print(f"  {feat}: {imp:.3f}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()