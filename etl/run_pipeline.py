"""
Wires extract -> transform -> load together.
This is the same sequence the Airflow DAG will call task-by-task later -
the DAG just adds scheduling, retries, and dependency management on top
of these same functions.
"""
from etl.extract import extract_clients, extract_usage, extract_feedback, extract_labels
from etl.transform import build_client_features
from etl.load import load_client_features


def run():
    clients = extract_clients()
    usage = extract_usage()
    feedback = extract_feedback()
    labels = extract_labels()

    features = build_client_features(clients, usage, feedback, labels)
    load_client_features(features)
    return features


if __name__ == "__main__":
    df = run()
    print(df[["client_id", "plan_type", "tenure_days", "avg_logins",
              "login_trend", "negative_feedback_ratio", "churned"]].head(10))