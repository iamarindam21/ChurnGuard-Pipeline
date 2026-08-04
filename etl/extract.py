"""
Extract layer - reads raw data from source systems.

Currently reading from local CSVs (simulated source systems).
In production this would hit a Postgres replica, a CRM API, or a
support ticket API - the function signatures stay the same either way,
only the implementation inside changes.
"""
import pandas as pd


def extract_clients() -> pd.DataFrame:
    return pd.read_csv("data/clients.csv", parse_dates=["signup_date"])


def extract_usage() -> pd.DataFrame:
    return pd.read_csv("data/usage.csv", parse_dates=["month"])


def extract_feedback() -> pd.DataFrame:
    return pd.read_csv("data/feedback.csv", parse_dates=["date"])


def extract_labels() -> pd.DataFrame:
    return pd.read_csv("data/churn_labels.csv")