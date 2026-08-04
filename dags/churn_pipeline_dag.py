"""
Airflow DAG for the Client Churn Intelligence pipeline.

Schedule: daily. Each ETL step is its own task for isolated retries and
clear lineage. Model training and LLM enrichment both depend on the ETL
task finishing, but not on each other, so they run in parallel.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from etl.extract import extract_clients, extract_usage, extract_feedback, extract_labels
from etl.transform import build_client_features
from etl.load import load_client_features
from ml.train_model import train_and_evaluate
from ai.feedback_enrichment import enrich_all_feedback

default_args = {
    "owner": "arindam",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="churnguard_pipeline",
    default_args=default_args,
    description="Extract client/usage/feedback data, engineer features, "
                 "train churn model, enrich feedback with LLM, load to warehouse.",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["churn", "ml", "ai"],
) as dag:

    def _extract_transform_load(**context):
        clients = extract_clients()
        usage = extract_usage()
        feedback = extract_feedback()
        labels = extract_labels()
        features = build_client_features(clients, usage, feedback, labels)
        load_client_features(features)

    def _train_model(**context):
        train_and_evaluate()

    def _enrich_feedback(**context):
        enrich_all_feedback()

    etl_task = PythonOperator(
        task_id="extract_transform_load",
        python_callable=_extract_transform_load,
    )

    train_task = PythonOperator(
        task_id="train_churn_model",
        python_callable=_train_model,
    )

    enrich_task = PythonOperator(
        task_id="enrich_feedback_with_llm",
        python_callable=_enrich_feedback,
    )

    etl_task >> [train_task, enrich_task]