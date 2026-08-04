# ChurnGuard — Client Churn & Feedback Intelligence Pipeline

End-to-end data engineering + AI pipeline that predicts client churn and
explains why, using Airflow, scikit-learn, and LLM-based feedback analysis.

## Business Problem

A B2B SaaS company wants to know, ahead of time, which clients are likely
to churn — and why — so account managers can intervene before it's too
late.

## Architecture
[clients.csv] [usage.csv] [feedback.csv] <- source systems
\ | /
v v v
Airflow DAG (daily orchestration)
extract -> transform -> load (SQLite)
|
-----------+-----------
v v
ML: Random Forest AI: Gemini LLM
churn classifier feedback enrichment
\ /
-----------+---------
v
FastAPI + Streamlit

## Tech Stack

- **Orchestration:** Apache Airflow (via Docker)
- **Data processing:** pandas
- **ML:** scikit-learn (Random Forest)
- **AI/LLM:** Google Gemini API
- **Warehouse:** SQLite
- **API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit
- **Containerization:** Docker / Docker Compose

## Results

- Churn model: 85.3% accuracy, 0.846 ROC-AUC, 0.778 precision, 0.667 recall
- Top churn drivers: negative feedback ratio, support ticket volume,
  declining feature usage trend
- Dashboard identifies 87 at-risk clients out of 300, representing
  approximately $28,000/month in at-risk revenue

## How to Run

1. Clone the repo and set up a virtual environment:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2. Generate data and run the pipeline:

python data/generate_data.py
python -m etl.run_pipeline
python -m ml.train_model
python -m ai.feedback_enrichment

3. Run the API:

uvicorn api.main:app --reload

4. Run the dashboard:

python -m streamlit run dashboard/app.py

5. (Optional) Run the Airflow DAG via Docker Compose - see the `dags/` folder.

## Notes

- Data is synthetic, modeled on realistic SaaS churn patterns (28%
  base churn rate, with deliberate noise so it isn't perfectly
  separable).
- LLM enrichment falls back to a rule-based mock when no
  GEMINI_API_KEY is set, so the pipeline runs end to end without
  requiring an API key.

  