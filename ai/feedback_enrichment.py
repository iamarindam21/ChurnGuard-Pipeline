"""
AI enrichment layer - reads raw client feedback text and uses an LLM
(Gemini free tier) to extract:
  - sentiment: positive / neutral / negative
  - churn_reason_category: pricing / support / missing_features /
    competitor / product_complexity / none

An LLM is used here instead of a classic sentiment library because the
categories needed are business-specific (a generic sentiment library
only gives positive/neutral/negative, not "pricing" vs "competitor").
Few-shot prompting means no labeled training data is needed for this
part - unlike the churn model, where labels existed and a trained
classifier made more sense.

Set GEMINI_API_KEY as an environment variable to use the real API.
Without a key, this falls back to a rule-based mock so the pipeline
still runs end-to-end for local development/testing.
"""
import os
import json
import sqlite3
import pandas as pd

from etl.extract import extract_feedback

DB_PATH = "data/warehouse.db"

PROMPT_TEMPLATE = """You are analyzing customer feedback for a B2B SaaS company.
Classify the feedback below.

Feedback: "{text}"

Respond ONLY with JSON in this exact format, no other text:
{{"sentiment": "positive|neutral|negative", "churn_reason_category": "pricing|support|missing_features|competitor|product_complexity|none"}}
"""


def call_gemini(text: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(PROMPT_TEMPLATE.format(text=text))
    raw = response.text.strip().strip("```json").strip("```")
    return json.loads(raw)


def rule_based_fallback(text: str) -> dict:
    """Offline mock with the same output schema as the real LLM call,
    used when no API key is set."""
    t = text.lower()
    if any(k in t for k in ["pricing", "high"]):
        return {"sentiment": "negative", "churn_reason_category": "pricing"}
    if any(k in t for k in ["support", "slow"]):
        return {"sentiment": "negative", "churn_reason_category": "support"}
    if any(k in t for k in ["competitor", "missing"]):
        return {"sentiment": "negative", "churn_reason_category": "missing_features"}
    if any(k in t for k in ["complex", "bug"]):
        return {"sentiment": "negative", "churn_reason_category": "product_complexity"}
    return {"sentiment": "positive", "churn_reason_category": "none"}


def enrich_all_feedback():
    feedback = extract_feedback()
    use_real_api = "GEMINI_API_KEY" in os.environ

    results = []
    for _, row in feedback.iterrows():
        try:
            enrichment = call_gemini(row["feedback_text"]) if use_real_api \
                else rule_based_fallback(row["feedback_text"])
        except Exception as e:
            print(f"[WARN] LLM call failed ({e}), falling back to rule-based")
            enrichment = rule_based_fallback(row["feedback_text"])

        results.append({
            "client_id": row["client_id"],
            "date": str(row["date"].date()),
            "feedback_text": row["feedback_text"],
            "sentiment": enrichment["sentiment"],
            "churn_reason_category": enrichment["churn_reason_category"],
        })

    enriched_df = pd.DataFrame(results)
    conn = sqlite3.connect(DB_PATH)
    try:
        enriched_df.to_sql("feedback_enriched", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    print(f"[AI] enriched {len(enriched_df)} feedback rows "
          f"({'real Gemini API' if use_real_api else 'offline rule-based fallback'})")
    return enriched_df


if __name__ == "__main__":
    df = enrich_all_feedback()
    print(df["churn_reason_category"].value_counts())