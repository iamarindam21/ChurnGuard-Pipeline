"""
Simulates 3 'source systems' a real company would have:
1. CRM export -> clients.csv (client master data)
2. Product usage logs -> usage.csv (monthly usage metrics per client)
3. Support/feedback system -> feedback.csv (free-text client feedback)
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

N_CLIENTS = 300
INDUSTRIES = ["Retail", "Healthcare", "Finance", "EdTech", "Manufacturing", "Logistics"]
PLANS = ["Basic", "Pro", "Enterprise"]
REGIONS = ["North", "South", "East", "West"]

# ---------- 1. clients.csv ----------
clients = []
for i in range(1, N_CLIENTS + 1):
    signup_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 700))
    plan = random.choices(PLANS, weights=[0.4, 0.4, 0.2])[0]
    monthly_revenue = {"Basic": 49, "Pro": 199, "Enterprise": 999}[plan] * random.uniform(0.9, 1.3)
    clients.append({
        "client_id": f"C{i:04d}",
        "signup_date": signup_date.strftime("%Y-%m-%d"),
        "plan_type": plan,
        "monthly_revenue": round(monthly_revenue, 2),
        "industry": random.choice(INDUSTRIES),
        "region": random.choice(REGIONS),
    })
clients_df = pd.DataFrame(clients)

churn_prob = np.random.rand(N_CLIENTS)
will_churn = churn_prob < 0.28

silent_churn_mask = will_churn & (np.random.rand(N_CLIENTS) < 0.35)
noisy_survivor_mask = (~will_churn) & (np.random.rand(N_CLIENTS) < 0.20)
shows_decline = (will_churn & ~silent_churn_mask) | noisy_survivor_mask

# ---------- 2. usage.csv (last 6 months per client) ----------
usage_rows = []
months = pd.date_range(end=pd.Timestamp("2026-06-30"), periods=6, freq="ME")
for idx, row in clients_df.iterrows():
    base_logins = random.randint(15, 60)
    base_feature_score = random.uniform(0.4, 0.9)
    declining = shows_decline[idx]
    for m_i, month in enumerate(months):
        decay = (1 - 0.15 * m_i) if declining else 1.0
        noise = random.uniform(0.85, 1.15)
        logins = max(0, int(base_logins * decay * noise))
        feature_score = max(0.0, min(1.0, base_feature_score * decay * noise))
        support_tickets = np.random.poisson(2.5 if declining else 0.8)
        usage_rows.append({
            "client_id": row["client_id"],
            "month": month.strftime("%Y-%m-01"),
            "logins": logins,
            "feature_usage_score": round(feature_score, 3),
            "support_tickets": int(support_tickets),
        })
usage_df = pd.DataFrame(usage_rows)

# ---------- 3. feedback.csv ----------
POSITIVE_TEMPLATES = [
    "The team loves the new dashboard, really improved our workflow.",
    "Great support response time, issue resolved quickly.",
    "Onboarding was smooth and the product fits our needs well.",
    "We're expanding usage across more departments this quarter.",
]
NEGATIVE_TEMPLATES = [
    "Pricing feels too high compared to the value we're getting.",
    "We've had repeated bugs in the reporting module, frustrating.",
    "Support has been slow to respond to our last three tickets.",
    "We're evaluating competitor tools because key features are missing.",
    "The platform is too complex for our team to adopt fully.",
]
feedback_rows = []
for idx, row in clients_df.iterrows():
    n_feedback = random.randint(1, 3)
    leans_negative = will_churn[idx] if random.random() < 0.75 else (not will_churn[idx])
    for _ in range(n_feedback):
        text = random.choice(NEGATIVE_TEMPLATES if leans_negative else POSITIVE_TEMPLATES)
        date = datetime(2026, random.randint(1, 6), random.randint(1, 28))
        feedback_rows.append({
            "client_id": row["client_id"],
            "date": date.strftime("%Y-%m-%d"),
            "feedback_text": text,
        })
feedback_df = pd.DataFrame(feedback_rows)

labels_df = pd.DataFrame({
    "client_id": clients_df["client_id"],
    "churned": will_churn.astype(int),
})

clients_df.to_csv("data/clients.csv", index=False)
usage_df.to_csv("data/usage.csv", index=False)
feedback_df.to_csv("data/feedback.csv", index=False)
labels_df.to_csv("data/churn_labels.csv", index=False)

print(f"Generated: {len(clients_df)} clients, {len(usage_df)} usage rows, "
      f"{len(feedback_df)} feedback rows, churn rate={will_churn.mean():.1%}")
      