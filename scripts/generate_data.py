"""
generate_data.py
Generates a 90-day synthetic platform health dataset with a planted
cascading fault anomaly at Day 75 (login failure → error spike → ticket surge).
Run once before starting the app: python scripts/generate_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_platform_health_data():
    np.random.seed(42)
    dates = [datetime.today() - timedelta(days=x) for x in range(90, 0, -1)]

    # Baseline: normal platform operations
    transactions    = np.random.normal(15000, 1000, 90)
    login_success   = np.random.normal(98.5, 0.5, 90)
    error_rate      = np.random.normal(1.2, 0.2, 90)
    support_tickets = np.random.normal(150, 20, 90)

    # Inject cascading fault at Day 75 (index 74-76)
    # This mirrors the hackathon brief's anomaly example exactly
    transactions[74:77]    *= 0.70        # 30% transaction drop
    login_success[74:77]   -= 15.0        # Severe login failure
    error_rate[74:77]      += 4.50        # Massive error spike
    support_tickets[74:77] *= 3.00        # Support ticket surge

    df = pd.DataFrame({
        'Date':               dates,
        'Transactions':       transactions.astype(int),
        'Login_Success_Rate': np.clip(login_success, 0, 100).round(2),
        'Error_Rate':         error_rate.round(2),
        'Support_Tickets':    support_tickets.astype(int)
    })

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/platform_health.csv', index=False)
    print("✅ Dataset generated: data/platform_health.csv")
    print(f"   Rows: {len(df)} | Columns: {list(df.columns)}")

if __name__ == "__main__":
    generate_platform_health_data()