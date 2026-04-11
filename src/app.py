"""
app.py — Pulsecast Main Application
------------------------------------
Predictive platform health forecasting with:
- SMA baseline forecast + uncertainty bands
- Forward forecast with growing uncertainty range
- Hold-out validation with MAE vs naive baseline
- Anomaly detection + path-event cascade tracing
- Keystone Element identification
- Gemini AI plain-English root cause analysis
- Scenario comparison engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

from forecasting import (
    calculate_forecast_and_anomalies,
    generate_future_forecast,
    run_holdout_validation,
    apply_scenario
)
from anomaly import (
    extract_anomalies,
    trace_cascade,
    calculate_cascade_severity
)
from keystone import identify_keystone

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pulsecast | Platform Health",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Pulsecast: Predictive Platform Health")
st.caption("Transparent AI forecasting — look ahead, not backwards.")

# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'platform_health.csv')
    df   = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Run: `python scripts/generate_data.py` first.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
target_metric   = st.sidebar.selectbox(
    "Target Metric",
    ['Transactions', 'Login_Success_Rate', 'Error_Rate', 'Support_Tickets']
)
lookback_window = st.sidebar.slider("Baseline Window (Days)", 3, 14, 7)
forecast_days   = st.sidebar.slider("Forecast Horizon (Days)", 1, 42, 7)

st.sidebar.markdown("---")
st.sidebar.subheader("🔀 Scenario Testing")
scenario_pct = st.sidebar.slider("Adjust Forecast by (%)", -30, 30, 0)

# ── Core Processing ────────────────────────────────────────────────────────────
analyzed_df  = calculate_forecast_and_anomalies(df, target_metric, lookback_window)
future_df    = generate_future_forecast(df, target_metric, forecast_days)
scenario_df  = apply_scenario(future_df, scenario_pct)
anomaly_df   = extract_anomalies(analyzed_df, target_metric)

# ── Summary Metrics ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
latest  = df[target_metric].iloc[-1]
avg_7d  = df[target_metric].tail(7).mean()
n_anom  = len(anomaly_df)

col1.metric("Latest Value",       f"{latest:,.2f}")
col2.metric("7-Day Average",      f"{avg_7d:,.2f}")
col3.metric("Anomalies Detected", n_anom,
            delta="⚠️ Review needed" if n_anom > 0 else None)
col4.metric("Forecast Horizon",   f"{forecast_days} days")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HISTORICAL FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.subheader(f"📊 Historical Forecast: {target_metric}")
st.line_chart(
    analyzed_df.set_index('Date')[
        [target_metric, 'Baseline_Forecast', 'Upper_Bound', 'Lower_Bound']
    ]
)
with st.expander("📖 How is this forecast produced?"):
    st.markdown(
        f"**Method:** Simple Moving Average (SMA) with a {lookback_window}-day window.  \n"
        "**Uncertainty bands:** ±2 standard deviations, covering ~95% of expected variation.  \n"
        "**Anomaly flag:** Any value outside the bands is statistically significant — not noise.  \n"
        "**Why SMA?** Simple models are transparent, reproducible, and often outperform "
        "overly complex ones on short-horizon forecasts."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HOLD-OUT VALIDATION (Phase 3 addition)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🧪 Forecast Validation (Hold-out Test)")
st.caption(
    "The model is tested on data it has never seen to prove it adds real "
    "predictive value — not just memorising the past."
)

validation = run_holdout_validation(df, target_metric, window=lookback_window)

if 'error' in validation:
    st.warning(validation['error'])
else:
    v1, v2, v3 = st.columns(3)
    v1.metric("SMA Forecast MAE",    f"{validation['model_mae']:,.4f}")
    v2.metric("Naive Baseline MAE",  f"{validation['baseline_mae']:,.4f}")
    v3.metric("Improvement",         f"{validation['improvement']:+.1f}%")
    st.info(validation['verdict'])

    with st.expander("📖 What does MAE mean?"):
        st.markdown(
            "**Mean Absolute Error (MAE)** is the average gap between what the model "
            "predicted and what actually happened.  \n"
            f"- **SMA MAE of {validation['model_mae']:,.2f}** means on average the forecast "
            f"was off by {validation['model_mae']:,.2f} units per day.  \n"
            f"- **Naive baseline** simply repeats the last observed value. "
            "If SMA beats this, it is genuinely learning patterns — not just guessing."
        )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FORWARD FORECAST + SCENARIO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader(f"🔭 {forecast_days}-Day Forward Forecast")

tab1, tab2 = st.tabs(["📈 Baseline Forecast", "🔀 Scenario Comparison"])

with tab1:
    st.line_chart(future_df.set_index('Date')[['Central', 'Low', 'High']])
    st.caption(
        f"Central estimate: **{future_df['Central'].iloc[0]:,.2f}** | "
        f"Range: {future_df['Low'].iloc[0]:,.2f} – {future_df['High'].iloc[-1]:,.2f} "
        f"(uncertainty grows with forecast horizon)"
    )

with tab2:
    comparison = pd.DataFrame({
        'Baseline':       future_df['Central'].values,
        'Scenario Low':   scenario_df['Low'].values,
        'Scenario':       scenario_df['Central'].values,
        'Scenario High':  scenario_df['High'].values
    }, index=future_df['Date'])
    st.line_chart(comparison)
    diff = scenario_df['Central'].iloc[0] - future_df['Central'].iloc[0]
    st.caption(
        f"At **{scenario_pct:+.0f}%** adjustment — "
        f"Scenario central: **{scenario_df['Central'].iloc[0]:,.2f}** "
        f"vs baseline **{future_df['Central'].iloc[0]:,.2f}** "
        f"(difference: {diff:+.2f})"
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("⚠️ Anomaly Detection")

if not anomaly_df.empty:
    st.error(f"Detected **{len(anomaly_df)}** sudden changes outside the normal forecast range.")
    st.dataframe(anomaly_df, use_container_width=True)
else:
    st.success("Platform health is stable. All metrics are within predicted forecast bands.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CASCADE TRACE (Phase 3 addition)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🔗 Path-Event Cascade Analysis")
st.caption(
    "When an anomaly is detected, this traces how the fault in one metric "
    "propagates to affect all other platform metrics — revealing cascade chains."
)

cascade_df = trace_cascade(df, target_metric)
severity   = calculate_cascade_severity(cascade_df)

sev1, sev2 = st.columns([1, 3])
sev1.metric("Cascade Severity", severity['label'].split(' ', 1)[-1],
            delta=f"Score: {severity['score']}/100")
sev2.info(severity['recommendation'])

if not cascade_df.empty:
    st.dataframe(cascade_df, use_container_width=True)
    with st.expander("📖 How does cascade tracing work?"):
        st.markdown(
            "For each anomaly window, Pulsecast computes the **Pearson correlation** "
            "between the trigger metric and every other metric.  \n"
            "- **Primary Cascade Effect (|r| ≥ 0.7):** The other metric moves strongly "
            "with the trigger — a direct fault propagation.  \n"
            "- **Secondary Effect (|r| ≥ 0.4):** Moderate co-movement — likely an "
            "indirect downstream effect.  \n"
            "- **Unaffected (|r| < 0.4):** The other metric is behaving independently.  \n\n"
            "Lag Days indicates whether the effect appears 1 day after the trigger, "
            "suggesting a cause-and-effect delay in the system."
        )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — KEYSTONE ELEMENT (Phase 3 addition)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🔑 Keystone Element Analysis")
st.caption(
    "Identifies the single metric that acts as the critical load-bearing "
    "driver of overall platform health — your highest-priority monitoring target."
)

keystone_result = identify_keystone(df)

if 'error' in keystone_result:
    st.warning(keystone_result['error'])
else:
    k1, k2, k3 = st.columns(3)
    k1.metric("Keystone Metric",  keystone_result['keystone_metric'])
    k2.metric("Load Factor",      f"{keystone_result['load_factor']:.1f}%")
    k3.metric("Correlation (r)",  f"{keystone_result['correlation']:.3f}")

    st.markdown(keystone_result['interpretation'])
    st.dataframe(keystone_result['all_correlations'], use_container_width=True)

    # Platform Health Score Chart
    health_df = pd.DataFrame({
        'Date':           df['Date'],
        'Health Score':   keystone_result['health_scores']
    }).set_index('Date')

    st.line_chart(health_df)
    st.caption(
        "Platform Health Score (0–100): composite of all four metrics weighted by "
        "business impact. Scores below 40 indicate an unhealthy platform state."
    )

    with st.expander("📖 How is the Keystone Element calculated?"):
        st.markdown(
            "Each metric is normalised to a z-score and weighted by business impact "
            "(Transactions: 35%, Login Success: 30%, Error Rate: -20%, Tickets: -15%).  \n"
            "A composite **Platform Health Score** is computed daily.  \n"
            "The metric with the highest **Pearson correlation** against this score "
            "is the Keystone — it explains the most variance in overall health.  \n"
            "The **Load Factor** is R² × 100: the percentage of system variance "
            "explained by that single metric alone."
        )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — GEMINI AI ROOT CAUSE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🤖 AI Root Cause Analysis")

def get_ai_explanation(metric, anomaly_rows, forecast_df, cascade_df, keystone_result):
    if not gemini_key:
        return "⚠️ Gemini API key not found. Add GEMINI_API_KEY to your .env file."

    anomaly_summary  = anomaly_rows[['Date', metric, 'Baseline_Forecast']].to_string(index=False)
    forecast_summary = (
        f"Next {forecast_days}-day forecast — "
        f"Central: {forecast_df['Central'].iloc[0]:.2f}, "
        f"Low: {forecast_df['Low'].iloc[0]:.2f}, "
        f"High: {forecast_df['High'].iloc[-1]:.2f}"
    )
    keystone_name = keystone_result.get('keystone_metric', 'unknown')
    cascade_summary = cascade_df.to_string(index=False) if not cascade_df.empty else "No cascade detected."

    prompt = f"""
You are a platform reliability analyst writing for a non-technical business manager.
Analyze this anomaly in the '{metric}' metric and write a plain-English summary.

Anomalies detected:
{anomaly_summary}

Cascade analysis:
{cascade_summary}

Keystone metric (critical driver): {keystone_name}

{forecast_summary}

Your response must:
1. State what happened in plain English (no jargon).
2. Explain whether this is an isolated fault or a cascade, based on the cascade data.
3. Name the keystone metric and why it matters right now.
4. Recommend one immediate next step.

Keep the entire response under 120 words.
"""
    try:
        model    = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

if not anomaly_df.empty:
    if st.button("🔍 Analyse with Gemini AI"):
        with st.spinner("Gemini is analysing the anomaly pattern..."):
            explanation = get_ai_explanation(
                target_metric, anomaly_df, future_df, cascade_df, keystone_result
            )
        st.info(explanation)
        with st.expander("📖 How was this explanation generated?"):
            st.markdown(
                "Anomaly rows, cascade correlation data, and the keystone metric were "
                "packaged into a structured prompt and sent to Google Gemini. "
                "The model was instructed to respond in plain English under 120 words, "
                "covering what happened, whether it cascaded, and what to do next. "
                "No raw model weights were used — this is a statistical analysis "
                "layer with an AI explanation layer on top."
            )
else:
    st.info("No anomalies detected. Adjust the metric or baseline window to explore different scenarios.")