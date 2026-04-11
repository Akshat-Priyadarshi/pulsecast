"""
app.py — Pulsecast Core Application
Predictive platform health forecasting with anomaly detection
and AI-powered plain-English root cause analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ── Environment Setup ──────────────────────────────────────────────────────────
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)

# ── Page Configuration ─────────────────────────────────────────────────────────
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
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# ── Forecasting Engine ─────────────────────────────────────────────────────────
def calculate_forecast_and_anomalies(df, metric, window=7):
    """
    Baseline forecast: Simple Moving Average (SMA).
    Uncertainty bands: ±2 standard deviations (95% confidence).
    Anomaly flag: any point outside the bands.
    """
    df = df.copy()
    df['Baseline_Forecast'] = df[metric].rolling(window=window).mean()
    rolling_std             = df[metric].rolling(window=window).std()
    df['Upper_Bound']       = df['Baseline_Forecast'] + (2 * rolling_std)
    df['Lower_Bound']       = df['Baseline_Forecast'] - (2 * rolling_std)
    df['Is_Anomaly']        = (
        (df[metric] > df['Upper_Bound']) |
        (df[metric] < df['Lower_Bound'])
    )
    return df

# ── Future Forecast (next N days) ─────────────────────────────────────────────
def generate_future_forecast(df, metric, forecast_days=7):
    """
    Extends the SMA forecast into future periods.
    Returns central estimate + low/high uncertainty range.
    """
    recent         = df[metric].tail(7)
    central        = recent.mean()
    std            = recent.std()
    future_dates   = pd.date_range(df['Date'].max(), periods=forecast_days + 1, freq='D')[1:]
    
    return pd.DataFrame({
        'Date':    future_dates,
        'Central': central,
        'Low':     central - (1.5 * std),
        'High':    central + (1.5 * std)
    })

# ── Gemini Root Cause Analysis ─────────────────────────────────────────────────
def get_ai_explanation(metric, anomaly_rows, forecast_df):
    """
    Sends anomaly context to Gemini and returns a plain-English explanation
    short enough for a non-technical manager to act on immediately.
    """
    if not gemini_key:
        return "⚠️ Gemini API key not found. Add GEMINI_API_KEY to your .env file."

    anomaly_summary = anomaly_rows[['Date', metric, 'Baseline_Forecast']].to_string(index=False)
    forecast_summary = (
        f"Next 7-day forecast — "
        f"Central: {forecast_df['Central'].iloc[0]:.2f}, "
        f"Low: {forecast_df['Low'].iloc[0]:.2f}, "
        f"High: {forecast_df['High'].iloc[0]:.2f}"
    )

    prompt = f"""
You are a platform reliability analyst writing for a non-technical business manager.
Analyze this anomaly in the '{metric}' metric and write a 3-sentence plain-English summary.

Anomalies detected:
{anomaly_summary}

{forecast_summary}

Your response must:
1. State what happened in plain English (no jargon).
2. Suggest one likely root cause based on the metric name.
3. Recommend one immediate next step the team should take.

Keep the entire response under 80 words.
"""
    try:
        model    = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

# ── Scenario Forecasting ───────────────────────────────────────────────────────
def apply_scenario(forecast_df, adjustment_pct):
    """Applies a percentage adjustment to the central forecast for scenario testing."""
    scenario = forecast_df.copy()
    factor = 1 + (adjustment_pct / 100)
    scenario['Central'] *= factor
    scenario['Low']     *= factor
    scenario['High']    *= factor
    return scenario

# ── Load Data ──────────────────────────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Run `python scripts/generate_data.py` first.")
    st.stop()

# ── Sidebar Controls ───────────────────────────────────────────────────────────
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

# ── Process ────────────────────────────────────────────────────────────────────
analyzed_df  = calculate_forecast_and_anomalies(df, target_metric, lookback_window)
future_df    = generate_future_forecast(df, target_metric, forecast_days)
scenario_df  = apply_scenario(future_df, scenario_pct)
anomalies    = analyzed_df[analyzed_df['Is_Anomaly']].dropna()

# ── Layout: Three Metric Cards ─────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
latest = df[target_metric].iloc[-1]
avg_7d = df[target_metric].tail(7).mean()
col1.metric("Latest Value",      f"{latest:,.2f}")
col2.metric("7-Day Average",     f"{avg_7d:,.2f}")
col3.metric("Anomalies Detected", len(anomalies),
            delta="⚠️ Review needed" if len(anomalies) > 0 else "✅ Stable")

st.markdown("---")

# ── Historical Forecast Chart ──────────────────────────────────────────────────
st.subheader(f"📊 Historical Forecast: {target_metric}")
st.line_chart(
    analyzed_df.set_index('Date')[[target_metric, 'Baseline_Forecast', 'Upper_Bound', 'Lower_Bound']]
)

# ── Future Forecast Chart ──────────────────────────────────────────────────────
st.subheader(f"🔭 {forecast_days}-Day Forward Forecast")

tab1, tab2 = st.tabs(["Baseline Forecast", "Scenario Comparison"])

with tab1:
    st.line_chart(future_df.set_index('Date')[['Central', 'Low', 'High']])
    st.caption(
        f"Central estimate: **{future_df['Central'].iloc[0]:,.2f}** | "
        f"Range: {future_df['Low'].iloc[0]:,.2f} – {future_df['High'].iloc[0]:,.2f}"
    )

with tab2:
    comparison = pd.DataFrame({
        'Baseline Central': future_df['Central'],
        'Scenario Central': scenario_df['Central'],
        'Scenario Low':     scenario_df['Low'],
        'Scenario High':    scenario_df['High']
    }, index=future_df['Date'])
    st.line_chart(comparison)
    diff = scenario_df['Central'].iloc[0] - future_df['Central'].iloc[0]
    st.caption(
        f"At **{scenario_pct:+.0f}%** adjustment — "
        f"Scenario central: {scenario_df['Central'].iloc[0]:,.2f} "
        f"(vs baseline {future_df['Central'].iloc[0]:,.2f}, "
        f"difference: {diff:+.2f})"
    )

st.markdown("---")

# ── Anomaly Detection ──────────────────────────────────────────────────────────
st.subheader("⚠️ Anomaly Detection")

if not anomalies.empty:
    st.error(f"Detected **{len(anomalies)}** sudden changes outside the normal forecast range.")
    st.dataframe(
        anomalies[['Date', target_metric, 'Baseline_Forecast', 'Upper_Bound', 'Lower_Bound']],
        use_container_width=True
    )
else:
    st.success("Platform health is stable. All metrics are within predicted forecast bands.")

st.markdown("---")

# ── Gemini AI Root Cause Analysis ─────────────────────────────────────────────
st.subheader("🤖 AI Root Cause Analysis")

if not anomalies.empty:
    if st.button("🔍 Analyse Anomalies with Gemini"):
        with st.spinner("Gemini is analysing the anomaly pattern..."):
            explanation = get_ai_explanation(target_metric, anomalies, future_df)
        st.info(explanation)
        with st.expander("📖 How was this explanation generated?"):
            st.markdown(
                "The anomaly rows (dates, actual values, and baseline forecasts) were "
                "sent to Google Gemini with a structured prompt asking for a plain-English "
                "root cause summary under 80 words. No raw model weights were used — "
                "this is a retrieval-augmented explanation layer on top of statistical outputs."
            )
else:
    st.info("No anomalies detected in the current view. Adjust the metric or baseline window to explore.")