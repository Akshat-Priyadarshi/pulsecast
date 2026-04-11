"""
anomaly.py
----------
Anomaly detection and path-event cascade analysis for Pulsecast.

Contains:
- Anomaly flag extraction from forecast bands
- Path-event cascade tracer: identifies how one metric fault
  propagates to affect other metrics (the reliability modeling angle)
- Cascade severity scoring

Design principle:
    A spike or dip is not just a number — it is an event in a system.
    By tracing correlations between metrics in the anomaly window,
    we can identify whether an anomaly is isolated or part of
    a cascading failure chain.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ── Anomaly Extraction ─────────────────────────────────────────────────────────

def extract_anomalies(analyzed_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Extracts rows flagged as anomalies from the analyzed DataFrame.

    Parameters:
        analyzed_df : Output from calculate_forecast_and_anomalies()
        metric      : The metric column that was analyzed

    Returns:
        DataFrame of anomaly rows with relevant columns only.
    """
    anomalies = analyzed_df[analyzed_df['Is_Anomaly'] == True].dropna()
    return anomalies[['Date', metric, 'Baseline_Forecast', 'Upper_Bound', 'Lower_Bound']].copy()


# ── Path-Event Cascade Tracer ──────────────────────────────────────────────────

def trace_cascade(df: pd.DataFrame, trigger_metric: str, window_days: int = 3) -> pd.DataFrame:
    """
    Traces how a fault in the trigger metric correlates with
    changes in all other metrics within a time window.

    This is the path-event analysis component. When an anomaly
    is detected, we look at what happened to ALL other metrics
    in the surrounding window — revealing cascade patterns.

    Parameters:
        df             : Full historical DataFrame
        trigger_metric : The metric where the anomaly was detected
        window_days    : Days before/after anomaly to examine (default: 3)

    Returns:
        DataFrame with columns:
            Metric          : Name of the affected metric
            Correlation     : Pearson correlation with trigger metric
            Direction       : "Amplified" or "Suppressed"
            Lag_Days        : Estimated lag before effect appears
            Cascade_Role    : "Primary Driver", "Secondary Effect", or "Unaffected"

    How it works:
        1. Identify the anomaly window in the trigger metric
        2. For each other metric, compute the Pearson correlation
           in that window
        3. Rank by correlation strength to determine cascade order
        4. Assign roles: primary driver, secondary effect, unaffected
    """
    all_metrics = ['Transactions', 'Login_Success_Rate', 'Error_Rate', 'Support_Tickets']
    other_metrics = [m for m in all_metrics if m != trigger_metric]

    # Find the anomaly window — rows where trigger metric deviates most from its mean
    trigger_mean = df[trigger_metric].mean()
    trigger_std  = df[trigger_metric].std()
    anomaly_mask = (df[trigger_metric] - trigger_mean).abs() > (1.5 * trigger_std)

    if anomaly_mask.sum() == 0:
        return pd.DataFrame(columns=['Metric', 'Correlation', 'Direction', 'Cascade_Role'])

    # Get the anomaly window indices (expand by window_days on each side)
    anomaly_indices = df.index[anomaly_mask].tolist()
    expanded_indices = set()
    for idx in anomaly_indices:
        for offset in range(-window_days, window_days + 1):
            new_idx = idx + offset
            if 0 <= new_idx < len(df):
                expanded_indices.add(new_idx)

    window_df = df.iloc[sorted(expanded_indices)].copy()

    results = []
    for metric in other_metrics:
        if len(window_df) < 3:
            continue

        try:
            corr, p_value = pearsonr(
                window_df[trigger_metric].values,
                window_df[metric].values
            )
        except Exception:
            corr, p_value = 0.0, 1.0

        abs_corr = abs(corr)

        # Determine cascade role based on correlation strength
        if abs_corr >= 0.7:
            role = "🔴 Primary Cascade Effect"
        elif abs_corr >= 0.4:
            role = "🟡 Secondary Effect"
        else:
            role = "🟢 Unaffected"

        direction = "Moves Together" if corr > 0 else "Moves Opposite"

        # Estimate lag: check if the effect appears 1 day after trigger
        lag_corr = 0.0
        if len(window_df) > 1:
            try:
                lag_corr, _ = pearsonr(
                    window_df[trigger_metric].values[:-1],
                    window_df[metric].values[1:]
                )
            except Exception:
                lag_corr = 0.0

        lag_days = 1 if abs(lag_corr) > abs_corr else 0

        results.append({
            'Metric':        metric,
            'Correlation':   round(corr, 3),
            'Abs_Corr':      round(abs_corr, 3),
            'Direction':     direction,
            'Lag_Days':      lag_days,
            'Cascade_Role':  role,
            'P_Value':       round(p_value, 4)
        })

    result_df = pd.DataFrame(results).sort_values('Abs_Corr', ascending=False)
    return result_df.drop(columns=['Abs_Corr'])


# ── Cascade Severity Score ─────────────────────────────────────────────────────

def calculate_cascade_severity(cascade_df: pd.DataFrame) -> dict:
    """
    Computes an overall severity score for a cascade event.

    Score is based on how many metrics are primarily affected
    and the average correlation strength of those effects.

    Returns:
        Dictionary with score (0-100), label, and recommendation.
    """
    if cascade_df.empty:
        return {'score': 0, 'label': 'No Cascade', 'recommendation': 'Monitor normally.'}

    primary   = cascade_df[cascade_df['Cascade_Role'].str.contains('Primary')]
    secondary = cascade_df[cascade_df['Cascade_Role'].str.contains('Secondary')]

    score = (len(primary) * 30) + (len(secondary) * 10)
    if not primary.empty:
        score += primary['Correlation'].abs().mean() * 30
    score = min(int(score), 100)

    if score >= 70:
        label          = "🔴 Severe Cascade"
        recommendation = "Immediate investigation required. Multiple metrics affected. Check system logs and escalate."
    elif score >= 40:
        label          = "🟡 Moderate Cascade"
        recommendation = "Review affected metrics within 24 hours. Identify the root trigger and monitor closely."
    else:
        label          = "🟢 Isolated Anomaly"
        recommendation = "Anomaly appears contained. Monitor the trigger metric for recurrence over the next 48 hours."

    return {
        'score':          score,
        'label':          label,
        'recommendation': recommendation
    }