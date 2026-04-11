"""
keystone.py
-----------
Keystone Element Method for Pulsecast.

Identifies which metric acts as the critical load-bearing driver
of overall platform health — the "keystone" whose change predicts
changes in all other metrics.

Design principle:
    In any complex system, not all components are equal.
    The keystone element is the one whose variance most strongly
    predicts the variance of the entire system. Identifying it
    allows teams to focus monitoring and intervention resources
    on the highest-leverage point.

Method:
    1. Normalise all metrics to a common scale (z-score)
    2. Compute a composite "platform health score" as a weighted average
    3. For each metric, compute its correlation with the health score
    4. The metric with the highest correlation is the keystone
    5. Compute a "load factor" — how much of the system variance
       this metric explains
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ── Metric Weights ─────────────────────────────────────────────────────────────
# These weights reflect the relative business impact of each metric.
# Transactions and Login Success are primary health indicators.
# Error Rate and Support Tickets are secondary (reactive) indicators.
# These can be adjusted based on domain knowledge.

METRIC_WEIGHTS = {
    'Transactions':       0.35,
    'Login_Success_Rate': 0.30,
    'Error_Rate':        -0.20,   # Negative: higher error rate = worse health
    'Support_Tickets':   -0.15    # Negative: more tickets = worse health
}


# ── Platform Health Score ──────────────────────────────────────────────────────

def compute_health_score(df: pd.DataFrame) -> pd.Series:
    """
    Computes a single composite platform health score per day
    by normalising each metric and applying business-impact weights.

    Parameters:
        df : Historical DataFrame with all four metric columns

    Returns:
        Series of daily health scores (higher = healthier platform)

    The score is normalised to a 0-100 scale for intuitive readability.
    A score below 40 is considered unhealthy.
    A score above 70 is considered healthy.
    """
    normalised = pd.DataFrame()

    for metric, weight in METRIC_WEIGHTS.items():
        if metric not in df.columns:
            continue
        col_mean = df[metric].mean()
        col_std  = df[metric].std()
        if col_std == 0:
            normalised[metric] = 0.0
        else:
            z_score = (df[metric] - col_mean) / col_std
            normalised[metric] = z_score * weight

    raw_score = normalised.sum(axis=1)

    # Scale to 0-100
    score_min = raw_score.min()
    score_max = raw_score.max()
    if score_max == score_min:
        return pd.Series(np.full(len(raw_score), 50.0), index=df.index)

    health_score = 100 * (raw_score - score_min) / (score_max - score_min)
    return health_score.round(2)


# ── Keystone Identifier ────────────────────────────────────────────────────────

def identify_keystone(df: pd.DataFrame) -> dict:
    """
    Identifies the keystone metric — the single metric whose behaviour
    most strongly drives overall platform health.

    Parameters:
        df : Historical DataFrame

    Returns:
        Dictionary containing:
            keystone_metric  : Name of the keystone metric
            load_factor      : % of system variance this metric explains
            correlation      : Pearson correlation with health score
            all_correlations : Ranked list of all metrics by influence
            interpretation   : Plain-English explanation of the result
            health_scores    : Full Series of daily health scores

    Load Factor interpretation:
        This is computed as the square of the Pearson correlation (R²),
        representing the proportion of variance in the health score
        that is explained by variance in the keystone metric alone.
    """
    metrics = [m for m in METRIC_WEIGHTS.keys() if m in df.columns]

    if len(metrics) < 2:
        return {'error': 'Not enough metrics to compute keystone element.'}

    health_scores = compute_health_score(df)

    correlations = {}
    for metric in metrics:
        try:
            corr, p_val = pearsonr(df[metric].values, health_scores.values)
            correlations[metric] = {
                'correlation': round(corr, 4),
                'load_factor': round(corr ** 2 * 100, 2),   # R² as percentage
                'p_value':     round(p_val, 4),
                'significant': p_val < 0.05
            }
        except Exception:
            correlations[metric] = {
                'correlation': 0.0,
                'load_factor': 0.0,
                'p_value':     1.0,
                'significant': False
            }

    # Rank by absolute correlation
    ranked = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]['correlation']),
        reverse=True
    )

    keystone_name   = ranked[0][0]
    keystone_data   = ranked[0][1]
    load_factor     = keystone_data['load_factor']
    correlation     = keystone_data['correlation']

    # Build ranked DataFrame for display
    ranked_df = pd.DataFrame([
        {
            'Metric':       name,
            'Correlation':  data['correlation'],
            'Load Factor':  f"{data['load_factor']}%",
            'Significant':  "✅ Yes" if data['significant'] else "❌ No",
            'Role':         "🔑 Keystone" if name == keystone_name else "Supporting"
        }
        for name, data in ranked
    ])

    # Plain-English interpretation
    direction = "positively" if correlation > 0 else "negatively"
    interpretation = (
        f"**{keystone_name}** is the keystone element, explaining "
        f"**{load_factor:.1f}%** of overall platform health variance. "
        f"It correlates {direction} with system health (r = {correlation:.3f}). "
        f"When {keystone_name} deviates from normal, the rest of the "
        f"platform is most likely to follow. This is your highest-priority "
        f"monitoring target."
    )

    return {
        'keystone_metric':  keystone_name,
        'load_factor':      load_factor,
        'correlation':      correlation,
        'all_correlations': ranked_df,
        'interpretation':   interpretation,
        'health_scores':    health_scores
    }