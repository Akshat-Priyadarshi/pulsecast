"""
forecasting.py
--------------
Core forecasting engine for Pulsecast.

Contains:
- Simple Moving Average (SMA) baseline forecast
- Uncertainty band calculation (±2 std dev)
- Future period forecast with low/central/high range
- Hold-out validation with MAE vs baseline MAE
- Scenario adjustment engine

Design principle:
    Uncertainty is not "error" — it is probabilistic information.
    Every forecast exposes its own methodology so non-experts
    can understand how results were produced.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


# ── Baseline Forecast (SMA + Uncertainty Bands) ────────────────────────────────

def calculate_forecast_and_anomalies(df: pd.DataFrame, metric: str, window: int = 7) -> pd.DataFrame:
    """
    Computes a Simple Moving Average baseline forecast with
    uncertainty bands using rolling standard deviation.

    Parameters:
        df      : DataFrame containing the time series data
        metric  : Column name of the metric to forecast
        window  : Rolling window size in days (default: 7)

    Returns:
        DataFrame with added columns:
            Baseline_Forecast : SMA prediction
            Upper_Bound       : Forecast + 2 std deviations
            Lower_Bound       : Forecast - 2 std deviations
            Is_Anomaly        : True if actual value breaches the bands

    Why SMA?
        The hackathon guidelines explicitly state that simple models
        often outperform overly complex ones. SMA is transparent,
        reproducible, and easy for non-experts to verify.

    Why ±2 std deviations?
        This covers approximately 95% of expected variation under
        normal operating conditions. Any point outside this band
        is statistically significant — not noise.
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


# ── Future Forecast ────────────────────────────────────────────────────────────

def generate_future_forecast(df: pd.DataFrame, metric: str, forecast_days: int = 7) -> pd.DataFrame:
    """
    Extends the SMA forecast into future periods not yet observed.

    Parameters:
        df            : Historical DataFrame
        metric        : Target metric column name
        forecast_days : Number of future days to forecast (1–42)

    Returns:
        DataFrame with columns: Date, Central, Low, High

    The uncertainty range uses ±1.5 std deviations for the forward
    forecast (slightly tighter than the anomaly bands) to reflect
    that short-term forecasts carry less uncertainty than long-term.
    The range widens proportionally with the horizon.
    """
    recent  = df[metric].tail(14)
    central = recent.mean()
    std     = recent.std()

    future_dates = pd.date_range(
        df['Date'].max(), periods=forecast_days + 1, freq='D'
    )[1:]

    # Uncertainty grows slightly with distance from present
    uncertainty_growth = np.linspace(1.0, 1.5, forecast_days)

    return pd.DataFrame({
        'Date':    future_dates,
        'Central': central,
        'Low':     central - (1.5 * std * uncertainty_growth),
        'High':    central + (1.5 * std * uncertainty_growth)
    })


# ── Hold-out Validation ────────────────────────────────────────────────────────

def run_holdout_validation(df: pd.DataFrame, metric: str, window: int = 7, test_size: int = 14) -> dict:
    """
    Splits historical data into train/test sets and validates the
    SMA forecast against a naive baseline (last observed value).

    Parameters:
        df        : Full historical DataFrame
        metric    : Target metric column name
        window    : SMA window size
        test_size : Number of days to hold out for testing (default: 14)

    Returns:
        Dictionary containing:
            model_mae    : MAE of the SMA forecast on held-out data
            baseline_mae : MAE of the naive baseline on held-out data
            improvement  : % improvement of model over naive baseline
            train_size   : Number of training days used
            test_size    : Number of test days evaluated
            verdict      : Plain-English interpretation of the result

    Why hold-out validation?
        Without testing on unseen data, any forecast can appear
        accurate simply by memorising the training set (over-fitting).
        Comparing to a naive baseline proves the model adds real value.

    Naive baseline:
        Predicts tomorrow = today's value. This is the simplest
        possible forecast. If our SMA cannot beat this, it is not
        adding value.
    """
    if len(df) < window + test_size:
        return {
            'error': f"Not enough data. Need at least {window + test_size} rows, got {len(df)}."
        }

    train_df = df.iloc[:-(test_size)].copy()
    test_df  = df.iloc[-(test_size):].copy().reset_index(drop=True)

    # SMA forecast on training data — extend into test period
    sma_value    = train_df[metric].tail(window).mean()
    sma_preds    = np.full(test_size, sma_value)

    # Naive baseline: last observed value repeated
    naive_value  = train_df[metric].iloc[-1]
    naive_preds  = np.full(test_size, naive_value)

    actuals      = test_df[metric].values

    model_mae    = mean_absolute_error(actuals, sma_preds)
    baseline_mae = mean_absolute_error(actuals, naive_preds)

    improvement  = ((baseline_mae - model_mae) / baseline_mae) * 100 if baseline_mae > 0 else 0.0

    if improvement > 10:
        verdict = f"✅ SMA forecast outperforms naive baseline by {improvement:.1f}%. The model adds real predictive value."
    elif improvement > 0:
        verdict = f"🟡 SMA forecast is marginally better than naive baseline ({improvement:.1f}% improvement). Consider a longer window."
    else:
        verdict = f"🔴 Naive baseline outperforms SMA on this metric. This may indicate high short-term volatility."

    return {
        'model_mae':    round(model_mae, 4),
        'baseline_mae': round(baseline_mae, 4),
        'improvement':  round(improvement, 2),
        'train_size':   len(train_df),
        'test_size':    test_size,
        'verdict':      verdict
    }


# ── Scenario Engine ────────────────────────────────────────────────────────────

def apply_scenario(forecast_df: pd.DataFrame, adjustment_pct: float) -> pd.DataFrame:
    """
    Applies a percentage adjustment to the central forecast for
    side-by-side scenario comparison.

    Parameters:
        forecast_df     : Output from generate_future_forecast()
        adjustment_pct  : Percentage change to apply (e.g., +10 or -15)

    Returns:
        Adjusted forecast DataFrame with same structure.
    """
    scenario = forecast_df.copy()
    factor   = 1 + (adjustment_pct / 100)
    scenario['Central'] *= factor
    scenario['Low']     *= factor
    scenario['High']    *= factor
    return scenario