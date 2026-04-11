"""
test_forecasting.py
-------------------
Validation tests for the Pulsecast forecasting engine.
Run with: pytest tests/
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from forecasting import (
    calculate_forecast_and_anomalies,
    generate_future_forecast,
    run_holdout_validation,
    apply_scenario
)
from anomaly import trace_cascade, calculate_cascade_severity
from keystone import identify_keystone, compute_health_score


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Creates a minimal 30-day platform health DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='D')
    df = pd.DataFrame({
        'Date':               dates,
        'Transactions':       np.random.normal(15000, 500, 30).astype(int),
        'Login_Success_Rate': np.random.normal(98.5, 0.3, 30).round(2),
        'Error_Rate':         np.random.normal(1.2, 0.1, 30).round(2),
        'Support_Tickets':    np.random.normal(150, 15, 30).astype(int)
    })
    return df


# ── Forecasting Tests ──────────────────────────────────────────────────────────

def test_forecast_columns_exist(sample_df):
    result = calculate_forecast_and_anomalies(sample_df, 'Transactions')
    assert 'Baseline_Forecast' in result.columns
    assert 'Upper_Bound' in result.columns
    assert 'Lower_Bound' in result.columns
    assert 'Is_Anomaly' in result.columns


def test_upper_bound_above_lower(sample_df):
    result = calculate_forecast_and_anomalies(sample_df, 'Transactions')
    valid  = result.dropna()
    assert (valid['Upper_Bound'] >= valid['Lower_Bound']).all()


def test_future_forecast_length(sample_df):
    result = generate_future_forecast(sample_df, 'Transactions', forecast_days=7)
    assert len(result) == 7


def test_future_forecast_uncertainty_grows(sample_df):
    result = generate_future_forecast(sample_df, 'Transactions', forecast_days=7)
    bands  = result['High'] - result['Low']
    assert bands.iloc[-1] >= bands.iloc[0], "Uncertainty should grow with forecast horizon"


def test_scenario_increases_forecast(sample_df):
    future   = generate_future_forecast(sample_df, 'Transactions', 7)
    scenario = apply_scenario(future, 10)
    assert (scenario['Central'] > future['Central']).all()


def test_scenario_decreases_forecast(sample_df):
    future   = generate_future_forecast(sample_df, 'Transactions', 7)
    scenario = apply_scenario(future, -10)
    assert (scenario['Central'] < future['Central']).all()


# ── Validation Tests ───────────────────────────────────────────────────────────

def test_holdout_returns_mae(sample_df):
    result = run_holdout_validation(sample_df, 'Transactions', window=7, test_size=7)
    assert 'model_mae' in result
    assert 'baseline_mae' in result
    assert result['model_mae'] >= 0


def test_holdout_insufficient_data():
    tiny_df = pd.DataFrame({
        'Date':         pd.date_range(end=pd.Timestamp.today(), periods=5, freq='D'),
        'Transactions': [100, 101, 102, 103, 104]
    })
    result = run_holdout_validation(tiny_df, 'Transactions', window=7, test_size=14)
    assert 'error' in result


# ── Keystone Tests ─────────────────────────────────────────────────────────────

def test_health_score_range(sample_df):
    scores = compute_health_score(sample_df)
    assert scores.min() >= 0
    assert scores.max() <= 100


def test_keystone_returns_valid_metric(sample_df):
    result = identify_keystone(sample_df)
    assert 'keystone_metric' in result
    assert result['keystone_metric'] in [
        'Transactions', 'Login_Success_Rate', 'Error_Rate', 'Support_Tickets'
    ]


def test_keystone_load_factor_valid(sample_df):
    result = identify_keystone(sample_df)
    assert 0 <= result['load_factor'] <= 100


# ── Cascade Tests ──────────────────────────────────────────────────────────────

def test_cascade_returns_dataframe(sample_df):
    result = trace_cascade(sample_df, 'Transactions')
    assert isinstance(result, pd.DataFrame)


def test_cascade_severity_score_range(sample_df):
    cascade = trace_cascade(sample_df, 'Transactions')
    severity = calculate_cascade_severity(cascade)
    assert 0 <= severity['score'] <= 100