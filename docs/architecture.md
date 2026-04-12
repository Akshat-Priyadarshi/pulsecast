# 🏗️ Pulsecast — Technical Architecture

---

## 🌐 System Overview

Pulsecast is structured as a four-layer pipeline:

- **Layer 1: Data Layer** — Synthetic or real CSV input
- **Layer 2: Analytics Layer** — `forecasting.py`, `anomaly.py`, `keystone.py`
- **Layer 3: AI Layer** — Google Gemini API (explanation)
- **Layer 4: Presentation Layer** — Streamlit dashboard (`app.py`)

Each layer is independent. The analytics layer produces all statistical outputs without requiring the AI layer. The AI layer only adds plain-English interpretation on top of already-computed results.

---

## 🔄 Data Flow

```text
platform_health.csv
   │
   ▼
load_data() in app.py
   │
   ├──► calculate_forecast_and_anomalies()
   │        │
   │        └──► Baseline SMA, Upper/Lower Bounds, Is_Anomaly flag
   │
   ├──► run_holdout_validation()
   │        │
   │        └──► Train/test split, SMA MAE, Naive MAE, Improvement %
   │
   ├──► generate_future_forecast()
   │        │
   │        └──► Central, Low, High for N future days
   │
   ├──► trace_cascade()
   │        │
   │        └──► Pearson correlations in anomaly window, cascade roles
   │
   ├──► calculate_cascade_severity()
   │        │
   │        └──► Severity score 0-100, label, recommendation
   │
   └──► identify_keystone()
            │
            └──► Health Score, keystone metric, R² load factor
```

All outputs feed into `app.py` for rendering.
Anomaly data + cascade + keystone → Gemini API → plain-English explanation.

---

## 📈 Forecasting Engine

### Simple Moving Average (SMA)

The baseline forecast is a Simple Moving Average computed over a configurable window (default: `7` days):

> `SMA(t) = (1/w) × Σ x(t-i)` _(for `i = 0` to `w-1`)_

Where `w` is the window size and `x(t)` is the observed value at time `t`.

**Why SMA?**
The hackathon guidelines explicitly state that simple models often outperform overly complex ones on short-horizon forecasts. SMA is fully transparent — any user can reproduce the result with a calculator.

### Uncertainty Bands

Bands are computed using rolling standard deviation:

> `Upper(t) = SMA(t) + 2σ(t)`  
> `Lower(t) = SMA(t) - 2σ(t)`

`±2σ` covers approximately `95%` of expected variation under normal conditions. Any observation outside this band has less than a `5%` probability of occurring by chance — it is flagged as an anomaly.

**Key principle:** Uncertainty is not error. It is probabilistic information about the range of plausible futures.

### Future Forecast Uncertainty Growth

For the forward forecast, uncertainty grows with distance from the present to reflect increasing unknowns:

> `Uncertainty(d) = 1.5σ × growth_factor(d)`  
> `growth_factor = linspace(1.0, 1.5, forecast_days)`

A `1`-day forecast carries less uncertainty than a `42`-day forecast. This prevents false confidence in long-range predictions.

---

## 🧪 Hold-out Validation

The data is split `80/20` (approximately):

- **Full dataset:** `90` days
- **Training set:** `76` days _(used to compute SMA)_
- **Test set:** `14` days _(never seen during training)_

Two forecasts are computed on the test set:

1. **SMA forecast:** The final SMA value from training extended flat.
2. **Naive baseline:** The last observed training value repeated. This is the simplest possible forecast — if SMA cannot beat it, the model adds no value.

Mean Absolute Error is computed for both:

> `MAE = (1/n) × Σ |actual(t) - predicted(t)|`

Improvement percentage:

> `Improvement = ((MAE_naive - MAE_sma) / MAE_naive) × 100`

This directly satisfies the hackathon requirement to compare predictions to a simple baseline to avoid over-fitting.

---

## 🔗 Path-Event Cascade Analysis

### Motivation

A metric anomaly is rarely isolated. In complex systems, a fault in one component cascades through dependent components. For example:

```text
Login_Success_Rate drops
   │
   ▼
Error_Rate spikes (users retrying failed logins)
   │
   ▼
Support_Tickets surge (users reporting login failures)
   │
   ▼
Transactions drop (users unable to complete flows)
```

### Method

1. Identify the anomaly window in the trigger metric (`±3` days around each anomaly point).
2. For each other metric, compute the Pearson correlation coefficient within that window:
   > `r(X,Y) = Σ[(Xi - X̄)(Yi - Ȳ)] / √[Σ(Xi - X̄)² × Σ(Yi - Ȳ)²]`
3. Assign cascade roles:
   - `|r| ≥ 0.7` → Primary Cascade Effect
   - `|r| ≥ 0.4` → Secondary Effect
   - `|r| < 0.4` → Unaffected
4. Estimate lag: re-compute correlation with a `1`-day offset to detect delayed propagation effects.

### Severity Score

> `Score = (primary_count × 30) + (secondary_count × 10) + (mean |r| of primary effects × 30)`

_Score capped at `100`._

---

## 🎯 Keystone Element Method

### Motivation

In any complex system, not all components are equal. The Keystone Element is the single metric whose variance most strongly predicts the variance of the entire system. Identifying it lets teams focus monitoring on the highest-leverage point.

### Step 1 — Composite Health Score

Each metric is normalised to a z-score and weighted by business impact:

**Weight map:**

- Transactions: `+0.35` _(primary revenue driver)_
- Login*Success_Rate: `+0.30` *(primary access indicator)\_
- Error*Rate: `-0.20` *(negative: higher = worse health)\_
- Support*Tickets: `-0.15` *(negative: more = worse health)\_

> `z(metric) = (x - μ) / σ`  
> `Health_Score(t) = Σ [ z(metric, t) × weight(metric) ]`

_Scaled to `0–100` for readability._

### Step 2 — Keystone Identification

For each metric, compute its Pearson correlation with the Health Score:

> `r(metric, Health_Score)`

The metric with the highest `|r|` is the Keystone.

### Step 3 — Load Factor (R²)

> `Load_Factor = r² × 100`

This is the percentage of system variance explained by that single metric alone. A load factor of `65%` means that metric alone accounts for `65%` of all variation in platform health.

---

## 🧠 AI Explanation Layer

### Design

The Gemini API is used purely as a natural language interface on top of already-computed statistical results. It does not perform any analysis itself.

**Input to Gemini:**

- Anomaly rows (date, actual, baselin e)
- Cascade correlation table
- Keystone metric name
- Forward forecast range

**Prompt structure:**

- **Role:** platform reliability analyst writing for non-technical managers
- **Output:** plain English, under `120` words
- **Required coverage:** what happened, cascade or isolated, keystone context, one immediate next step

### Why Gemini Free Tier?

- No credit card required for the free tier
- Sufficient request volume for demo and judging
- Simple Python SDK: one package, one API key, one function call
- Stable enough for hackathon demo conditions

---

## 🔒 Security

- API keys are stored in `.env` (git-ignored)
- `.env.example` lists required variables without values
- No hardcoded credentials anywhere in the codebase
- No real user data — synthetic dataset only
