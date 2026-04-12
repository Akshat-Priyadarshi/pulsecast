# 📈 Pulsecast: Predictive Platform Health

> Transparent AI forecasting — look ahead, not backwards.

Pulsecast is an AI-powered predictive forecasting tool that transforms
historical platform health data into actionable forward-looking insights.
It detects early warning signs, traces cascading system faults, identifies
the single most critical metric driving overall platform health, and delivers
plain-English explanations that non-technical teams can act on immediately.
Built for operations, reliability, and product teams who need honest signals,
not overconfident predictions.

---

## ✅ Features

The following features are fully implemented and working:

- **Short-term forecasting** — Simple Moving Average baseline with a
  configurable 3–14 day window, extended up to 42 days forward.

- **Uncertainty bands** — Every forecast shows a low/central/high range
  using ±2 standard deviations. Uncertainty grows proportionally with
  forecast horizon.

- **Hold-out validation** — The model is tested on 14 days of unseen data
  and compared against a naive baseline. MAE and improvement percentage
  are displayed on screen so judges and users can verify the model adds
  real value.

- **Anomaly detection** — Any data point breaching the forecast bands is
  flagged immediately with its date, actual value, and expected range.

- **Path-event cascade analysis** — When an anomaly is detected, Pulsecast
  traces how the fault in one metric propagates to all other metrics using
  Pearson correlation in the anomaly window. Each affected metric is assigned
  a cascade role (Primary Effect, Secondary Effect, or Unaffected) and a
  lag estimate.

- **Cascade severity scoring** — A 0–100 severity score summarises whether
  an anomaly is isolated or part of a multi-metric failure chain, with a
  plain-English recommendation for each severity level.

- **Keystone Element identification** — Identifies the single metric that
  explains the most variance in overall platform health (R² load factor).
  This is your highest-priority monitoring target on any given day.

- **Platform Health Score** — A composite 0–100 daily health score computed
  from all four metrics, weighted by business impact, displayed as a time
  series chart.

- **Scenario comparison** — Users can apply a percentage adjustment
  (−30% to +30%) to the forward forecast and compare the scenario against
  the baseline side by side with exact difference values.

- **Gemini AI root cause analysis** — Anomaly data, cascade results, and
  the keystone metric are sent to Google Gemini, which returns a plain-English
  explanation under 120 words covering what happened, whether it cascaded,
  and what to do next.

- **Synthetic dataset generator** — A reproducible script generates 90 days
  of realistic platform health data with a deliberately planted cascading
  fault at Day 75.

- **13 automated tests** — Full pytest suite covering forecasting, validation,
  keystone identification, and cascade analysis.

---

## 🛠 Install and Run

### Prerequisites

- Python 3.10 or higher
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pulsecast.git
cd pulsecast
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
nano .env
```

Replace `your_api_key_here` with your real Gemini API key. Save and close.

### 5. Generate the synthetic dataset

```bash
python scripts/generate_data.py
```

Expected output:
