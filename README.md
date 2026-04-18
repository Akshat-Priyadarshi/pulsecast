# 📈 Pulsecast: Predictive Platform Health

> _"Transparent AI forecasting — look ahead, not backwards."_  
> Pulsecast is an AI-powered predictive forecasting tool that transforms historical platform health data into actionable forward-looking insights. It detects early warning signs, traces cascading system faults, identifies the single most critical metric driving overall platform health, and delivers plain-English explanations that non-technical teams can act on immediately. Built for operations, reliability, and product teams who need honest signals, not overconfident predictions.

---

## 🚀 Key Features

The following features are fully implemented and working:

- ⏱️ **Short-term forecasting** Simple Moving Average baseline with a configurable `3–14` day window, extended up to `42` days forward.

- 📉 **Uncertainty bands** Every forecast shows a low/central/high range using `±2` standard deviations. Uncertainty grows proportionally with forecast horizon.

- 🧪 **Hold-out validation** The model is tested on `14` days of unseen data and compared against a naive baseline. MAE and improvement percentage are displayed on screen so judges and users can verify the model adds real value.

- 🚨 **Anomaly detection** Any data point breaching the forecast bands is flagged immediately with its date, actual value, and expected range.

- 🔗 **Path-event cascade analysis** When an anomaly is detected, Pulsecast traces how the fault in one metric propagates to all other metrics using Pearson correlation in the anomaly window. Each affected metric is assigned a cascade role (Primary Effect, Secondary Effect, or Unaffected) and a lag estimate.

- 📊 **Cascade severity scoring** A `0–100` severity score summarises whether an anomaly is isolated or part of a multi-metric failure chain, with a plain-English recommendation for each severity level.

- 🎯 **Keystone Element identification** Identifies the single metric that explains the most variance in overall platform health (`R²` load factor). This is your highest-priority monitoring target on any given day.

- ❤️ **Platform Health Score** A composite `0–100` daily health score computed from all four metrics, weighted by business impact, displayed as a time series chart.

- 🔀 **Scenario comparison** Users can apply a percentage adjustment (`−30%` to `+30%`) to the forward forecast and compare the scenario against the baseline side by side with exact difference values.

- 🧠 **Gemini AI root cause analysis** Anomaly data, cascade results, and the keystone metric are sent to Google Gemini, which returns a plain-English explanation under 120 words covering what happened, whether it cascaded, and what to do next.

- ⚙️ **Synthetic dataset generator** A reproducible script generates `90` days of realistic platform health data with a deliberately planted cascading fault at Day 75.

- ✅ **13 automated tests** Full pytest suite covering forecasting, validation, keystone identification, and cascade analysis.

---

## 📁 Folder Structure

```bash
pulsecast/
├── src/
│   ├── app.py               # Main Streamlit application
│   ├── forecasting.py       # SMA engine, validation, scenario logic
│   ├── anomaly.py           # Anomaly detection, cascade tracer
│   └── keystone.py          # Keystone Element identifier
├── tests/
│   └── test_forecasting.py  # 13 pytest cases
├── data/
│   └── platform_health.csv  # Generated synthetic dataset
├── scripts/
│   └── generate_data.py     # Dataset generator
├── docs/
│   └── architecture.md      # Technical depth and system design
├── .env.example             # API key template
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## ⚙️ Tech Stack

- **Language**: Python 3.10 (Core application)
- **Frontend**: Streamlit 1.32 (Interactive dashboard UI)
- **Data processing**: Pandas 2.2, NumPy 1.26 (Time series manipulation)
- **Statistical analysis**: SciPy 1.12 (Pearson correlation for cascade tracing)
- **Validation**: Scikit-learn 1.4 (MAE computation for hold-out testing)
- **AI explanation**: Google Gemini API [Free Tier] (Plain-English root cause summaries)
- **Environment**: python-dotenv (Secure API key management)
- **Testing**: pytest (Automated test suite)

---

## 🛠️ Getting Started

1. **Prerequisites** - Python 3.10 or higher  
   - A free Gemini API key from [Google AI Studio](https://aistudio.google.com)  
   - Git

2. **Clone the repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/pulsecast.git](https://github.com/YOUR_USERNAME/pulsecast.git)
   cd pulsecast
   ```

3. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure your API key**
   ```bash
   cp .env.example .env
   nano .env
   ```
   *Replace `your_api_key_here` with your real Gemini API key. Save and close.*

6. **Generate the synthetic dataset**
   ```bash
   python scripts/generate_data.py
   ```
   *Expected output: ✅ Dataset generated: data/platform_health.csv | Rows: 90 | Columns: ['Date', 'Transactions', 'Login_Success_Rate', 'Error_Rate', 'Support_Tickets']*

7. **Run the application**
   ```bash
   cd src
   streamlit run app.py
   ```
   *The app opens automatically at `http://localhost:8501`*

8. **Run the tests (optional)**
   ```bash
   cd ~/pulsecast
   pytest tests/ -v
   ```
   *Expected: 13 passed.*

---

## 💡 Usage Examples

**Example 1 — Detecting a Cascading Fault**
1. Open the app at `http://localhost:8501`
2. Select **Login_Success_Rate** from the sidebar metric selector
3. The historical chart immediately shows the Day 75 anomaly — a severe drop in login success rate breaching the lower forecast band
4. Scroll to **Path-Event Cascade Analysis** — you will see that Error_Rate and Support_Tickets show Primary Cascade Effect, confirming this is a system-wide fault, not an isolated blip
5. Click **Analyse with Gemini AI** for a plain-English summary of what happened and what to do

**Example 2 — Scenario Testing**
1. Select **Transactions** from the metric selector
2. In the sidebar, drag the **Adjust Forecast by (%)** slider to `+10%`
3. Open the **Scenario Comparison** tab in the Forward Forecast section
4. The chart shows baseline vs scenario side by side with exact difference values — e.g. *"Scenario central: 15,234 vs baseline 13,849 (difference: +1,385)"*

**Example 3 — Identifying Your Keystone Metric**
1. Scroll to **Keystone Element Analysis**
2. The table ranks all four metrics by their correlation with the composite Platform Health Score
3. The top metric (load factor shown as a percentage) is your highest-priority monitoring target today

---

## 🏗️ Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full technical design including system diagrams, data flow, and mathematical methodology.

**Brief summary:**
```text
CSV Data
   │
   ▼
forecasting.py ───► SMA Baseline + Uncertainty Bands
   │                   │
   ▼                   ▼
anomaly.py ───────► Cascade Tracer + Severity Score
   │
   ▼
keystone.py ──────► Health Score + Keystone Identifier
   │
   ▼
Gemini API ───────► Plain-English Explanation
   │
   ▼
app.py ───────────► Streamlit Dashboard
```

---

## ⚠️ Limitations

The following limitations are honest descriptions of the current state:

- **Dataset is synthetic.** The included dataset is generated, not real production data. The tool is designed to accept any CSV in the same format — real data will produce real insights.
- **Gemini dependency.** The AI explanation feature requires a valid Gemini API key. All statistical features (forecasting, anomaly detection, cascade analysis, keystone identification) work fully without it.
- **SMA only.** The current forecasting engine uses Simple Moving Average. More advanced models (ARIMA, Prophet) would improve accuracy on data with strong seasonality, but are not implemented in this version.
- **Four fixed metrics.** The current version is built around the four platform health metrics in the synthetic dataset. Support for arbitrary CSV column names is not yet implemented.
- **No authentication.** The Streamlit app has no user login. It is intended as a local or internal tool, not a public-facing deployment.

---

## 🔮 Future Improvements

With additional time, the following enhancements would be prioritised:

- Support for arbitrary user-uploaded CSVs with automatic column detection
- ARIMA or Facebook Prophet integration for seasonal time series
- Email or Slack alerting when anomalies are detected
- Persistent anomaly history log across sessions
- Multi-metric simultaneous forecast comparison

---

## 📄 License

This project is licensed under the **Apache License 2.0**.  
See [LICENSE](LICENSE) for details.
