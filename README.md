# AgentWeather

A 24-hour weather and air quality forecasting system for Karachi. It collects data every hour, trains models daily, and uses four LangGraph agents to forecast, raise alerts, write a news-style weather report and monitor drift. Every agent decision is logged.

**Live dashboard:** https://karachi-aqi-predictor-afifa-dwrn5hcjphgswquuyexfd2.streamlit.app/

**Project report:** [AgentWeather project report (Google Doc)](https://docs.google.com/document/d/1xZZrSJHuTYMdfkFW-3QExC4RWA0HI2T_QZ-V36lJpP0/edit?usp=drivesdk)
## What it forecasts
Six metrics, 24 hours ahead: AQI, temperature (C), precipitation (mm and rain probability), humidity (%), UV index, and wind (speed and direction).

## Architecture
```
Open-Meteo APIs (air quality + weather)
        |  every hour (GitHub Actions)
        v
pipelines/ingest.py -> impute -> features -> Hopsworks Feature Store
        |
        v
agents/graph.py (LangGraph): Forecast -> Alert -> Weather Report -> Drift
        |                       |                     |
        v                       v                     v
 Forecast Ledger          Agent call log          Drift log   (all stored in Hopsworks)
        |
        v
pipelines/train.py (daily, and when drift is detected) -> Hopsworks Model Registry
        |
        v
app.py -> Streamlit dashboard
```

## Technology stack
Python, Pandas, NumPy, Scikit-learn, PyTorch (LSTM), Statsmodels (SARIMAX), SHAP, Hopsworks (Feature Store and Model Registry), LangGraph, Google Gemini API, GitHub Actions, Streamlit Community Cloud, Plotly, Git.

## Data sources
- Open-Meteo Air Quality API: US AQI, PM2.5, PM10, UV index
- Open-Meteo Weather Archive API and Forecast API: temperature, humidity, precipitation, wind speed and direction, pressure

The sources are time-aligned on the hourly timestamp into one record per timestamp. Where the archive and forecast APIs overlap, the archive has priority (source priority fusion). Open-Meteo was used because it is free, needs no API key, and provides air quality and weather together. AQI follows the US EPA scale (0 to 500).

## Feature pipeline
- **Imputation (documented strategy):** reindex to a complete hourly grid, linear interpolation for gaps up to 6 hours, UV set to 0 when missing (night), longer gaps dropped.
- **Time features:** hour, day, month, day of week, hour sine and cosine.
- **Derived features:** lags (1, 3, 6, 12, 24 hours), 24-hour rolling means, 3-hour rate of change, 24-hour precipitation total, wind direction as sine and cosine.
- **Storage:** raw observations (`karachi_raw_obs`) and features with targets (`karachi_features`) in the Hopsworks Feature Store.
- **Backfill:** the same code was run on data from 1 January 2024 (25,027 hourly rows). Targets are shifted 24 hours ahead. The newest 24 hours have no target and are used only for forecasting.

## Training pipeline
- **Split:** time-based, 70% train, 15% validation, 15% test.
- **Model families:** SARIMAX (statistical), Random Forest (classical ML), LSTM (deep learning), each compared with a naive baseline (the value from 24 hours ago).
- **Metrics:** RMSE, MAE and R2 for every model and metric.
- **Model Registry:** the better of Random Forest and LSTM on validation RMSE is registered. A new model replaces the current one only if it scores better on validation.
- **Schedule:** daily, and whenever the Drift Agent requests it.

### Results (test set)
| Metric | Naive R2 | Best model R2 |
|--------|----------|---------------|
| AQI | 0.51 | 0.63 |
| Temperature | 0.90 | 0.91 |
| Humidity | 0.74 | 0.80 |
| UV index | 0.94 | 0.95 |
| Wind speed | 0.56 | 0.62 |

Wind direction mean circular error: 14.3 degrees (naive 15.0). Rain probability classifier: accuracy 0.812, F1 0.659, ROC-AUC 0.867 (naive accuracy 0.785). SARIMAX scored poorly (negative R2 on most metrics) and is compared but not deployed. Precipitation in mm is hard to forecast because rain is rare in Karachi, so rain probability is the main rain output. Scores for every model and metric are in the dashboard and the report.

## Agentic layer (LangGraph, one shared graph)
| Agent | What it does |
|-------|--------------|
| Forecast | Loads the latest features and best models from the Registry, forecasts all metrics, saves them to the Forecast Ledger |
| Alert | Fixed rules check all metrics for hazards. An LLM writes alerts for the general public, sensitive groups and outdoor workers |
| Weather Report | Writes one news-style report daily and after a new alert |
| Drift | Data drift (population stability index, threshold 0.25) and model drift (recent error divided by validation error, threshold 1.5). On a breach it logs the event, raises a warning and starts an extra training run |

The Alert and Weather Report agents use Google Gemini. If a call fails, they fall back to template text built from the forecast, so alerts and reports are never missing.

## Logging (stored in the Hopsworks Feature Store)
- **Forecast Ledger:** time, metric and forecast of every forecast
- **Agent call log:** prompt, response, latency and status of every LLM call. Failed calls form the failure log
- **Drift log:** time, metric, drift type, score, threshold and action taken

## Explainability and EDA
SHAP values are computed for each metric's Random Forest. The ten strongest features per metric are stored in the Feature Store and shown in the dashboard (for AQI the strongest is PM2.5). The dashboard also has EDA for every metric: daily trend, distribution, average by hour and by month, and summary statistics.

## Automation (GitHub Actions)
- `.github/workflows/hourly.yml`: ingestion every hour, then the agents. If drift is detected, it raises a warning and starts an extra training run.
- `.github/workflows/daily.yml`: daily model training.

## Dashboard
Five tabs:
- **Forecast:** 24-hour forecasts for all six metrics and the recent AQI trend
- **Weather Report:** the latest news-style report and active alert
- **Agent Activity:** alerts, agent call log, failed calls, drift events and the Forecast Ledger
- **EDA:** analysis of each metric
- **Models and SHAP:** model comparison scores and SHAP explanations

## Repository structure
```
agents/                 common.py, forecast_agent.py, alert_agent.py, report_agent.py, drift_agent.py, graph.py
pipelines/              ingest.py, train.py
.github/workflows/      hourly.yml, daily.yml
app.py                  Streamlit dashboard
requirements.txt        dependencies
AgentWeather_Report.docx  project report
```

## Setup
1. Create a Hopsworks project named `AgentWeather` and generate an API key.
2. Add GitHub repository secrets: `HOPSWORKS_API_KEY` and `GEMINI_API_KEY`.
3. Add `HOPSWORKS_API_KEY` to the Streamlit app secrets (TOML format).
4. Run the Hourly pipeline and Daily training workflows from the Actions tab.

## Known limitations
- About two years of history, so seasonal patterns are learned from two cycles only.
- Precipitation in mm is weak; the rain probability model is more reliable.
- The free Hopsworks tier has a compute budget, which limits how often the pipelines can run.
- Only one city and one data provider are used.

## Author
Afifa Bakhtyar, Data Science Intern
