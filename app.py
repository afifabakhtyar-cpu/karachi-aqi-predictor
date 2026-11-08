import streamlit as st
import pandas as pd
import requests
import joblib

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️")

LAT, LON = 24.8607, 67.0011
TZ = "Asia/Karachi"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
W_FORECAST = "https://api.open-meteo.com/v1/forecast"
AQ_VARS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
W_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"

def get(url, params):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json()["hourly"])

def make_features(d):
    d = d.copy().sort_values("time").reset_index(drop=True)
    for col in ["us_aqi", "pm2_5", "pm10"]:
        for lag in [1, 3, 6, 12, 24, 48]:
            d[f"{col}_lag{lag}"] = d[col].shift(lag)
        d[f"{col}_roll24"] = d[col].rolling(24).mean()
    d["hour"] = d["time"].dt.hour
    d["month"] = d["time"].dt.month
    d["dayofweek"] = d["time"].dt.dayofweek
    return d

def category(aqi):
    if aqi <= 50: return "Good 🟢"
    if aqi <= 100: return "Moderate 🟡"
    if aqi <= 150: return "Unhealthy for Sensitive Groups 🟠"
    if aqi <= 200: return "Unhealthy 🔴"
    if aqi <= 300: return "Very Unhealthy 🟣"
    return "Hazardous ⚫"

@st.cache_resource
def load_model():
    return joblib.load("aqi_model.pkl")

@st.cache_data(ttl=3600)
def get_forecast():
    bundle = load_model()
    model, features = bundle["model"], bundle["features"]
    aq = get(AQ_URL, dict(latitude=LAT, longitude=LON, hourly=AQ_VARS,
                          past_days=7, forecast_days=1, timezone=TZ))
    w = get(W_FORECAST, dict(latitude=LAT, longitude=LON, hourly=W_VARS,
                             past_days=7, forecast_days=1, timezone=TZ))
    rec = aq.merge(w, on="time", how="inner")
    rec["time"] = pd.to_datetime(rec["time"])
    now = pd.Timestamp.now(tz=TZ).tz_localize(None)
    rec = rec[rec["time"] <= now].dropna(subset=["us_aqi", "pm2_5"])
    rec = make_features(rec).dropna(subset=features)
    latest = rec.iloc[[-1]]
    pred = model.predict(latest[features])[0]
    base = latest["time"].iloc[0]
    current = float(latest["us_aqi"].iloc[0])
    rows = [{"Date": (base + pd.Timedelta(hours=h)).strftime("%d %b"),
             "AQI": round(float(p))} for h, p in zip([24, 48, 72], pred)]
    return base, current, pd.DataFrame(rows), rec

st.title("🌫️ Karachi AQI Predictor")
st.caption("Next 3 days air quality forecast using machine learning")

try:
    base, current, fc, rec = get_forecast()
    st.metric("Current AQI", round(current), category(current))
    st.subheader("3-Day Forecast")
    cols = st.columns(3)
    for col, (_, row) in zip(cols, fc.iterrows()):
        col.metric(row["Date"], row["AQI"], category(row["AQI"]))
    st.bar_chart(fc.set_index("Date")["AQI"])
    st.subheader("Last 7 days AQI")
    st.line_chart(rec.set_index("time")["us_aqi"])
    st.caption(f"Latest data: {base}")
except Exception as e:
    st.error(f"Something went wrong: {e}")
