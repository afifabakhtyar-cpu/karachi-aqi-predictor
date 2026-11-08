import os, sys
import numpy as np, pandas as pd, requests, hopsworks

LAT, LON = 24.8607, 67.0011
TZ = "Asia/Karachi"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
W_FORECAST = "https://api.open-meteo.com/v1/forecast"
AQ_VARS = "us_aqi,pm2_5,pm10,uv_index"
W_VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,pressure_msl"

def get(url, params):
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["hourly"])
    d["time"] = pd.to_datetime(d["time"])
    return d

now = pd.Timestamp.now(tz=TZ).tz_localize(None).floor("h")
aq = get(AQ_URL, dict(latitude=LAT, longitude=LON, hourly=AQ_VARS, past_days=10, forecast_days=1, timezone=TZ))
w = get(W_FORECAST, dict(latitude=LAT, longitude=LON, hourly=W_VARS, past_days=10, forecast_days=1, timezone=TZ))
raw = aq.merge(w, on="time", how="inner")
raw = raw[raw["time"] <= now].sort_values("time").reset_index(drop=True)
print("Fetched:", raw.shape, raw["time"].max())

# Imputation: reindex hourly, interpolate gaps up to 6h, UV night = 0, drop longer gaps
full = raw.set_index("time").asfreq("h")
clean = full.interpolate(method="linear", limit=6)
clean["uv_index"] = clean["uv_index"].fillna(0)
CORE = ["us_aqi", "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_direction_10m", "pressure_msl", "pm2_5", "pm10"]
clean = clean.dropna(subset=CORE).reset_index()

BASE = ["us_aqi", "temperature_2m", "relative_humidity_2m", "precipitation", "uv_index", "wind_speed_10m", "wind_sin", "wind_cos", "pm2_5"]

def make_features(d):
    d = d.sort_values("time").reset_index(drop=True).copy()
    rad = np.deg2rad(d["wind_direction_10m"])
    d["wind_sin"] = np.sin(rad)
    d["wind_cos"] = np.cos(rad)
    for c in BASE:
        for lag in [1, 3, 6, 12, 24]:
            d[f"{c}_lag{lag}"] = d[c].shift(lag)
        d[f"{c}_roll24"] = d[c].rolling(24).mean()
        d[f"{c}_diff3"] = d[c] - d[c].shift(3)
    d["precip_last24"] = d["precipitation"].rolling(24).sum()
    d["hour"] = d["time"].dt.hour
    d["day"] = d["time"].dt.day
    d["month"] = d["time"].dt.month
    d["dayofweek"] = d["time"].dt.dayofweek
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    return d

feat = make_features(clean)
feat["y_aqi"] = feat["us_aqi"].shift(-24)
feat["y_temp"] = feat["temperature_2m"].shift(-24)
feat["y_humidity"] = feat["relative_humidity_2m"].shift(-24)
feat["y_uv"] = feat["uv_index"].shift(-24)
feat["y_wind_speed"] = feat["wind_speed_10m"].shift(-24)
feat["y_wind_sin"] = feat["wind_sin"].shift(-24)
feat["y_wind_cos"] = feat["wind_cos"].shift(-24)
feat["y_precip"] = feat["precipitation"].rolling(24).sum().shift(-24)
feat["y_rain"] = np.where(feat["y_precip"].isna(), np.nan, (feat["y_precip"] >= 0.5).astype(float))
FEATURE_COLS = [c for c in feat.columns if not c.startswith("y_") and c != "time"]
feat = feat.dropna(subset=FEATURE_COLS).reset_index(drop=True)

def prep(d):
    d = d.copy()
    d["ts"] = (d["time"].astype("int64") // 10**9).astype("int64")
    d.columns = [c.lower() for c in d.columns]
    return d

project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"], project="AgentWeather")
fs = project.get_feature_store()
fs.get_feature_group("karachi_raw_obs", version=1).insert(prep(raw), write_options={"wait_for_job": False})
fs.get_feature_group("karachi_features", version=1).insert(prep(feat), write_options={"wait_for_job": False})
print("Ingestion done:", len(feat), "feature rows upserted")
