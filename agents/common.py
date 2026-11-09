import os, time, json
import numpy as np, pandas as pd, joblib, requests
import hopsworks

LAT, LON = 24.8607, 67.0011
TZ = "Asia/Karachi"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
W_FORECAST = "https://api.open-meteo.com/v1/forecast"
GEMINI_MODEL = "gemini-3.6-flash"

TARGETS = ["y_aqi", "y_temp", "y_humidity", "y_uv", "y_wind_speed", "y_wind_sin", "y_wind_cos", "y_precip"]

_project = None
def get_project():
    global _project
    if _project is None:
        _project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"], project="AgentWeather")
    return _project

def get_fs():
    return get_project().get_feature_store()

def log_rows(fg_name, df, key_cols, desc):
    """Append rows to a logging feature group in Hopsworks."""
    fs = get_fs()
    df = df.copy()
    df["ts"] = pd.Timestamp.now()
    df["key"] = df[key_cols].astype(str).agg("_".join, axis=1) + "_" + str(int(time.time() * 1000))
    fg = fs.get_or_create_feature_group(name=fg_name, version=1, primary_key=["key"], event_time="ts", description=desc)
    fg.insert(df, write_options={"wait_for_job": False})

def call_llm(agent, prompt, fallback_text):
    """Call Gemini. On any failure return fallback_text. Every call is logged."""
    start = time.time()
    status, response = "ok", ""
    try:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing")
        url = "https://generativelanguage.googleapis.com/v1beta/models/" + GEMINI_MODEL + ":generateContent"
        r = requests.post(url, params={"key": key}, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=40)
        r.raise_for_status()
        response = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if not response:
            raise RuntimeError("empty response")
    except Exception as e:
        status = "failed: " + str(e)[:150]
        response = fallback_text
    latency = round(time.time() - start, 2)
    try:
        log_rows("agent_call_log", pd.DataFrame([{"agent": agent, "prompt": prompt[:1500], "response": response[:1500], "latency_s": float(latency), "status": status, "fallback_used": status != "ok"}]), ["agent"], "Every LLM call: prompt, response, latency, status")
    except Exception as e:
        print("log failed:", e)
    return response, status == "ok"

def load_model(name):
    mr = get_project().get_model_registry()
    m = mr.get_best_model(name, "val_rmse", "min")
    folder = m.download()
    return joblib.load(os.path.join(folder, "model.pkl")), folder
