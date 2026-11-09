import os, json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hopsworks

YOUR_NAME = "Afifa Bakhtyar"
NAVY = "#0b2545"
TEAL = "#0d9488"
BLUE = "#134074"

st.set_page_config(page_title="AgentWeather", page_icon="🌦️", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.hero {background: linear-gradient(135deg, #0b2545 0%, #134074 45%, #0d9488 100%);
  padding: 2rem 1.5rem; border-radius: 20px; text-align: center; color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.3rem; color: white;}
.hero p {margin: .5rem 0 0 0; opacity: .9;}
.card {padding: 1rem; border-radius: 16px; text-align: center; color: white;
  background: linear-gradient(145deg, #0b2545, #0d9488); margin-bottom: .8rem;}
.card .lab {font-size: .85rem; opacity: .9;}
.card .val {font-size: 1.9rem; font-weight: 700;}
.news {padding: 1.2rem; border-radius: 14px; background: white; color: #0b2545;
  border-left: 6px solid #0d9488; box-shadow: 0 2px 8px rgba(11,37,69,.12); white-space: pre-wrap;}
.alert {padding: 1rem; border-radius: 12px; background: #fff3cd; color: #5c4400;
  border-left: 6px solid #e67e22; white-space: pre-wrap;}
.footer-custom {text-align: center; padding: 1rem; margin-top: 2rem; color: white;
  background: linear-gradient(90deg, #0b2545, #0d9488); border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

CFG = {"staticPlot": True}

def style(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color=NAVY), dragmode=False, margin=dict(t=50, b=40, l=40, r=20))
    return fig

@st.cache_resource(ttl=3600)
def fs():
    key = st.secrets["HOPSWORKS_API_KEY"]
    key = "".join(str(key).split())
    return hopsworks.login(api_key_value=key, project="AgentWeather").get_feature_store()

def read(name):
    try:
        return fs().get_feature_group(name, version=1).read()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def load_all():
    return {n: read(n) for n in ["karachi_features", "weather_reports", "forecast_ledger",
                                 "agent_call_log", "drift_log", "model_scores", "shap_importance"]}

st.markdown('<div class="hero"><h1>🌦️ AgentWeather</h1><p>24-hour weather and air quality forecast for Karachi, powered by AI agents</p></div>', unsafe_allow_html=True)

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("Data: Open-Meteo | Store: Hopsworks | Agents: LangGraph + Gemini")

try:
    D = load_all()
except Exception as e:
    st.error("Could not connect to Hopsworks: " + str(e))
    st.stop()

feat = D["karachi_features"]
reports = D["weather_reports"]
ledger = D["forecast_ledger"]
calls = D["agent_call_log"]
drift = D["drift_log"]
scores = D["model_scores"]
shap_df = D["shap_importance"]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Forecast", "Weather Report", "Agent Activity", "EDA", "Models and SHAP"])

with tab1:
    if ledger.empty:
        st.info("No forecast yet. Run the Hourly pipeline in GitHub Actions.")
    else:
        ledger = ledger.sort_values("ts")
        last = ledger.groupby("metric").tail(1).set_index("metric")["forecast"]
        st.caption("Latest forecast for: " + str(ledger.iloc[-1]["forecast_for"]))
        ang = None
        items = [("AQI", last.get("aqi"), ""), ("Temperature", last.get("temperature_c"), " C"),
                 ("Humidity", last.get("humidity_pct"), " %"), ("UV index", last.get("uv_index"), ""),
                 ("Wind speed", last.get("wind_speed_kmh"), " km/h"), ("Wind direction", last.get("wind_direction_deg"), " deg"),
                 ("Precipitation", last.get("precipitation_mm"), " mm"), ("Rain probability", last.get("rain_probability_pct"), " %")]
        cols = st.columns(4)
        for i, (lab, v, u) in enumerate(items):
            txt = "n/a" if v is None or pd.isna(v) else str(round(float(v), 1)) + u
            cols[i % 4].markdown('<div class="card"><div class="lab">' + lab + '</div><div class="val">' + txt + '</div></div>', unsafe_allow_html=True)
        if not feat.empty:
            h = feat.sort_values("time").tail(72)
            fig = px.line(h, x="time", y="us_aqi", title="AQI, last 72 hours")
            fig.update_traces(line_color=TEAL, line_width=3)
            st.plotly_chart(style(fig), use_container_width=True, config=CFG)

with tab2:
    if reports.empty:
        st.info("No report yet. Reports are written daily and when an alert fires.")
    else:
        r = reports.sort_values("ts").iloc[-1]
        st.subheader("Latest weather report")
        st.markdown('<div class="news">' + str(r["report"]) + '</div>', unsafe_allow_html=True)
        if str(r.get("alert", "")).strip():
            st.subheader("Active alert")
            st.markdown('<div class="alert">' + str(r["alert"]) + '</div>', unsafe_allow_html=True)

with tab3:
    st.subheader("Agent call log")
    if calls.empty:
        st.info("No LLM calls logged yet.")
    else:
        c = calls.sort_values("ts", ascending=False)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total LLM calls", len(c))
        failed = c[c["status"].astype(str).str.startswith("failed")]
        m2.metric("Failed calls", len(failed))
        m3.metric("Avg latency (s)", round(float(c["latency_s"].mean()), 2))
        st.dataframe(c[["ts", "agent", "status", "latency_s", "fallback_used"]].head(50), use_container_width=True)
        st.subheader("Failure log")
        if failed.empty:
            st.success("No failed calls.")
        else:
            st.dataframe(failed[["ts", "agent", "status"]].head(50), use_container_width=True)
    st.subheader("Drift events")
    if drift.empty:
        st.info("No drift events yet.")
    else:
        d = drift.sort_values("ts", ascending=False)
        st.dataframe(d[["ts", "metric", "drift_type", "score", "threshold", "action_taken"]].head(60), use_container_width=True)
    st.subheader("Forecast ledger")
    if not ledger.empty:
        st.dataframe(ledger.sort_values("ts", ascending=False)[["ts", "forecast_for", "metric", "forecast"]].head(60), use_container_width=True)

with tab4:
    if feat.empty:
        st.info("No data.")
    else:
        f = feat.sort_values("time")
        opts = {"AQI": "us_aqi", "Temperature (C)": "temperature_2m", "Humidity (%)": "relative_humidity_2m",
                "UV index": "uv_index", "Wind speed (km/h)": "wind_speed_10m", "Precipitation (mm)": "precipitation"}
        pick = st.selectbox("Metric", list(opts.keys()))
        col = opts[pick]
        daily = f.set_index("time")[col].resample("D").mean().reset_index()
        fig = px.line(daily.tail(365), x="time", y=col, title=pick + ": daily average")
        fig.update_traces(line_color=TEAL)
        st.plotly_chart(style(fig), use_container_width=True, config=CFG)
        a, b = st.columns(2)
        fig2 = px.histogram(f, x=col, nbins=40, title="Distribution")
        fig2.update_traces(marker_color=BLUE)
        a.plotly_chart(style(fig2, 320), use_container_width=True, config=CFG)
        hh = f.groupby("hour")[col].mean().reset_index()
        fig3 = px.bar(hh, x="hour", y=col, title="Average by hour of day")
        fig3.update_traces(marker_color=TEAL)
        b.plotly_chart(style(fig3, 320), use_container_width=True, config=CFG)
        mm = f.groupby("month")[col].mean().reset_index()
        fig4 = px.bar(mm, x="month", y=col, title="Average by month")
        fig4.update_traces(marker_color=NAVY)
        st.plotly_chart(style(fig4, 320), use_container_width=True, config=CFG)
        st.caption("Summary statistics")
        st.dataframe(f[col].describe().round(2).to_frame().T, use_container_width=True)

with tab5:
    st.subheader("Model comparison (test set)")
    if scores.empty:
        st.info("No scores yet.")
    else:
        s = scores.sort_values("run_time").drop_duplicates("key", keep="last")
        st.dataframe(s[["target", "model", "test_mae", "test_rmse", "test_r2"]].sort_values(["target", "test_rmse"]), use_container_width=True)
    st.subheader("SHAP: top features per metric")
    if shap_df.empty:
        st.info("No SHAP data yet.")
    else:
        t = st.selectbox("Metric", sorted(shap_df["target"].unique()))
        sd = shap_df[shap_df["target"] == t].sort_values("mean_abs_shap")
        fig = px.bar(sd, x="mean_abs_shap", y="feature", orientation="h", title="Mean absolute SHAP value")
        fig.update_traces(marker_color=TEAL)
        st.plotly_chart(style(fig, 420), use_container_width=True, config=CFG)

st.markdown('<div class="footer-custom">Developed by <b>' + YOUR_NAME + '</b> | Data Science Intern | 2026</div>', unsafe_allow_html=True)
