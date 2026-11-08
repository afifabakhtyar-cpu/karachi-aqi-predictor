import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ====== WRITE YOUR NAME HERE ======
YOUR_NAME = "Afifa Bakhtyar"
# ==================================

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")

LAT, LON = 24.8607, 67.0011
TZ = "Asia/Karachi"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
W_FORECAST = "https://api.open-meteo.com/v1/forecast"
AQ_VARS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
W_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
TEAL = "#0d9488"
NAVY = "#0b2545"

# ---------- Styling ----------
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #0b2545 0%, #134074 45%, #0d9488 100%);
    padding: 2.2rem 1.5rem; border-radius: 20px; text-align: center;
    color: white; margin-bottom: 1.3rem;
    box-shadow: 0 8px 24px rgba(11,37,69,.25);
}
.hero h1 { margin: 0; font-size: 2.3rem; color: white; }
.hero p { margin: .5rem 0 0 0; opacity: .9; font-size: 1.05rem; }
.card {
    padding: 1.1rem; border-radius: 16px; text-align: center; color: white;
    box-shadow: 0 6px 16px rgba(11,37,69,.20); margin-bottom: .8rem;
}
.card .day { font-size: .95rem; opacity: .95; }
.card .val { font-size: 2.5rem; font-weight: 700; line-height: 1.2; }
.card .cat { font-size: .85rem; }
.advice {
    padding: 1rem 1.2rem; border-radius: 12px; background: white;
    border-left: 6px solid; margin-top: .5rem; color: #0b2545;
    box-shadow: 0 2px 8px rgba(11,37,69,.10);
}
.footer {
    text-align: center; padding: 1.5rem 0 .5rem 0; margin-top: 2rem;
    border-top: 2px solid #0d9488; color: #0b2545;
}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
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

def info(aqi):
    if aqi <= 50:
        return "Good", "#2e9e5b", "Air quality is good. Outdoor activities are perfectly fine."
    if aqi <= 100:
        return "Moderate", "#d4a017", "Air quality is acceptable, but unusually sensitive people (e.g. asthma) should take care."
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#e67e22", "Children, elderly and people with breathing problems should limit time outdoors."
    if aqi <= 200:
        return "Unhealthy", "#d63031", "Everyone should reduce heavy outdoor activity. Wear a mask outside."
    if aqi <= 300:
        return "Very Unhealthy", "#8e44ad", "Avoid going out unless necessary. Wear an N95 mask."
    return "Hazardous", "#2d3436", "Stay indoors and keep windows closed."

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
    rec = make_features(rec)
    hist = rec.copy()
    rec = rec.dropna(subset=features)
    latest = rec.iloc[[-1]]
    pred = model.predict(latest[features])[0]
    base = latest["time"].iloc[0]
    current = float(latest["us_aqi"].iloc[0])
    rows = [{"Date": (base + pd.Timedelta(hours=h)).strftime("%a, %d %b"),
             "AQI": round(float(p))} for h, p in zip([24, 48, 72], pred)]
    return base, current, pd.DataFrame(rows), hist, latest.iloc[0]

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
days = st.sidebar.slider("History (days)", 1, 7, 7)
show_bands = st.sidebar.checkbox("Show AQI colour bands", True)
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption("Data: Open-Meteo API")
st.sidebar.caption("Model: Ridge Regression (multi-output)")
st.sidebar.caption("Built by " + YOUR_NAME)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>🌫️ Karachi AQI Predictor</h1>
  <p>3-day air quality forecast powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

try:
    base, current, fc, hist, latest = get_forecast()
    label, color, advice = info(current)

    tab1, tab2, tab3, tab4 = st.tabs(["📅 Forecast", "📈 History", "🧪 Pollutants", "ℹ️ About"])

    # ----- Tab 1: Forecast -----
    with tab1:
        left, right = st.columns([1, 1.4])
        with left:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=round(current),
                title={"text": f"Current AQI: {label}"},
                gauge={"axis": {"range": [0, 300]}, "bar": {"color": color},
                       "steps": [{"range": [0, 50], "color": "#c8f0d6"},
                                 {"range": [50, 100], "color": "#fff3c4"},
                                 {"range": [100, 150], "color": "#ffe0b8"},
                                 {"range": [150, 200], "color": "#ffc9c9"},
                                 {"range": [200, 300], "color": "#e3c9f0"}]}))
            gauge.update_layout(height=300, margin=dict(t=60, b=10, l=20, r=20),
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge, use_container_width=True)
        with right:
            st.subheader("Next 3 days")
            cols = st.columns(3)
            for col, (_, row) in zip(cols, fc.iterrows()):
                l, c, _ = info(row["AQI"])
                col.markdown(
                    f'<div class="card" style="background:{c}">'
                    f'<div class="day">{row["Date"]}</div>'
                    f'<div class="val">{row["AQI"]}</div>'
                    f'<div class="cat">{l}</div></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="advice" style="border-color:{color}">'
                f'💡 <b>Health advice:</b> {advice}</div>', unsafe_allow_html=True)

        colors = [info(a)[1] for a in fc["AQI"]]
        bar = go.Figure(go.Bar(x=fc["Date"], y=fc["AQI"], marker_color=colors,
                               text=fc["AQI"], textposition="outside"))
        bar.update_layout(title="Forecast chart", yaxis_title="AQI", height=320,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar, use_container_width=True)

    # ----- Tab 2: History -----
    with tab2:
        cutoff = hist["time"].max() - pd.Timedelta(days=days)
        h = hist[hist["time"] >= cutoff]
        fig = px.line(h, x="time", y="us_aqi", title=f"AQI over the last {days} days")
        fig.update_traces(line_color=TEAL, line_width=3)
        if show_bands:
            for lo, hi, c in [(0, 50, "green"), (50, 100, "yellow"),
                              (100, 150, "orange"), (150, 200, "red")]:
                fig.add_hrect(y0=lo, y1=hi, fillcolor=c, opacity=0.10, line_width=0)
        fig.update_layout(height=420, xaxis_title="Time", yaxis_title="AQI",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Average AQI", round(h["us_aqi"].mean()))
        c2.metric("Highest AQI", round(h["us_aqi"].max()))
        c3.metric("Lowest AQI", round(h["us_aqi"].min()))

    # ----- Tab 3: Pollutants -----
    with tab3:
        opts = {"PM2.5": "pm2_5", "PM10": "pm10", "Carbon Monoxide": "carbon_monoxide",
                "Nitrogen Dioxide": "nitrogen_dioxide", "Sulphur Dioxide": "sulphur_dioxide",
                "Ozone": "ozone", "Temperature": "temperature_2m",
                "Humidity": "relative_humidity_2m", "Wind speed": "wind_speed_10m"}
        pick = st.multiselect("What do you want to see?", list(opts.keys()),
                              default=["PM2.5", "PM10"])
        if pick:
            cutoff = hist["time"].max() - pd.Timedelta(days=days)
            h = hist[hist["time"] >= cutoff]
            fig2 = px.line(h, x="time", y=[opts[p] for p in pick],
                           color_discrete_sequence=[TEAL, "#134074", "#e67e22",
                                                    "#8e44ad", "#d63031", "#2e9e5b"])
            fig2.update_layout(height=420, legend_title="",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Please select at least one option above.")

    # ----- Tab 4: About -----
    with tab4:
        st.markdown(f"""
### About this project
This app predicts Karachi's **US AQI** for the next 24, 48 and 72 hours.

**Process:**
1. Hourly pollution and weather data collected from the Open-Meteo API (from Jan 2024)
2. Data cleaning and feature engineering (lag values, rolling averages, hour/month)
3. Ridge and Random Forest models compared using MAE and RMSE
4. Best model (Ridge, MAE ≈ 8.6) saved and deployed on Streamlit

**Tools:** Python, Pandas, Scikit-learn, Plotly, Streamlit, GitHub

**Latest data time:** {base}
""")

except Exception as e:
    st.error(f"Something went wrong: {e}")

# ---------- Footer ----------
st.markdown(
    f'<div class="footer">Developed by <b>{YOUR_NAME}</b> | Data Science Intern | 2026</div>',
    unsafe_allow_html=True)
