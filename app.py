import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ====== APNA NAAM YAHAN LIKHEIN ======
YOUR_NAME = "Afifa Bakhtyar Rahman"
# =====================================

st.set_page_config(page_title="Air Pearls Karachi AQI Predictor", page_icon="🌫️", layout="wide")

LAT, LON = 24.8607, 67.0011
TZ = "Asia/Karachi"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
W_FORECAST = "https://api.open-meteo.com/v1/forecast"
AQ_VARS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
W_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"

# ---------- Styling ----------
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2rem 1.5rem; border-radius: 18px; text-align: center;
    color: white; margin-bottom: 1.2rem;
}
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p { margin: .4rem 0 0 0; opacity: .85; }
.card {
    padding: 1.1rem; border-radius: 16px; text-align: center; color: white;
    box-shadow: 0 4px 14px rgba(0,0,0,.25); margin-bottom: .8rem;
}
.card .day { font-size: .95rem; opacity: .9; }
.card .val { font-size: 2.4rem; font-weight: 700; line-height: 1.2; }
.card .cat { font-size: .9rem; }
.advice {
    padding: 1rem 1.2rem; border-radius: 12px; background: rgba(120,120,120,.12);
    border-left: 6px solid; margin-top: .5rem;
}
.footer {
    text-align: center; padding: 1.5rem 0 .5rem 0; margin-top: 2rem;
    border-top: 1px solid rgba(120,120,120,.3); opacity: .85;
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
        return "Good", "#2e9e5b", "Hawa saaf hai. Bahar ki activities bilkul theek hain."
    if aqi <= 100:
        return "Moderate", "#d4a017", "Hawa qabil-e-qabool hai, lekin bohat sensitive log (asthma wale) ehtiyat karein."
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#e67e22", "Bachay, buzurg aur saans ke mareez bahar kam nikalein."
    if aqi <= 200:
        return "Unhealthy", "#d63031", "Sab ko bahar ki mehnat wali activities kam karni chahiye. Mask pehnein."
    if aqi <= 300:
        return "Very Unhealthy", "#8e44ad", "Zaroori ho to hi bahar jayein, N95 mask lagayein."
    return "Hazardous", "#2d3436", "Ghar ke andar rahein, khidkiyan band rakhein."

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
show_bands = st.sidebar.checkbox("AQI colour bands dikhayein", True)
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
  <p>Machine learning se agle 3 din ki hawa ki quality ka forecast</p>
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
            gauge.update_layout(height=300, margin=dict(t=60, b=10, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True)
        with right:
            st.subheader("Agle 3 din")
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
        bar.update_layout(title="Forecast chart", yaxis_title="AQI", height=320)
        st.plotly_chart(bar, use_container_width=True)

    # ----- Tab 2: History -----
    with tab2:
        cutoff = hist["time"].max() - pd.Timedelta(days=days)
        h = hist[hist["time"] >= cutoff]
        fig = px.line(h, x="time", y="us_aqi", title=f"Pichle {days} din ka AQI")
        fig.update_traces(line_color="#2c5364")
        if show_bands:
            for lo, hi, c in [(0, 50, "green"), (50, 100, "yellow"),
                              (100, 150, "orange"), (150, 200, "red")]:
                fig.add_hrect(y0=lo, y1=hi, fillcolor=c, opacity=0.10, line_width=0)
        fig.update_layout(height=420, xaxis_title="Time", yaxis_title="AQI")
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
        pick = st.multiselect("Kya dekhna hai?", list(opts.keys()), default=["PM2.5", "PM10"])
        if pick:
            cutoff = hist["time"].max() - pd.Timedelta(days=days)
            h = hist[hist["time"] >= cutoff]
            fig2 = px.line(h, x="time", y=[opts[p] for p in pick])
            fig2.update_layout(height=420, legend_title="")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Upar se kam az kam ek cheez select karein.")

    # ----- Tab 4: About -----
    with tab4:
        st.markdown(f"""
### Project ke baare mein
Ye app Karachi ka **US AQI** agle 24, 48 aur 72 ghante ke liye predict karta hai.

**Process:**
1. Open-Meteo API se ghante wala pollution aur weather data (Jan 2024 se)
2. Data cleaning aur feature engineering (lag values, rolling averages, hour/month)
3. Ridge aur Random Forest models ki comparison (MAE aur RMSE se)
4. Best model (Ridge, MAE ≈ 8.6) ko save karke Streamlit par deploy kiya

**Tools:** Python, Pandas, Scikit-learn, Plotly, Streamlit, GitHub

**Latest data time:** {base}
""")

except Exception as e:
    st.error(f"Something went wrong: {e}")

# ---------- Footer ----------
st.markdown(
    f'<div class="footer">Developed by <b>{YOUR_NAME}</b> | Data Science Intern, 10Pearls | 2026</div>',
    unsafe_allow_html=True)
