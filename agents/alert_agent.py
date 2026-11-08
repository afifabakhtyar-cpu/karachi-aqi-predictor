from agents.common import call_llm

def check_hazards(f):
    h = []
    if f["aqi"] > 150:
        h.append(("AQI", "Unhealthy air quality (AQI %d)" % round(f["aqi"])))
    elif f["aqi"] > 100:
        h.append(("AQI", "Air unhealthy for sensitive groups (AQI %d)" % round(f["aqi"])))
    if f["temperature_c"] >= 40:
        h.append(("Temperature", "Extreme heat (%.1f C)" % f["temperature_c"]))
    elif f["temperature_c"] <= 5:
        h.append(("Temperature", "Cold conditions (%.1f C)" % f["temperature_c"]))
    if f["uv_index"] >= 8:
        h.append(("UV", "Very high UV index (%.1f)" % f["uv_index"]))
    if f["wind_speed_kmh"] >= 40:
        h.append(("Wind", "Strong winds (%.0f km/h)" % f["wind_speed_kmh"]))
    if f["precipitation_mm"] >= 10 or f["rain_probability_pct"] >= 80:
        h.append(("Rain", "Heavy rain likely (%.1f mm, %d%% chance)" % (f["precipitation_mm"], round(f["rain_probability_pct"]))))
    if f["humidity_pct"] >= 90:
        h.append(("Humidity", "Very high humidity (%d%%)" % round(f["humidity_pct"])))
    return h

def run_alert(forecast):
    hazards = check_hazards(forecast)
    if not hazards:
        return [], ""
    lines = "; ".join(m for _, m in hazards)
    fallback = ("ALERT for Karachi in the next 24h: " + lines + ".\n"
                "General public: limit unnecessary outdoor time.\n"
                "Sensitive groups (children, elderly, asthma): stay indoors where possible.\n"
                "Outdoor workers: take frequent breaks, use protection and stay hydrated.")
    prompt = ("You are a weather alert writer for Karachi, Pakistan. Hazards forecast for the next 24 hours: " + lines +
              ". Write a short alert with exactly three labelled parts: General public, Sensitive groups, Outdoor workers. "
              "Max 2 sentences each. Plain English, no markdown.")
    text, _ = call_llm("alert", prompt, fallback)
    return hazards, text
