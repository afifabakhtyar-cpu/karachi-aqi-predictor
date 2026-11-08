from agents.common import call_llm

def run_report(base_time, forecast, alert_text):
    facts = ("AQI %d, temperature %.1f C, humidity %d%%, UV index %.1f, wind %.0f km/h from %d degrees, "
             "rain probability %d%% with %.1f mm expected." % (
                 round(forecast["aqi"]), forecast["temperature_c"], round(forecast["humidity_pct"]),
                 forecast["uv_index"], forecast["wind_speed_kmh"], round(forecast["wind_direction_deg"]),
                 round(forecast["rain_probability_pct"]), forecast["precipitation_mm"]))
    fallback = ("Karachi weather report. Over the next 24 hours expect " + facts +
                (" " + alert_text if alert_text else " No weather hazards are expected."))
    prompt = ("Write a short news-style weather report (about 120 words) for Karachi for the next 24 hours, "
              "starting with a headline. Use only these facts: " + facts +
              (" Active alert: " + alert_text if alert_text else " No hazards.") + " Plain text, no markdown.")
    text, _ = call_llm("report", prompt, fallback)
    return text
