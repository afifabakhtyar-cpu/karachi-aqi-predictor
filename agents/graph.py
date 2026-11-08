import json, sys
import pandas as pd
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from agents.common import get_fs, log_rows
from agents.forecast_agent import run_forecast
from agents.alert_agent import run_alert
from agents.report_agent import run_report
from agents.drift_agent import run_drift

class State(TypedDict, total=False):
    base_time: Any
    forecast: dict
    hazards: list
    alert_text: str
    report: str
    drift: Any
    retrain: bool
    daily: bool

def forecast_node(s):
    base_time, forecast, _ = run_forecast()
    rows = [{"forecast_for": str(base_time + pd.Timedelta(hours=24)), "metric": k, "forecast": float(v)} for k, v in forecast.items()]
    log_rows("forecast_ledger", pd.DataFrame(rows), ["metric"], "Forecast Ledger: every forecast made by the model")
    return {"base_time": base_time, "forecast": forecast}

def alert_node(s):
    hazards, text = run_alert(s["forecast"])
    return {"hazards": hazards, "alert_text": text}

def report_node(s):
    text = run_report(s["base_time"], s["forecast"], s.get("alert_text", ""))
    rep = pd.DataFrame([{"for_time": str(s["base_time"]), "report": text, "alert": s.get("alert_text", ""), "forecast_json": json.dumps(s["forecast"])}])
    log_rows("weather_reports", rep, ["for_time"], "News-style weather reports and alerts")
    return {"report": text}

def drift_node(s):
    out, retrain = run_drift()
    return {"drift": out, "retrain": retrain}

def route_after_alert(s):
    # Report agent: runs daily, and after a new alert
    if s.get("daily") or s.get("hazards"):
        return "report"
    return "drift"

g = StateGraph(State)
g.add_node("forecast", forecast_node)
g.add_node("alert", alert_node)
g.add_node("report", report_node)
g.add_node("drift", drift_node)
g.set_entry_point("forecast")
g.add_edge("forecast", "alert")
g.add_conditional_edges("alert", route_after_alert, {"report": "report", "drift": "drift"})
g.add_edge("report", "drift")
g.add_edge("drift", END)
app = g.compile()

if __name__ == "__main__":
    daily = "--daily" in sys.argv
    final = app.invoke({"daily": daily})
    print("Forecast:", json.dumps(final["forecast"], indent=1))
    print("Hazards:", final.get("hazards"))
    print("Retrain needed:", final.get("retrain"))
    open("retrain_flag.txt", "w").write("1" if final.get("retrain") else "0")
