import numpy as np, pandas as pd
from agents.common import get_fs, get_project, log_rows, TARGETS

DATA_FEATURES = ["us_aqi", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "uv_index", "pm2_5"]
DATA_THRESHOLD = 0.25   # PSI
MODEL_THRESHOLD = 1.5   # recent RMSE / validation RMSE

def psi(a, b, bins=10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    pa = np.histogram(a, edges)[0] / len(a) + 1e-4
    pb = np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0] / len(b) + 1e-4
    return float(np.sum((pb - pa) * np.log(pb / pa)))

def run_drift():
    fs = get_fs()
    df = fs.get_feature_group("karachi_features", version=1).read().sort_values("time").reset_index(drop=True)
    events = []
    live = df.tail(24 * 7)
    train = df.iloc[: int(len(df) * 0.7)]
    for c in DATA_FEATURES:
        s = psi(train[c], live[c])
        action = "retrain requested" if s > DATA_THRESHOLD else "none"
        events.append({"metric": c, "drift_type": "data", "score": s, "threshold": DATA_THRESHOLD, "action_taken": action})

    mr = get_project().get_model_registry()
    labeled = df.dropna(subset=["y_aqi"]).tail(24 * 7)
    pairs = {"y_aqi": "us_aqi", "y_temp": "temperature_2m", "y_humidity": "relative_humidity_2m", "y_wind_speed": "wind_speed_10m", "y_uv": "uv_index"}
    import joblib, os
    for t, base in pairs.items():
        try:
            m = mr.get_best_model("aw_" + t, "val_rmse", "min")
            val_rmse = float(m.training_metrics["val_rmse"])
            bundle = joblib.load(os.path.join(m.download(), "model.pkl"))
            if bundle["kind"] != "rf":
                continue
            pred = bundle["model"].predict(labeled[bundle["features"]])
            rmse = float(np.sqrt(np.mean((labeled[t].values - pred) ** 2)))
            ratio = rmse / max(val_rmse, 1e-6)
            action = "retrain requested" if ratio > MODEL_THRESHOLD else "none"
            events.append({"metric": t, "drift_type": "model", "score": ratio, "threshold": MODEL_THRESHOLD, "action_taken": action})
        except Exception as e:
            print("model drift skipped", t, e)

    out = pd.DataFrame(events)
    log_rows("drift_log", out, ["metric", "drift_type"], "Drift events: time, metric, type, score, threshold, action")
    need_retrain = bool((out["action_taken"] == "retrain requested").any())
    return out, need_retrain
