import numpy as np, pandas as pd, torch, torch.nn as nn, os
from agents.common import get_fs, load_model, TARGETS

def _lstm_predict(bundle, folder, recent):
    cols, W, H = bundle["cols"], bundle["window"], bundle["hidden"]
    class Net(nn.Module):
        def __init__(self, c, k):
            super().__init__()
            self.l = nn.LSTM(c, H, batch_first=True)
            self.o = nn.Linear(H, k)
        def forward(self, x):
            h, _ = self.l(x)
            return self.o(h[:, -1])
    net = Net(len(cols), len(bundle["targets"]))
    net.load_state_dict(torch.load(os.path.join(folder, "lstm.pt"), map_location="cpu"))
    net.eval()
    arr = recent[cols].values[-W:].astype("float32")
    arr = (arr - bundle["am"]) / bundle["asd"]
    with torch.no_grad():
        out = net(torch.tensor(arr[None])).numpy()[0] * bundle["ysd"] + bundle["ym"]
    return float(out[bundle["index"]])

def run_forecast():
    fs = get_fs()
    df = fs.get_feature_group("karachi_features", version=1).read().sort_values("time").reset_index(drop=True)
    latest = df.iloc[[-1]]
    base_time = pd.Timestamp(latest["time"].iloc[0])
    result = {}
    for t in TARGETS:
        bundle, folder = load_model("aw_" + t)
        if bundle["kind"] == "rf":
            val = float(bundle["model"].predict(latest[bundle["features"]])[0])
        else:
            val = _lstm_predict(bundle, folder, df)
        result[t] = val
    bundle, _ = load_model("aw_rain")
    result["rain_prob"] = float(bundle["model"].predict_proba(latest[bundle["features"]])[0, 1])

    ang = (np.degrees(np.arctan2(result["y_wind_sin"], result["y_wind_cos"])) + 360) % 360
    forecast = {
        "aqi": max(0.0, result["y_aqi"]),
        "temperature_c": result["y_temp"],
        "humidity_pct": float(min(max(result["y_humidity"], 0), 100)),
        "uv_index": max(0.0, result["y_uv"]),
        "wind_speed_kmh": max(0.0, result["y_wind_speed"]),
        "wind_direction_deg": float(ang),
        "precipitation_mm": max(0.0, result["y_precip"]),
        "rain_probability_pct": float(result["rain_prob"] * 100),
    }
    return base_time, forecast, latest
