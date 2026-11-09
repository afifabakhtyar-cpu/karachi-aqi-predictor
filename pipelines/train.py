import os, warnings
import numpy as np, pandas as pd, joblib, torch, torch.nn as nn, shap
import hopsworks
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

key = "".join(os.environ["HOPSWORKS_API_KEY"].split())
project = hopsworks.login(api_key_value=key, project="AgentWeather")
fs = project.get_feature_store()
mr = project.get_model_registry()

df = fs.get_feature_group("karachi_features", version=1).read()
df = df.sort_values("time").reset_index(drop=True)
T = {"y_aqi": "us_aqi", "y_temp": "temperature_2m", "y_humidity": "relative_humidity_2m", "y_uv": "uv_index", "y_wind_speed": "wind_speed_10m", "y_wind_sin": "wind_sin", "y_wind_cos": "wind_cos", "y_precip": "precip_last24"}
tcols = list(T.keys())
FEATS = [c for c in df.columns if not c.startswith("y_") and c not in ["time", "ts"]]
lab = df.dropna(subset=tcols + ["y_rain"]).reset_index(drop=True)
n = len(lab)
i1 = int(n * 0.70)
i2 = int(n * 0.85)
tr = lab.iloc[:i1 - 24]
va = lab.iloc[i1:i2 - 24]
te = lab.iloc[i2:]
print("Rows:", n, "train", len(tr), "val", len(va), "test", len(te))

def sc(y, p):
    return float(mean_absolute_error(y, p)), float(np.sqrt(mean_squared_error(y, p))), float(r2_score(y, p))

NONNEG = ["y_precip", "y_uv", "y_wind_speed"]
def fix(t, p):
    p = np.asarray(p, dtype=float)
    return np.clip(p, 0, None) if t in NONNEG else p

def run_sarimax(t, base):
    cols = [c for c in [base, base + "_roll24", "hour_sin", "hour_cos"] if c in lab.columns]
    mu = tr[cols].mean()
    sd = tr[cols].std().replace(0, 1)
    f = lambda d: ((d[cols] - mu) / sd).values
    try:
        k = 800
        res = SARIMAX(tr[t].values[-k:], exog=f(tr)[-k:], order=(1, 0, 0)).fit(disp=False, maxiter=30)
        pv = res.get_forecast(steps=len(va), exog=f(va)).predicted_mean
        res2 = res.append(va[t].values, exog=f(va), refit=False)
        pt = res2.get_forecast(steps=len(te), exog=f(te)).predicted_mean
        return np.asarray(pv), np.asarray(pt)
    except Exception as e:
        print("SARIMAX failed", t, e)
        return None, None

LC = ["us_aqi", "temperature_2m", "relative_humidity_2m", "precipitation", "uv_index", "wind_speed_10m", "wind_sin", "wind_cos", "pm2_5", "hour_sin", "hour_cos"]
W = 24
HID = 32
EPOCHS = 6
arr = lab[LC].values.astype("float32")
am = arr[:i1].mean(0)
asd = arr[:i1].std(0) + 1e-6
arr = (arr - am) / asd
Y = lab[tcols].values.astype("float32")
ym = Y[:i1].mean(0)
ysd = Y[:i1].std(0) + 1e-6
Ys = (Y - ym) / ysd
X = np.ascontiguousarray(sliding_window_view(arr, W, axis=0).transpose(0, 2, 1))
ridx = np.arange(W - 1, n)
Ysamp = Ys[W - 1:]
mtr = ridx < i1 - 24
mva = (ridx >= i1) & (ridx < i2 - 24)
mte = ridx >= i2

class Net(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.l = nn.LSTM(c, HID, batch_first=True)
        self.o = nn.Linear(HID, k)
    def forward(self, x):
        h, _ = self.l(x)
        return self.o(h[:, -1])

net = Net(len(LC), len(tcols))
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
Xtr = torch.tensor(X[mtr])
Ytr = torch.tensor(Ysamp[mtr])
for ep in range(EPOCHS):
    perm = torch.randperm(len(Xtr))
    for b in range(0, len(Xtr), 512):
        idx = perm[b:b + 512]
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(Xtr[idx]), Ytr[idx])
        loss.backward()
        opt.step()
    print("LSTM epoch", ep + 1)
net.eval()
with torch.no_grad():
    lstm_va = net(torch.tensor(X[mva])).numpy() * ysd + ym
    lstm_te = net(torch.tensor(X[mte])).numpy() * ysd + ym

rows, rfs, choice = [], {}, {}
for j, t in enumerate(tcols):
    base = T[t]
    P = {"Naive": (va[base].values, te[base].values)}
    sv, st_ = run_sarimax(t, base)
    if sv is not None:
        P["SARIMAX"] = (fix(t, sv), fix(t, st_))
    rf = RandomForestRegressor(n_estimators=30, max_depth=10, min_samples_leaf=30, max_features=0.5, n_jobs=-1, random_state=42)
    rf.fit(tr[FEATS], tr[t])
    rfs[t] = rf
    P["RandomForest"] = (fix(t, rf.predict(va[FEATS])), fix(t, rf.predict(te[FEATS])))
    P["LSTM"] = (fix(t, lstm_va[:, j]), fix(t, lstm_te[:, j]))
    for name, (pv, pt) in P.items():
        v = sc(va[t], pv)
        s = sc(te[t], pt)
        rows.append([t, name, v[1], s[0], s[1], s[2]])
    cand = {k: sc(va[t], P[k][0])[1] for k in ["RandomForest", "LSTM"]}
    choice[t] = min(cand, key=cand.get)
    print("done", t, choice[t])

scores = pd.DataFrame(rows, columns=["target", "model", "val_rmse", "test_mae", "test_rmse", "test_r2"]).round(3)
print(scores.to_string(index=False))

clf = RandomForestClassifier(n_estimators=30, max_depth=8, min_samples_leaf=30, max_features=0.5, n_jobs=-1, random_state=42)
clf.fit(tr[FEATS], tr["y_rain"])
rain_val_rmse = float(np.sqrt(np.mean((va["y_rain"].values - clf.predict_proba(va[FEATS])[:, 1]) ** 2)))

shap_rows = []
sample = te[FEATS].sample(min(100, len(te)), random_state=0)
for t in tcols:
    sv_ = shap.TreeExplainer(rfs[t]).shap_values(sample)
    imp = np.abs(sv_).mean(0)
    for f_, v_ in sorted(zip(FEATS, imp), key=lambda z: -z[1])[:10]:
        shap_rows.append([t + "_" + f_, t, f_, float(v_)])
shap_df = pd.DataFrame(shap_rows, columns=["key", "target", "feature", "mean_abs_shap"])

def register(name, val_rmse, folder, desc):
    old = None
    try:
        old = mr.get_best_model(name, "val_rmse", "min").training_metrics["val_rmse"]
    except Exception:
        pass
    if old is not None and val_rmse >= old:
        print("KEPT old model", name, "old", round(old, 3), "new", round(val_rmse, 3))
        return
    m = mr.python.create_model(name, metrics={"val_rmse": float(val_rmse)}, description=desc)
    m.save(folder)
    print("REGISTERED", name, round(val_rmse, 3))

os.makedirs("reg", exist_ok=True)
for t in tcols:
    folder = "reg/aw_" + t
    os.makedirs(folder, exist_ok=True)
    vr = float(scores[(scores.target == t) & (scores.model == choice[t])]["val_rmse"].iloc[0])
    if choice[t] == "RandomForest":
        joblib.dump({"kind": "rf", "model": rfs[t], "features": FEATS}, folder + "/model.pkl", compress=3)
    else:
        torch.save(net.state_dict(), folder + "/lstm.pt")
        joblib.dump({"kind": "lstm", "index": tcols.index(t), "cols": LC, "features": FEATS, "am": am, "asd": asd, "ym": ym, "ysd": ysd, "window": W, "hidden": HID, "targets": tcols}, folder + "/model.pkl")
    register("aw_" + t, vr, folder, choice[t] + " for " + t)

os.makedirs("reg/aw_rain", exist_ok=True)
joblib.dump({"kind": "rf_clf", "model": clf, "features": FEATS}, "reg/aw_rain/model.pkl", compress=3)
register("aw_rain", rain_val_rmse, "reg/aw_rain", "RandomForest rain probability")

now = pd.Timestamp.now()
sc_df = scores.copy()
sc_df["key"] = sc_df["target"] + "_" + sc_df["model"]
sc_df["run_time"] = now
fs.get_or_create_feature_group(name="model_scores", version=1, primary_key=["key"], event_time="run_time", description="Model comparison scores").insert(sc_df, write_options={"wait_for_job": False})
shap_df["run_time"] = now
fs.get_or_create_feature_group(name="shap_importance", version=1, primary_key=["key"], event_time="run_time", description="Top SHAP features").insert(shap_df, write_options={"wait_for_job": False})
print("TRAINING DONE")
