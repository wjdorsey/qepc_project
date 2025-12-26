import pandas as pd
import numpy as np

p = r"logs\backtest_player_points_preds_2022-10-01_2024-06-22_20251226_034832.parquet"
df = pd.read_parquet(p)

pred_col = "predicted_points_raw"

# find the actual points column
for c in ["actual_points", "points_actual", "actual_pts", "pts_actual", "points", "pts"]:
    if c in df.columns:
        act_col = c
        break
else:
    raise SystemExit(f"Can't find actual points column. Columns: {sorted(df.columns)}")

date_col = "game_date" if "game_date" in df.columns else "game_datetime"
dates = pd.to_datetime(df[date_col]).dt.date.astype(str)

def subset(start, end):
    m = (dates >= start) & (dates <= end)
    return df.loc[m, [pred_col, act_col]].copy()

train = subset("2022-10-01", "2023-06-22")
valid = subset("2023-10-01", "2024-06-22")

def metrics(d, slope, intercept, clip=True):
    pred = intercept + slope * d[pred_col].to_numpy()
    if clip:
        pred = np.maximum(pred, 0.0)
    act = d[act_col].to_numpy()
    mae = float(np.mean(np.abs(pred - act)))
    bias = float(np.mean(pred - act))  # positive = over-predict
    corr = float(np.corrcoef(pred, act)[0, 1]) if len(act) > 1 else float("nan")
    return mae, bias, corr

candidates = [
    ("MAE-fit", 0.864218, 1.550546),
    ("MAE-fit + train-bias-correct", 0.864218, 1.550546 + 0.597),
    ("Manual", 0.854, 2.266),
]

print("Rows:", len(df))
print("Using columns:", pred_col, act_col, "| date:", date_col)

for name, s, b in candidates:
    t = metrics(train, s, b)
    v = metrics(valid, s, b)
    print(f"\n{name}: slope={s:.6f}, intercept={b:.6f}")
    print(f"  TRAIN: MAE={t[0]:.3f} bias={t[1]:+.3f} corr={t[2]:.3f}")
    print(f"  VALID: MAE={v[0]:.3f} bias={v[1]:+.3f} corr={v[2]:.3f}")
