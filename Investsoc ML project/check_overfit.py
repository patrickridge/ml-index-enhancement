import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor

panel = pd.read_parquet("data/panel_monthly_enriched.parquet")
panel["date"] = pd.to_datetime(panel["date"])

exclude = {"date", "ticker", "fwd_ret_1m"}
feat_cols = [c for c in panel.columns if c not in exclude]

from config import TRAIN_END, VALID_END
TRAIN_END = pd.Timestamp(TRAIN_END)
VALID_END = pd.Timestamp(VALID_END)

train = panel[panel["date"] <= TRAIN_END].dropna(subset=feat_cols + ["fwd_ret_1m"])
valid = panel[(panel["date"] > TRAIN_END) & (panel["date"] <= VALID_END)].dropna(subset=feat_cols + ["fwd_ret_1m"])
test  = panel[panel["date"] > VALID_END].dropna(subset=feat_cols + ["fwd_ret_1m"])

from config import LGBM_PARAMS
model = LGBMRegressor(**LGBM_PARAMS)
model.fit(
    train[feat_cols], train["fwd_ret_1m"],
    eval_set=[(valid[feat_cols], valid["fwd_ret_1m"])],
    eval_metric="l2",
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
)

def long_only_sharpe(df, top_n=50):
    rets = []
    for _, g in df.groupby("date"):
        g = g.dropna(subset=["score", "fwd_ret_1m"])
        if len(g) < top_n:
            continue
        rets.append(g.nlargest(top_n, "score")["fwd_ret_1m"].mean())
    r = pd.Series(rets).dropna()
    if len(r) < 3:
        return dict(months=len(r), ann=np.nan, vol=np.nan, sharpe=np.nan)
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    return dict(months=len(r), ann=ann, vol=vol, sharpe=ann/vol if vol > 0 else np.nan)

print(f"Best iteration: {model.best_iteration_}")
print()
for name, split in [("TRAIN", train), ("VALID", valid), ("TEST", test)]:
    split = split.copy()
    split["score"] = model.predict(split[feat_cols], num_iteration=model.best_iteration_)
    s = long_only_sharpe(split)
    print(f"{name}: months={s['months']:3d} | ann={s['ann']*100:6.2f}% | vol={s['vol']*100:6.2f}% | sharpe={s['sharpe']:.2f}")
