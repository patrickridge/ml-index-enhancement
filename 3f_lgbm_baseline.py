"""
3f_lgbm_baseline.py - Reproducible LightGBM baseline on the same protocol as 3c.

The LightGBM numbers quoted in the README came from data/scores_lgbm.parquet,
a file dated March with no script in the repo that produces it. lightgbm was
listed in requirements.txt but nothing imported it. That baseline could not be
regenerated, which matters because it is the comparison the transformer results
are measured against.

It was also not measured on equal terms. Four problems, all pointing the same
way:

  1. Training hygiene. 3c applies the index weight floor before training, so it
     never sees delisted names. The old LGBM scores predate that fix, so they
     were produced by a model trained on the same zombie tickers the audit was
     about. Contamination inflates apparent performance, so the dirty model had
     a tailwind the clean one did not.

  2. Sample window. Old LGBM covers 2023-01 to 2025-11 (35 months). Clean CS-T
     covers 2024-07 to 2025-11 (17 months). The CS-T window sits entirely
     inside the LGBM window and is the harder half of it: mega-cap
     concentration through 2024-25 punishes anything tilting away from cap
     weights, and LGBM got the 2023 recovery as well.

  3. Date keys. The old file stamps rows with the last trading day, not the
     calendar month end, so merging it against spx_weights.parquet silently
     drops most months. Its quoted IR came from 18 of its 35 months.

  4. Feature set and splits. Unknown, because the script is gone.

This reruns LightGBM under 3c's exact protocol: same panel, same weight floor
and return gate, same expanding-window walk-forward, same retrain cadence, same
output schema, calendar month-end keys throughout. The only thing that differs
between the two score files is the model. Honours ML_OOS_START the same way, so
once the full-history transformer run lands, both sit on the same months.

What it found, on the legacy window with hyperparameters held fixed:

    floor off   IC +0.0450  ICIR +1.273      legacy file  IC +0.0441  ICIR +1.264
    floor on    IC -0.0191  ICIR -0.505

Reproducing the legacy numbers to three decimals by disabling one filter is
about as direct as evidence gets that the filter is the whole story. The
portfolio IR goes from +0.56 to -1.86 across that same switch. Notes item 24
has the full write-up, including why the -1.86 is more negative than the IC
alone would justify.

Writes to scores_lgbm_clean.parquet rather than overwriting the legacy file, so
the old and new numbers can be compared before anything is replaced.

Usage:
    python 3f_lgbm_baseline.py                          # 17-month window, matches default 3c
    ML_OOS_START=2013-12-31 python 3f_lgbm_baseline.py  # full history, matches Job B
    ML_OOS_START=2022-12-31 python 3f_lgbm_baseline.py  # legacy window, for the comparison above
    ML_SKIP_WEIGHT_FLOOR=1 python 3f_lgbm_baseline.py   # the audit arm, writes elsewhere

Reads:  data/panel_monthly_enriched.parquet, data/spx_weights.parquet
Writes: data/scores_lgbm_clean.parquet
        data/scores_lgbm_zombie_audit.parquet  (only with ML_SKIP_WEIGHT_FLOOR)
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from scipy import stats as sp_stats

from config import (
    DATA_DIR, START_DATE, VALID_END,
    MIN_SPX_WEIGHT, MAX_ABS_MONTHLY_RET,
    RETRAIN_EVERY, USE_ORTHOGONALIZED_FEATURES, MACRO_COLS,
    drop_dead_features,
)

_panel_file = ("panel_monthly_orthogonalized.parquet" if USE_ORTHOGONALIZED_FEATURES
               else "panel_monthly_enriched.parquet")
PANEL_IN = DATA_DIR / _panel_file

# The audit run writes somewhere else on purpose. A contaminated file sitting
# under a name that says "clean" is how the original problem survived as long
# as it did, and the fix is not to rely on remembering which is which.
OUT_SCORES = (DATA_DIR / "scores_lgbm_zombie_audit.parquet"
              if os.environ.get("ML_SKIP_WEIGHT_FLOOR")
              else DATA_DIR / "scores_lgbm_clean.parquet")

# Deliberately unremarkable. The point of this script is a fair reference
# model, not a tuned competitor, and tuning it would reintroduce the selection
# problem the rest of the project is trying to avoid.
LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "l2",
    "learning_rate":     0.03,
    "num_leaves":        31,
    "min_data_in_leaf":  200,
    "feature_fraction":  0.7,
    "bagging_fraction":  0.8,
    "bagging_freq":      1,
    "lambda_l2":         1.0,
    "verbosity":         -1,
    "num_threads":       0,
}
N_ROUNDS   = 2000
EARLY_STOP = 100

# Cross-sectional percentile target. 3c trains on the raw return under masked
# MSE, so "raw" looks like the like-for-like choice, but it is degenerate here:
# the squared-error objective spends its capacity on the market component that
# every stock in a month shares and none of the features predict, so boosting
# early-stops after one round and the model returns a near-constant score. Only
# 3 of 17 test months had a defined Spearman IC. Ranking within the month
# removes the common component and leaves the cross-sectional problem the model
# is actually being asked to solve, which is why it is the default. "raw" is
# kept so the failure is reproducible rather than just asserted.
TARGET_MODE = os.environ.get("ML_LGBM_TARGET", "rank")

# "ic" stops on validation rank IC, "l2" on squared error. See _make_ic_eval.
EARLY_STOP_METRIC = os.environ.get("ML_LGBM_STOP", "ic")


def load_panel():
    """Panel load plus the zombie filter, copied in behaviour from 3c."""
    panel = pd.read_parquet(PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= START_DATE].reset_index(drop=True)

    # Audit switch. Setting this reproduces the pre-fix behaviour, which is the
    # only way to attribute a performance gap to the filter rather than to
    # hyperparameters: run the identical script twice, floor on and floor off.
    # It exists to be measured, not used.
    if os.environ.get("ML_SKIP_WEIGHT_FLOOR"):
        print("[AUDIT] Weight floor disabled. Delisted names are in the "
              "training set. Results are not valid for anything but the A/B.")
        return panel.reset_index(drop=True)

    spx_path = DATA_DIR / "spx_weights.parquet"
    if not spx_path.exists():
        raise SystemExit(f"{spx_path} not found. The weight floor is the whole "
                         "point of this script, so refusing to run without it.")

    spx_w = pd.read_parquet(spx_path)
    spx_w["date"] = pd.to_datetime(spx_w["date"]) + pd.offsets.MonthEnd(0)
    panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)

    spx_w = spx_w[spx_w["spx_weight"] >= MIN_SPX_WEIGHT]
    valid_keys = set(zip(spx_w["date"], spx_w["ticker"]))
    pre_rows = len(panel)
    panel = panel[[k in valid_keys for k in zip(panel["date"], panel["ticker"])]]
    panel = panel.reset_index(drop=True)

    bad = panel["fwd_ret_1m"].abs() > MAX_ABS_MONTHLY_RET
    if bad.any():
        print(f"Return gate: dropping {int(bad.sum()):,} rows with "
              f"|fwd_ret_1m| > {MAX_ABS_MONTHLY_RET:.0%}")
        panel = panel[~bad].reset_index(drop=True)

    print(f"Zombie filter: {pre_rows:,} -> {len(panel):,} rows "
          f"({pre_rows - len(panel):,} dropped, weight floor {MIN_SPX_WEIGHT:.0e})")
    return panel


def make_target(df):
    """Raw forward return, or its within-month percentile if asked for."""
    if TARGET_MODE == "rank":
        return df.groupby("date")["fwd_ret_1m"].transform(
            lambda s: s.rank(pct=True) - 0.5)
    return df["fwd_ret_1m"]


def _make_ic_eval(valid_dates):
    """Early-stopping metric: mean Spearman IC across validation months.

    The default l2 metric stops after one or two rounds here. That is not
    LightGBM being cautious, it is the metric measuring the wrong thing: most
    of the variance in a monthly return is the market move common to every
    stock, which no cross-sectional feature can predict, so l2 is dominated by
    a term the model cannot touch and looks like it is getting worse while the
    ranking is still improving. Scoring the ranking directly stops on the
    quantity the portfolio is built from.

    Chosen because l2 is mis-specified for a cross-sectional ranking problem,
    not because of what it did to the result. Both are reported.
    """
    months = pd.Series(valid_dates).values

    def feval(preds, dset):
        y = dset.get_label()
        df = pd.DataFrame({"m": months, "p": preds, "y": y})
        ics = df.groupby("m").apply(
            lambda g: sp_stats.spearmanr(g["p"], g["y"])[0]
            if g["p"].nunique() > 1 and len(g) > 5 else np.nan)
        return "rank_ic", float(np.nanmean(ics)) if len(ics) else 0.0, True

    return feval


def fit(train_df, valid_df, feat_cols):
    dtrain = lgb.Dataset(train_df[feat_cols], label=make_target(train_df))
    dvalid = lgb.Dataset(valid_df[feat_cols], label=make_target(valid_df),
                         reference=dtrain)
    params = dict(LGBM_PARAMS)
    feval = None
    if EARLY_STOP_METRIC == "ic":
        params["metric"] = "None"
        feval = _make_ic_eval(valid_df["date"].values)
    return lgb.train(
        params, dtrain, num_boost_round=N_ROUNDS,
        valid_sets=[dvalid], feval=feval,
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )


def _ic(g):
    """Spearman IC for one month. Undefined if the model scored everything the
    same, which happens when the fit collapses, so that is reported as NaN
    rather than silently dropped."""
    if len(g) <= 5 or g["score"].nunique() < 2:
        return np.nan
    return sp_stats.spearmanr(g["score"], g["fwd_ret_1m"])[0]


def monthly_ic(df):
    """Mean Spearman IC and annualised ICIR, plus how many months were usable."""
    ics = df.groupby("date")[["score", "fwd_ret_1m"]].apply(_ic).dropna()
    if ics.empty:
        return np.nan, np.nan, 0
    icir = ics.mean() / ics.std() * np.sqrt(12) if ics.std() > 0 else np.nan
    return ics.mean(), icir, len(ics)


def main():
    print(f"Panel:  {PANEL_IN.name}")
    print(f"Target: {TARGET_MODE} | early stop on: {EARLY_STOP_METRIC}")
    panel = load_panel()

    exclude = {"date", "ticker", "fwd_ret_1m"}
    macro_cols = [c for c in MACRO_COLS if c in panel.columns]
    feat_cols = [c for c in panel.columns if c not in exclude]

    # Macro columns are constant across stocks by definition, so they are held
    # out of the check that would otherwise flag every one of them.
    stock_cols = drop_dead_features(
        panel, [c for c in feat_cols if c not in macro_cols], label="stock")
    feat_cols = stock_cols + macro_cols

    print(f"Features: {len(feat_cols)} ({len(stock_cols)} stock, "
          f"{len(macro_cols)} macro) | Rows: {len(panel):,} | "
          f"Tickers: {panel['ticker'].nunique()}")

    # Same split arithmetic as 3c so the two runs line up month for month.
    months    = sorted(panel["date"].unique())
    oos_start = pd.Timestamp(os.environ.get("ML_OOS_START", VALID_END))

    pre_oos = [m for m in months if m <= oos_start]
    if len(pre_oos) < 24:
        raise SystemExit(f"Only {len(pre_oos)} months before {oos_start.date()}; "
                         "need at least 24.")

    n_val        = min(18, max(6, len(pre_oos) // 5))
    train_months = pre_oos[:-n_val]
    valid_months = pre_oos[-n_val:]
    test_months  = [m for m in months if m > oos_start]

    if not test_months:
        raise SystemExit(f"No months after {oos_start.date()} to score.")

    print(f"OOS starts after: {oos_start.date()}")
    print(f"Train: {train_months[0].date()} -> {train_months[-1].date()} "
          f"({len(train_months)} months)")
    print(f"Valid: {valid_months[0].date()} -> {valid_months[-1].date()} "
          f"({len(valid_months)} months)")
    print(f"Test:  {test_months[0].date()} -> {test_months[-1].date()} "
          f"({len(test_months)} months)")

    panel = panel.dropna(subset=["fwd_ret_1m"]).reset_index(drop=True)
    by_month = {m: g for m, g in panel.groupby("date")}

    model = fit(panel[panel["date"].isin(train_months)],
                panel[panel["date"].isin(valid_months)], feat_cols)
    print(f"\nInitial fit: {model.best_iteration} rounds")

    all_scores = []
    for i, m in enumerate(test_months):
        # Expanding-window retrain on the same cadence as 3c. Every month is
        # scored by a model fit only on months strictly before it.
        if i > 0 and i % RETRAIN_EVERY == 0:
            seen = [mo for mo in months if mo < m]
            rl_val_n = min(18, max(6, len(seen) // 5))
            tr = panel[panel["date"].isin(seen[:-rl_val_n])]
            va = panel[panel["date"].isin(seen[-rl_val_n:])]
            if len(va):
                model = fit(tr, va, feat_cols)
                print(f"  Retraining at {m.date()} (train={len(seen) - rl_val_n}, "
                      f"val={rl_val_n} months) -> {model.best_iteration} rounds")

        if m not in by_month:
            continue
        sub = by_month[m].copy()
        sub["score"] = model.predict(sub[feat_cols],
                                     num_iteration=model.best_iteration)
        all_scores.append(sub[["date", "ticker", "score", "fwd_ret_1m"]])

    scores_df = pd.concat(all_scores, ignore_index=True)
    scores_df.to_parquet(OUT_SCORES, index=False)
    print(f"\nSaved scores: {OUT_SCORES} | rows={len(scores_df):,} | "
          f"months={scores_df['date'].nunique()}")

    ic, icir, n = monthly_ic(scores_df)
    print(f"OOS Spearman IC: {ic:+.4f} | ICIR {icir:+.3f} | {n} months")

    # Side by side with the legacy file, if it is still around. Different
    # windows, so this is an orientation check rather than a fair test.
    legacy = DATA_DIR / "scores_lgbm.parquet"
    if legacy.exists():
        old = pd.read_parquet(legacy)
        o_ic, o_icir, o_n = monthly_ic(old)
        print(f"Legacy scores_lgbm.parquet: IC {o_ic:+.4f} | ICIR {o_icir:+.3f} "
              f"| {o_n} months ({old['date'].min().date()} to "
              f"{old['date'].max().date()})")
        print("Windows differ, so treat the gap as a starting point, not a result.")


if __name__ == "__main__":
    main()
