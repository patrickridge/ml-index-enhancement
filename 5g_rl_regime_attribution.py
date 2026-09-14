"""
5g_rl_regime_attribution.py - Where does the RL overlay's information ratio come from?

The overlay posts IR 0.494 over 143 walk-forward months and beats a fixed tilt
in all five folds. Consistency across folds is reassuring, but folds are just
consecutive two-year blocks - they do not tell you whether the edge is broad or
concentrated in one kind of market.

Splits the same active-return series two ways: by named historical regime, and
by realised benchmark volatility. A strategy that only works in calm markets is
a different product from one that works everywhere, and the fold table cannot
distinguish them.

Each regime gets a bootstrap interval as well as a point estimate. Regime
subsets are short - some have under 20 months - so point estimates there are
even less trustworthy than the headline, and quoting them bare would repeat the
mistake this project already made once.

Reads:  data/bt_wf_rl.csv
Writes: data/rl_regime_attribution.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
IN_PATH  = DATA_DIR / "bt_wf_rl.csv"
OUT_PATH = DATA_DIR / "rl_regime_attribution.csv"

N_BOOT     = 10_000
BLOCK_SIZE = 6
SEED       = 0

# Same named regimes as 4c_regime_engine.py
REGIMES = {
    "QE bull 2010-2019":     ("2010-01-01", "2019-12-31"),
    "COVID 2020-2021":       ("2020-01-01", "2021-12-31"),
    "Rate-hike bear 2022":   ("2022-01-01", "2022-12-31"),
    "AI bull 2023+":         ("2023-01-01", "2099-12-31"),
}


def info_ratio(r):
    r = np.asarray(r)
    if len(r) < 6:
        return np.nan
    sd = r.std(ddof=1)
    if sd <= 0:
        return np.nan
    return ((1 + r).prod() ** (12 / len(r)) - 1) / (sd * np.sqrt(12))


def boot_ci(r, rng, n_boot=N_BOOT, block=BLOCK_SIZE):
    r = np.asarray(r)
    n = len(r)
    if n < 12:
        return np.nan, np.nan
    nb  = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        st  = rng.integers(0, n, size=nb)
        idx = ((st[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        out[i] = info_ratio(r[idx])
    out = out[np.isfinite(out)]
    return tuple(np.percentile(out, [2.5, 97.5])) if len(out) else (np.nan, np.nan)


def main():
    if not IN_PATH.exists():
        print(f"ERROR: {IN_PATH} not found. Run 5c_walk_forward.py first.")
        return

    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["active_ret"])

    rng  = np.random.default_rng(SEED)
    rows = []

    overall    = info_ratio(df["active_ret"])
    o_lo, o_hi = boot_ci(df["active_ret"], rng)
    print("=" * 74)
    print("RL OVERLAY - REGIME ATTRIBUTION")
    print("=" * 74)
    print(f"  Overall  n={len(df):>3}  IR {overall:+.3f}  "
          f"95% CI [{o_lo:+.3f}, {o_hi:+.3f}]\n")

    print("  By named regime")
    print("  " + "-" * 66)
    for name, (s, e) in REGIMES.items():
        sub = df[(df["date"] >= s) & (df["date"] <= e)]
        if len(sub) < 6:
            print(f"    {name:<24} n={len(sub):>3}  (too short)")
            continue
        ir       = info_ratio(sub["active_ret"])
        lo, hi   = boot_ci(sub["active_ret"], rng)
        ann      = sub["active_ret"].mean() * 12 * 100
        hit      = (sub["active_ret"] > 0).mean() * 100
        star     = "" if (np.isnan(lo) or lo < 0 < hi) else " *"
        print(f"    {name:<24} n={len(sub):>3}  IR {ir:+.3f}  "
              f"CI [{lo:+.2f},{hi:+.2f}]  alpha {ann:+.2f}%  hit {hit:.0f}%{star}")
        rows.append({"split": "named", "regime": name, "n": len(sub),
                     "IR": round(ir, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "ann_alpha_pct": round(ann, 2), "hit_pct": round(hit, 1)})

    # Volatility split: benchmark vol above/below its own median
    df["bench_vol"] = df["bench_ret"].rolling(6, min_periods=3).std() * np.sqrt(12)
    med = df["bench_vol"].median()
    print("\n  By benchmark volatility (6m rolling, split at median "
          f"{med*100:.1f}%)")
    print("  " + "-" * 66)
    for name, mask in (("Low vol",  df["bench_vol"] <= med),
                       ("High vol", df["bench_vol"] >  med)):
        sub = df[mask].dropna(subset=["bench_vol"])
        if len(sub) < 6:
            continue
        ir     = info_ratio(sub["active_ret"])
        lo, hi = boot_ci(sub["active_ret"], rng)
        hit    = (sub["active_ret"] > 0).mean() * 100
        star   = "" if (np.isnan(lo) or lo < 0 < hi) else " *"
        print(f"    {name:<24} n={len(sub):>3}  IR {ir:+.3f}  "
              f"CI [{lo:+.2f},{hi:+.2f}]  hit {hit:.0f}%{star}")
        rows.append({"split": "vol", "regime": name, "n": len(sub),
                     "IR": round(ir, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "ann_alpha_pct": round(sub["active_ret"].mean() * 12 * 100, 2),
                     "hit_pct": round(hit, 1)})

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print("\n* = 95% interval excludes zero. Expect few or none: regime subsets")
    print("  are short, and short samples cannot establish much on their own.")


if __name__ == "__main__":
    main()
