"""
4h_bootstrap_ci.py - Confidence intervals on every headline information ratio.

Every IR in this project is a point estimate over 17 to 143 months. At those
sample sizes the standard error is large enough that two strategies with
visibly different IRs can be statistically indistinguishable, so quoting the
point estimate alone overstates what the backtest actually established.

Uses a circular block bootstrap rather than an iid one. Monthly active returns
are autocorrelated - momentum crowding, regime persistence, slow-moving factor
exposure - and resampling individual months would break that dependence and
produce intervals that are too tight.

Reads:  data/bt_ie_*.csv, data/bt_wf_rl.csv  (whichever exist)
Writes: data/bootstrap_ci.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
OUT_PATH = DATA_DIR / "bootstrap_ci.csv"

N_BOOT     = 10_000
BLOCK_SIZE = 6        # months; long enough to carry regime persistence
SEED       = 0

SERIES = {
    "CS-Transformer IE": ("bt_ie_cs_transformer.csv", "active_ret"),
    "FT-Transformer IE": ("bt_ie_transformer.csv",    "active_ret"),
    "LightGBM IE":       ("bt_ie_lgbm.csv",           "active_ret"),
    "Factor-combo IE":   ("bt_ie_factor_combo.csv",   "active_ret"),
    "RL walk-forward":   ("bt_wf_rl.csv",             "active_ret"),
}


def info_ratio(r: np.ndarray) -> float:
    """Annualised active return over annualised tracking error."""
    if len(r) < 6:
        return np.nan
    sd = r.std(ddof=1)
    if sd <= 0:
        return np.nan
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    return ann / (sd * np.sqrt(12))


def circular_block_bootstrap(r: np.ndarray, n_boot: int, block: int,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Resample blocks of consecutive months, wrapping at the end of the series.

    Wrapping keeps every observation equally likely to be drawn; a non-circular
    version under-samples the tail.
    """
    n       = len(r)
    n_block = int(np.ceil(n / block))
    out     = np.empty(n_boot)

    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_block)
        idx    = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        out[i] = info_ratio(r[idx])
    return out


def main():
    rng  = np.random.default_rng(SEED)
    rows = []

    print(f"Circular block bootstrap | {N_BOOT:,} resamples | block {BLOCK_SIZE} months\n")

    for label, (fname, col) in SERIES.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  [skip] {label}: {fname} not found")
            continue

        df = pd.read_csv(path)
        if col not in df.columns:
            print(f"  [skip] {label}: no '{col}' column")
            continue

        r = df[col].dropna().to_numpy()
        if len(r) < 12:
            print(f"  [skip] {label}: only {len(r)} months")
            continue

        point = info_ratio(r)
        boot  = circular_block_bootstrap(r, N_BOOT, BLOCK_SIZE, rng)
        boot  = boot[np.isfinite(boot)]

        lo, hi   = np.percentile(boot, [2.5, 97.5])
        p_pos    = float((boot > 0).mean())
        crosses0 = lo < 0 < hi

        rows.append({
            "strategy":   label,
            "months":     len(r),
            "IR":         round(point, 3),
            "ci_lo_95":   round(lo, 3),
            "ci_hi_95":   round(hi, 3),
            "P(IR>0)":    round(p_pos, 3),
            "significant": not crosses0,
        })

        flag = " " if crosses0 else " *"
        print(f"  {label:<20} n={len(r):>3}  IR {point:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  P(IR>0)={p_pos:.1%}{flag}")

    if not rows:
        print("\nNo backtest files found. Run 4b_index_enhancement.py first.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print("\n* = 95% interval excludes zero.")
    print("An interval spanning zero means the backtest did not establish skill,")
    print("whatever the point estimate looks like.")


if __name__ == "__main__":
    main()
