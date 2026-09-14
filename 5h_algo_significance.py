"""
5h_algo_significance.py - Are the RL algorithms actually different from each other?

5d reports SAC 0.462, PPO 0.672, GRPO 0.339, DAPO 0.599 and it is tempting to
read that as a ranking. With five folds and an information ratio whose standard
error is large, most of that spread is sampling noise.

Two tests, neither of which needs a re-run:

  1. Per-algorithm dispersion across folds. Crude - five observations - but it
     shows how wide each interval is relative to the gaps between algorithms.

  2. Paired comparison against the fixed-alpha baseline. The folds are matched:
     every algorithm sees the same five periods, so differencing within a fold
     removes the period effect and tests the question that actually matters -
     does adaptive sizing beat a static tilt - with far more power than
     comparing two noisy IRs head to head.

Reads:  data/algo_comparison.csv
Writes: data/algo_significance.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
IN_PATH  = DATA_DIR / "algo_comparison.csv"
OUT_PATH = DATA_DIR / "algo_significance.csv"

ALGOS = ["sac", "ppo", "grpo", "dapo"]


def main():
    if not IN_PATH.exists():
        print(f"ERROR: {IN_PATH} not found. Run 5d_algorithm_comparison.py first.")
        return

    df = pd.read_csv(IN_PATH)
    n  = len(df)
    print("=" * 72)
    print(f"RL ALGORITHM SIGNIFICANCE  ({n} walk-forward folds)")
    print("=" * 72)

    fixed = df["fixed_ir"].to_numpy()

    print("\nPer-algorithm IR across folds")
    print("-" * 72)
    print(f"{'algo':<8}{'mean':>9}{'sd':>9}{'se':>9}{'95% CI':>22}")
    rows = []
    for a in ALGOS + ["fixed"]:
        ir = df[f"{a}_ir"].to_numpy()
        m, sd = ir.mean(), ir.std(ddof=1)
        se    = sd / np.sqrt(n)
        crit  = stats.t.ppf(0.975, n - 1)
        lo, hi = m - crit * se, m + crit * se
        print(f"{a.upper():<8}{m:>+9.3f}{sd:>9.3f}{se:>9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}")
        rows.append({"algo": a, "test": "level", "mean": round(m, 3),
                     "se": round(se, 3), "ci_lo": round(lo, 3),
                     "ci_hi": round(hi, 3), "p": np.nan})

    print("\nPaired vs fixed-alpha baseline (same folds, differenced)")
    print("-" * 72)
    print(f"{'algo':<8}{'mean diff':>12}{'se':>9}{'t':>8}{'p':>9}  verdict")
    for a in ALGOS:
        d      = df[f"{a}_ir"].to_numpy() - fixed
        m, sd  = d.mean(), d.std(ddof=1)
        se     = sd / np.sqrt(n)
        t      = m / se if se > 0 else np.nan
        p      = 2 * stats.t.sf(abs(t), n - 1)
        verdict = "beats fixed" if p < 0.05 else "not significant"
        print(f"{a.upper():<8}{m:>+12.3f}{se:>9.3f}{t:>8.2f}{p:>9.3f}  {verdict}")
        rows.append({"algo": a, "test": "paired_vs_fixed", "mean": round(m, 3),
                     "se": round(se, 3), "ci_lo": np.nan, "ci_hi": np.nan,
                     "p": round(p, 4)})

    print("\nPairwise between algorithms (paired, same folds)")
    print("-" * 72)
    any_sig = False
    for i, a in enumerate(ALGOS):
        for b in ALGOS[i + 1:]:
            d  = df[f"{a}_ir"].to_numpy() - df[f"{b}_ir"].to_numpy()
            se = d.std(ddof=1) / np.sqrt(n)
            t  = d.mean() / se if se > 0 else np.nan
            p  = 2 * stats.t.sf(abs(t), n - 1)
            if p < 0.05:
                any_sig = True
            print(f"  {a.upper():<5} vs {b.upper():<5} "
                  f"diff {d.mean():+.3f}  p={p:.3f}"
                  f"{'  *' if p < 0.05 else ''}")
            rows.append({"algo": f"{a}_vs_{b}", "test": "paired_pairwise",
                         "mean": round(d.mean(), 3), "se": round(se, 3),
                         "ci_lo": np.nan, "ci_hi": np.nan, "p": round(p, 4)})

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")

    if not any_sig:
        print("\nNo pair of algorithms differs significantly. The choice of policy")
        print("gradient is not what generates the result; sizing the tilt at all is.")


if __name__ == "__main__":
    main()
