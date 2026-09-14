"""
2i_multiple_testing.py - How many of the 263 factors survive a multiple-testing correction?

2a reports how many factors clear |t| > 2 individually. That count is close to
meaningless on its own: testing 263 factors at a 5% level produces roughly 13
false positives by construction, so a raw count in that region is consistent
with no real signal anywhere in the library.

Applies Benjamini-Hochberg-Yekutieli. BHY rather than plain BH because factor
IC time series are heavily cross-correlated - momentum variants move together,
volatility variants move together - and BH assumes independence or positive
regression dependence. BHY carries an extra log correction that holds under
arbitrary dependence, which is the honest assumption here.

The factor-mining track in research/factor_mining/validation_engine.py already
applies BHY to its candidates. The main 2a screen never did.

Reads:  data/factor_ic_summary.csv
Writes: data/factor_bhy.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
IN_PATH  = DATA_DIR / "factor_ic_summary.csv"
OUT_PATH = DATA_DIR / "factor_bhy.csv"

FDR_Q = 0.10     # target false discovery rate


def bhy_threshold(pvals: np.ndarray, q: float):
    """
    Benjamini-Hochberg-Yekutieli step-up under arbitrary dependence.

    Returns (boolean reject mask aligned to input order, p-value cutoff).
    The c(m) = sum(1/i) factor is what makes this valid when tests are
    correlated; dropping it gives plain BH and an anti-conservative answer.
    """
    m     = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    c_m       = np.sum(1.0 / np.arange(1, m + 1))
    crit      = (np.arange(1, m + 1) / m) * q / c_m
    passing   = np.where(ranked <= crit)[0]

    reject = np.zeros(m, dtype=bool)
    if len(passing) == 0:
        return reject, 0.0

    k_max = passing[-1]
    reject[order[: k_max + 1]] = True
    return reject, float(ranked[k_max])


def main():
    if not IN_PATH.exists():
        print(f"ERROR: {IN_PATH} not found. Run 2a_factor_analysis.py first.")
        return

    df = pd.read_csv(IN_PATH)

    # 2a's column naming has drifted over time; accept the plausible variants
    t_col = next((c for c in ("t_stat", "tstat", "t", "ic_t") if c in df.columns), None)
    n_col = next((c for c in ("n_months", "months", "n") if c in df.columns), None)
    f_col = next((c for c in ("factor", "feature", "name") if c in df.columns), None)

    if t_col is None:
        print(f"ERROR: no t-stat column in {IN_PATH.name}. Columns: {list(df.columns)}")
        return

    df = df.dropna(subset=[t_col]).copy()
    n_tests = len(df)

    # Two-sided p from the t-stat. Use the t distribution where a month count is
    # available, otherwise fall back to the normal approximation.
    if n_col is not None and df[n_col].notna().all():
        dof = (df[n_col] - 1).clip(lower=1)
        df["p_value"] = 2 * stats.t.sf(df[t_col].abs(), dof)
    else:
        df["p_value"] = 2 * stats.norm.sf(df[t_col].abs())

    reject, cutoff = bhy_threshold(df["p_value"].to_numpy(), FDR_Q)
    df["bhy_reject"] = reject

    n_raw  = int((df["p_value"] < 0.05).sum())
    n_bhy  = int(reject.sum())
    n_exp  = 0.05 * n_tests

    print("=" * 68)
    print("MULTIPLE-TESTING CORRECTION (Benjamini-Hochberg-Yekutieli)")
    print("=" * 68)
    print(f"  Factors tested                    : {n_tests}")
    print(f"  Passing raw p < 0.05              : {n_raw}")
    print(f"  Expected false positives at 5%    : {n_exp:.0f}")
    print(f"  Surviving BHY at FDR q={FDR_Q:.2f}       : {n_bhy}")
    print(f"  BHY p-value cutoff                : {cutoff:.6f}")

    if n_raw <= n_exp:
        print("\n  The raw count is at or below the number expected by chance.")
        print("  Nothing here is distinguishable from a library of pure noise.")

    if n_bhy:
        keep = df[reject].reindex(df[reject]["p_value"].sort_values().index)
        show = [c for c in (f_col, t_col, "p_value") if c]
        print(f"\n  Survivors:")
        print(keep[show].head(25).to_string(index=False))
    else:
        print("\n  No factor survives the correction.")
        print("  Individually significant factors are what 263 tests produce by chance.")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
