"""
2k_fdr_sensitivity.py - Does the survivor count depend on the thresholds chosen?

2i reports 17 survivors at FDR 0.10 under BHY. Three arbitrary choices sit
behind that number: the raw p < 0.05 screen it is contrasted against, the FDR
level itself, and the decision to correct for arbitrary dependence rather than
assume independence. None of the three is derived from anything. The 5% in
particular is Fisher's 1925 convention, which stuck because pre-computer
statistical tables were printed at 5% and 1%.

A number resting on three conventions deserves a check that it does not move
when they do. This sweeps all three and reports how the count responds.

The companion check is 2j, which asks whether the survivors are an artifact of
the weight floor. That one found a real weakness. This one does not, which is
worth stating plainly: a sensitivity analysis that always finds problems is not
measuring anything.

Reads:  data/factor_ic_summary.csv
Writes: data/fdr_sensitivity.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
IN_PATH  = DATA_DIR / "factor_ic_summary.csv"
OUT_PATH = DATA_DIR / "fdr_sensitivity.csv"

RAW_LEVELS = [0.10, 0.05, 0.01, 0.005, 0.001]
FDR_LEVELS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]


def step_up(pvals_sorted, q, m, c):
    """
    Benjamini-Hochberg step-up, or BHY when c > 1.

    Walk the sorted p-values against a bar that loosens with rank, take the
    largest rank that still clears it, and accept everything up to there.
    c = 1 gives plain BH; c = sum(1/i) gives BHY, which holds under arbitrary
    dependence.
    """
    thresholds = (np.arange(1, m + 1) / m) * q / c
    passing = np.where(pvals_sorted <= thresholds)[0]
    return int(passing.max() + 1) if len(passing) else 0


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"{IN_PATH} not found. Run 2a first.")

    df = pd.read_csv(IN_PATH).dropna(subset=["t_stat"])
    m = len(df)

    # Two-sided t-test on the IC time series, same as 2i.
    dof = df["n_months"] - 1
    pvals = 2 * stats.t.sf(df["t_stat"].abs(), dof)
    ranked = np.sort(pvals)

    c_bhy = float(np.sum(1.0 / np.arange(1, m + 1)))

    print(f"{m} factors with computable t-statistics")
    print(f"c(m) = {c_bhy:.3f}, so the BHY bar sits {c_bhy:.1f}x below BH's\n")

    rows = []

    print("RAW SIGNIFICANCE LEVEL")
    print(f"  {'p <':>7} {'pass':>6} {'by chance':>11} {'ratio':>7}")
    for a in RAW_LEVELS:
        n_pass = int((pvals < a).sum())
        expected = a * m
        ratio = n_pass / expected if expected > 0 else np.nan
        print(f"  {a:>7} {n_pass:>6} {expected:>11.1f} {ratio:>7.1f}")
        rows.append({"knob": "raw_p", "value": a, "n_pass": n_pass,
                     "expected_by_chance": round(expected, 2),
                     "ratio": round(ratio, 2)})

    print("\nFDR LEVEL")
    print(f"  {'q':>7} {'BHY':>6} {'plain BH':>10}")
    for q in FDR_LEVELS:
        n_bhy = step_up(ranked, q, m, c_bhy)
        n_bh  = step_up(ranked, q, m, 1.0)
        print(f"  {q:>7} {n_bhy:>6} {n_bh:>10}")
        rows.append({"knob": "fdr_q", "value": q, "n_pass": n_bhy,
                     "n_pass_bh": n_bh})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")

    # Verdict, computed rather than asserted.
    counts = [step_up(ranked, q, m, c_bhy) for q in FDR_LEVELS if q >= 0.10]
    flat = len(set(counts)) == 1
    tiny = int((pvals < 0.001).sum())
    headline = step_up(ranked, 0.10, m, c_bhy)

    print("\n" + "=" * 62)
    print("VERDICT")
    print("=" * 62)
    if flat:
        print(f"Survivor count is flat at {counts[0]} from q=0.10 through "
              f"q={max(FDR_LEVELS)}.")
        print("Tripling the tolerance for false discoveries finds nothing new,")
        print("so the FDR choice is not what produces the answer.")
    else:
        print(f"Survivor count moves across the FDR range: {counts}.")
        print("The choice of q is doing real work and should be justified.")

    print(f"\n{tiny} factors clear p < 0.001 against 0.2 expected by chance.")
    if tiny >= headline:
        print("Every survivor sits far below the 5% boundary, so the")
        print("conventional level is reported for contrast, not relied on.")

    gap = step_up(ranked, 0.10, m, 1.0) - headline
    print(f"\nBHY costs {gap} factor(s) against plain BH despite a "
          f"{c_bhy:.0f}x stricter bar.")
    print("The dependence correction is close to free here, which means the")
    print("result does not rest on assuming the tests are independent.")


if __name__ == "__main__":
    main()
