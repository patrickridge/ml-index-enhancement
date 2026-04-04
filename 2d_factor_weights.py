"""
2d_factor_weights.py — Factor Selection, Weighting & Partial Signal Flags
==========================================================================
Combines all factor analysis outputs into a single clean table for use
in model training and portfolio construction.

Steps:
  1. Filter — keep factors with |IC| > 0 AND regime_stable (sign consistent
     across ≥ 75% of regimes). Result: ~43 / 59 factors.

  2. IC decay weighting — for each surviving factor, compute a persistence
     weight = mean |IC| over lags 0–12 months (area under curve for 1 year).
     Factors whose signal persists longer are weighted more heavily.
     Weights are normalised to sum to 1 across all survivors.

  3. Quantile adjustment flag — factors where quintile ordering is NOT
     monotonic (Q5 > Q4 > Q3 > Q2 > Q1 broken) get `use_partial = True`.
     For these, only the top / bottom 100 stocks carry signal — the middle
     300 should be held at benchmark weight. Monotonic factors can use the
     full continuous tilt.

Outputs:
  data/factor_selected.csv   — clean factor list, ready for model training
  figures/factor_weights.png — bar chart of normalised decay weights

Run after 5_factor_analysis.py and 5b_ic_decay_all.py.
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

IC_SUMMARY_IN    = DATA_DIR / "factor_ic_summary.csv"
REGIME_IN        = DATA_DIR / "factor_regime_stability.csv"
QUINTILE_IN      = DATA_DIR / "factor_quintile_returns.csv"
IC_DECAY_ALL_IN  = DATA_DIR / "factor_ic_decay_all.csv"

OUT_CSV          = DATA_DIR / "factor_selected.csv"
OUT_FIG          = FIG_DIR  / "factor_weights.png"

# How many decay lags (months) to use for persistence weight
DECAY_LAGS = 12   # mean |IC| over lags 0–12 months


def main():
    print("=" * 65)
    print("FACTOR SELECTION, WEIGHTING & PARTIAL FLAGS")
    print("=" * 65)

    # ── Load inputs ──────────────────────────────────────────────────────────────
    ic_df      = pd.read_csv(IC_SUMMARY_IN)        # factor, ic_mean, icir, t_stat, …
    regime_df  = pd.read_csv(REGIME_IN)            # factor, regime_stable, majority_sign, …
    quint_df   = pd.read_csv(QUINTILE_IN)          # factor, monotonic, Q1..Q5, …
    decay_df   = pd.read_csv(IC_DECAY_ALL_IN,
                              index_col=0)          # index=factor, cols=lag 0,1,…60

    # Rename decay columns to int lags
    decay_df.columns = [int(c) for c in decay_df.columns]

    print(f"\nLoaded:")
    print(f"  IC summary:        {len(ic_df)} factors")
    print(f"  Regime stability:  {len(regime_df)} factors")
    print(f"  Quintile returns:  {len(quint_df)} factors")
    print(f"  IC decay (all):    {len(decay_df)} factors × {len(decay_df.columns)} lags")

    # ── Merge ───────────────────────────────────────────────────────────────────
    merged = ic_df[["factor", "ic_mean", "ic_std", "icir", "t_stat"]].copy()
    merged = merged.merge(
        regime_df[["factor", "regime_stable", "majority_sign",
                   "regimes_positive", "ic_qe_bull", "ic_covid_recovery",
                   "ic_rate_hike_bear", "ic_ai_bull"]],
        on="factor", how="left"
    )

    # ── Filter 1: |IC| > 0 (keep both directions) ────────────────────────────────
    merged["abs_ic"] = merged["ic_mean"].abs()
    f1 = merged[merged["abs_ic"] > 0].copy()
    print(f"\nFilter 1 — |IC| > 0: {len(f1)} / {len(merged)} kept")

    # ── Filter 2: Regime stable ────────────────────────────────────────────────
    f2 = f1[f1["regime_stable"] == True].copy()
    print(f"Filter 2 — Regime stable (≥75% sign-consistent): {len(f2)} / {len(f1)} kept")

    pos = f2[f2["ic_mean"] > 0]["factor"].tolist()
    neg = f2[f2["ic_mean"] < 0]["factor"].tolist()
    print(f"  Positive IC ({len(pos)}): {pos}")
    print(f"  Contrarian  ({len(neg)}): {neg}")

    # ── Add monotonicity flag from quintile test ──────────────────────────────
    quint_mono = quint_df[["factor", "monotonic"]].copy()
    f2 = f2.merge(quint_mono, on="factor", how="left")

    # Factors NOT in quintile CSV weren't in top 20 tested — mark as unknown (NaN)
    # use_partial = True for non-monotonic factors (only top/bottom 100 signal)
    # use_partial = False for monotonic factors (full continuous tilt applies)
    f2["use_partial"] = f2["monotonic"].apply(
        lambda x: True if x is False or x == False else False
    )
    # If monotonicity untested (NaN), assume full use
    f2["use_partial"] = f2["use_partial"].fillna(False)

    n_partial = f2["use_partial"].sum()
    n_full    = (~f2["use_partial"]).sum()
    print(f"\nQuintile monotonicity:")
    print(f"  Full signal (monotonic or untested): {n_full} factors")
    print(f"  Partial signal (top/bottom 100 only): {n_partial} factors")
    if n_partial > 0:
        print(f"  Partial factors: {f2[f2['use_partial']]['factor'].tolist()}")

    # ── IC decay weighting ───────────────────────────────────────────────────
    # Weight = mean |IC| over lags 0–DECAY_LAGS months (area under decay curve)
    # This rewards factors whose signal persists longer — more reliable for
    # monthly rebalancing strategies with low turnover.
    print(f"\nComputing decay weights (mean |IC| over lags 0–{DECAY_LAGS}) …")

    lag_cols = [c for c in range(DECAY_LAGS + 1) if c in decay_df.columns]
    decay_window = decay_df[lag_cols]

    # Mean |IC| across lags 0–12 for each factor
    decay_weight_raw = decay_window.abs().mean(axis=1)   # Series: factor → raw weight

    # Map to surviving factors only
    f2["decay_weight_raw"] = f2["factor"].map(decay_weight_raw)

    # Factors with missing decay data (shouldn't happen but guard) → use abs IC
    missing_decay = f2["decay_weight_raw"].isna().sum()
    if missing_decay > 0:
        print(f"  Warning: {missing_decay} factors missing decay data — using |IC| as fallback")
        f2["decay_weight_raw"] = f2["decay_weight_raw"].fillna(f2["abs_ic"])

    # Normalise to sum to 1
    total = f2["decay_weight_raw"].sum()
    f2["weight"] = (f2["decay_weight_raw"] / total).round(4)

    print(f"  Total raw weight before normalisation: {total:.4f}")
    print(f"  Normalised — sum: {f2['weight'].sum():.4f}")

    # Sort by weight descending
    f2 = f2.sort_values("weight", ascending=False).reset_index(drop=True)

    print(f"\n{'─'*70}")
    print(f"TOP 15 FACTORS BY DECAY WEIGHT")
    print(f"{'─'*70}")
    print(f"{'Factor':<28} {'IC':>7} {'ICIR':>7} {'Mono':>6} {'Partial':>8} {'Weight':>7}")
    print(f"{'─'*70}")
    for _, row in f2.head(15).iterrows():
        mono_str    = "✓" if row.get("monotonic") is True else ("✗" if row.get("monotonic") is False else "?")
        partial_str = "partial" if row["use_partial"] else "full"
        print(f"  {row['factor']:<26} {row['ic_mean']:>+7.4f}  {row['icir']:>6.3f}  "
              f"{mono_str:>4}  {partial_str:>8}  {row['weight']:>6.4f}")

    # ── Save ────────────────────────────────────────────────────────────────────
    out_cols = [
        "factor", "ic_mean", "ic_std", "icir", "t_stat",
        "majority_sign", "regime_stable", "regimes_positive",
        "ic_qe_bull", "ic_covid_recovery", "ic_rate_hike_bear", "ic_ai_bull",
        "monotonic", "use_partial",
        "decay_weight_raw", "weight"
    ]
    f2[out_cols].to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}  ({len(f2)} factors)")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FACTOR SELECTION COMPLETE")
    print(f"{'='*65}")
    print(f"  Total features analysed:      {len(merged)}")
    print(f"  Surviving after filters:      {len(f2)}")
    print(f"  → Positive IC (buy high):     {len(pos)}")
    print(f"  → Contrarian  (buy low):      {len(neg)}")
    print(f"  → Full signal (monotonic):    {n_full}")
    print(f"  → Partial only (top/bot 100): {n_partial}")
    print(f"\nHighest-weight factor: {f2.iloc[0]['factor']}  (weight={f2.iloc[0]['weight']:.4f})")
    print(f"Lowest-weight factor:  {f2.iloc[-1]['factor']}  (weight={f2.iloc[-1]['weight']:.4f})")

    # ── Plot ────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(6, len(f2) * 0.35)))

    colors = []
    for _, row in f2.iterrows():
        if row["ic_mean"] > 0:
            colors.append("#2ecc71")   # green = positive IC
        else:
            colors.append("#e74c3c")   # red = contrarian

    y_pos = range(len(f2))
    ax.barh(list(y_pos), f2["weight"], color=colors, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(f2["factor"], fontsize=8)
    ax.set_xlabel("Normalised Decay Weight  (mean |IC| lags 0–12, sum=1)")
    ax.set_title(
        f"Factor Weights — {len(f2)} Survivors  "
        f"(green = positive IC, red = contrarian)\n"
        f"Weight ∝ signal persistence over 12-month window"
    )
    # Mark partial factors
    for i, row in enumerate(f2.itertuples()):
        if row.use_partial:
            ax.text(row.weight + 0.0002, i, "partial", fontsize=7,
                    va="center", color="gray")
    ax.axvline(1 / len(f2), color="black", linewidth=0.8, linestyle="--",
               label="equal weight")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT_FIG.name}")

    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
