"""
4c_regime_engine.py - Per-Regime Backtest Breakdown
====================================================
Loads existing index enhancement backtest results and breaks performance
out by market regime to answer: does the strategy hold up in bear markets,
not just the AI bull run?

Two regime methods:
  1. Rule-based (default) - same 4 hardcoded regimes as 2a_factor_analysis.py
  2. HMM (optional, --hmm flag) - 2-state Hidden Markov Model fitted to
     SPX monthly returns + VIX (detects risk-on / risk-off dynamically)

Metrics per regime:
  ann_alpha, tracking_error, info_ratio, hit_rate, max_active_dd, n_months

Outputs:
  data/ie_regime_breakdown.csv
  figures/ie_regime_breakdown.png  - IR bar chart per model × regime

Usage:
  python 4c_regime_engine.py          # rule-based regimes
  python 4c_regime_engine.py --hmm    # HMM-detected regimes
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Rule-based regimes (shared with 2a_factor_analysis.py)
REGIMES = {
    "QE Bull\n2010–2019":        ("2010-01-01", "2019-12-31"),
    "COVID Recovery\n2020–2021": ("2020-06-01", "2021-12-31"),
    "Rate Hike Bear\n2022":      ("2022-01-01", "2022-12-31"),
    "AI Bull\n2023–":            ("2023-01-01", "2099-12-31"),
}
COVID_EXCLUDE = ("2020-03-01", "2020-05-31")

# IE backtest files: model_name → csv path
MODELS = {
    "CS-Transformer": DATA_DIR / "bt_ie_cs_transformer.csv",
    "FT-Transformer": DATA_DIR / "bt_ie_transformer.csv",
    "LGBM":           DATA_DIR / "bt_ie_lgbm.csv",
}


# ── Stats helper ───────────────────────────────────────────────────────────────
def regime_stats(df: pd.DataFrame) -> dict:
    """Compute annualised performance metrics for a slice of monthly returns."""
    n = len(df)
    if n < 2:
        return dict(ann_alpha=np.nan, track_err=np.nan, info_ratio=np.nan,
                    hit_rate=np.nan, max_active_dd=np.nan, n_months=n)

    ar = df["active_ret"].values
    ann_alpha   = float(np.mean(ar) * 12)
    track_err   = float(np.std(ar, ddof=1) * np.sqrt(12))
    info_ratio  = ann_alpha / track_err if track_err > 1e-9 else np.nan
    hit_rate    = float(np.mean(ar > 0))

    # Max active drawdown on cumulative active return series
    cum = np.cumsum(ar)
    running_max = np.maximum.accumulate(cum)
    drawdowns   = cum - running_max
    max_active_dd = float(np.min(drawdowns))

    return dict(ann_alpha=round(ann_alpha * 100, 2),    # %
                track_err=round(track_err * 100, 2),    # %
                info_ratio=round(info_ratio, 3),
                hit_rate=round(hit_rate * 100, 1),       # %
                max_active_dd=round(max_active_dd * 100, 2),  # %
                n_months=n)


# ── Rule-based regime labels ───────────────────────────────────────────────────
def label_regimes_rule(dates: pd.Series) -> pd.Series:
    """Assign each date a regime label using hardcoded date ranges."""
    covid_s = pd.Timestamp(COVID_EXCLUDE[0])
    covid_e = pd.Timestamp(COVID_EXCLUDE[1])
    labels  = pd.Series("other", index=dates.index)

    for name, (start, end) in REGIMES.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        # Exclude COVID crash period from all regimes
        covid_mask = (dates >= covid_s) & (dates <= covid_e)
        labels[mask & ~covid_mask] = name

    return labels


# ── HMM regime labels ──────────────────────────────────────────────────────────
def label_regimes_hmm(dates: pd.Series) -> pd.Series:
    """
    Fit a 2-state Gaussian HMM on SPX monthly returns + VIX level from the
    panel, then label each date as 'Risk-On' or 'Risk-Off'.
    Falls back to rule-based if hmmlearn not installed.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  hmmlearn not installed - falling back to rule-based regimes.")
        print("  Install with:  pip install hmmlearn")
        return label_regimes_rule(dates)

    print("  Fitting 2-state HMM on SPX returns + VIX …")
    panel = pq.read_table(DATA_DIR / "panel_monthly_enriched.parquet",
                          columns=["date", "ticker", "spx_ret_1m", "vix_level"]).to_pandas()
    panel["date"] = pd.to_datetime(panel["date"])

    # One row per month (any ticker - macro cols are same for all)
    macro = panel.groupby("date")[["spx_ret_1m", "vix_level"]].first().reset_index()
    macro = macro.sort_values("date").dropna()

    # Fit HMM on returns + VIX
    X = macro[["spx_ret_1m", "vix_level"]].values
    model = GaussianHMM(n_components=2, covariance_type="full",
                        n_iter=200, random_state=42)
    model.fit(X)
    states = model.predict(X)

    # Identify which state = risk-on (higher mean SPX return)
    mean_ret = [X[states == s, 0].mean() for s in range(2)]
    risk_on_state = int(np.argmax(mean_ret))

    state_map = macro.set_index("date")["spx_ret_1m"].copy()
    state_map[:] = np.nan
    for i, row in macro.iterrows():
        state_map.loc[row["date"]] = states[i]

    labels = dates.map(lambda d: "Risk-On" if state_map.get(d, -1) == risk_on_state
                       else ("Risk-Off" if state_map.get(d, -1) == 1 - risk_on_state
                             else "other"))
    regime_counts = labels.value_counts()
    print(f"  HMM states: {regime_counts.to_dict()}")
    return labels


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    use_hmm = "--hmm" in sys.argv

    print("=" * 65)
    print("REGIME BACKTESTING ENGINE")
    print(f"Method: {'HMM (2-state)' if use_hmm else 'Rule-based (4 regimes)'}")
    print("=" * 65)

    # ── Load backtest results ─────────────────────────────────────────────────
    all_dfs = {}
    for model_name, fpath in MODELS.items():
        if not fpath.exists():
            print(f"  SKIP {model_name} - {fpath} not found")
            continue
        df = pd.read_csv(fpath)
        df["date"] = pd.to_datetime(df["date"])
        all_dfs[model_name] = df
        print(f"  Loaded {model_name}: {len(df)} months  "
              f"({df['date'].min().date()} → {df['date'].max().date()})")

    if not all_dfs:
        print("No backtest files found. Run 4b_index_enhancement.py first.")
        return

    # ── Assign regimes ────────────────────────────────────────────────────────
    # Use the date range from the first model for labelling
    sample_dates = next(iter(all_dfs.values()))["date"]
    if use_hmm:
        regime_labels = label_regimes_hmm(sample_dates)
    else:
        regime_labels = label_regimes_rule(sample_dates)

    regime_names = [r for r in (regime_labels.unique()) if r != "other"]
    if use_hmm:
        regime_names = sorted(set(regime_labels) - {"other"})
    else:
        regime_names = [r for r in REGIMES.keys() if r in regime_labels.values]

    # ── Compute per-regime stats ──────────────────────────────────────────────
    records = []
    for model_name, df in all_dfs.items():
        labels = label_regimes_rule(df["date"]) if not use_hmm else label_regimes_hmm(df["date"])

        # Full period stats
        stats = regime_stats(df)
        records.append({"model": model_name, "regime": "FULL PERIOD", **stats})

        # Per-regime stats
        for regime in regime_names:
            sub = df[labels == regime]
            stats = regime_stats(sub)
            records.append({"model": model_name, "regime": regime.replace("\n", " "), **stats})

    results = pd.DataFrame(records)

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"{'MODEL':<20} {'REGIME':<28} {'ANN α':>7} {'TE':>6} {'IR':>7} {'HIT%':>6} {'MaxDD':>7} {'N':>4}")
    print(f"{'─'*80}")

    for _, row in results.iterrows():
        regime_str = row["regime"]
        if row["n_months"] < 2:
            print(f"  {row['model']:<18} {regime_str:<28} {'-':>7} {'-':>6} {'-':>7} {'-':>6} {'-':>7} {row['n_months']:>4}")
            continue
        ir_str = f"{row['info_ratio']:>7.3f}" if not np.isnan(row['info_ratio']) else f"{'-':>7}"
        print(f"  {row['model']:<18} {regime_str:<28} "
              f"{row['ann_alpha']:>6.1f}% {row['track_err']:>5.1f}% "
              f"{ir_str} {row['hit_rate']:>5.1f}% "
              f"{row['max_active_dd']:>6.1f}% {row['n_months']:>4}")

    print(f"{'─'*80}")
    print("  ann_alpha = annualised active return vs benchmark")
    print("  TE = tracking error  |  IR = Information Ratio  |  HIT% = % months beating benchmark")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = DATA_DIR / "ie_regime_breakdown.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # ── Plot: IR by regime per model ──────────────────────────────────────────
    plot_df = results[results["regime"] != "FULL PERIOD"].copy()
    plot_df = plot_df[plot_df["n_months"] >= 3]  # skip regimes with too few months
    plot_df = plot_df.dropna(subset=["info_ratio"])

    if not plot_df.empty:
        regimes_plot = plot_df["regime"].unique()
        models_plot  = list(all_dfs.keys())
        x = np.arange(len(regimes_plot))
        width = 0.22

        colors = {"CS-Transformer": "#1565C0", "FT-Transformer": "#2E7D32", "LGBM": "#E65100"}

        fig, ax = plt.subplots(figsize=(max(9, len(regimes_plot) * 2.5), 5.5))

        for i, model in enumerate(models_plot):
            mdf = plot_df[plot_df["model"] == model].set_index("regime")
            vals = [mdf.loc[r, "info_ratio"] if r in mdf.index else np.nan for r in regimes_plot]
            bars = ax.bar(x + i * width - width, vals, width,
                          label=model, color=colors.get(model, f"C{i}"),
                          alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
                   label="IR=0.5 (good)")
        ax.set_xticks(x)
        ax.set_xticklabels(regimes_plot, fontsize=9)
        ax.set_ylabel("Information Ratio", fontsize=11)
        ax.set_title("Information Ratio by Market Regime\n"
                     f"({'HMM-detected' if use_hmm else 'Rule-based'} regimes)",
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out_png = FIG_DIR / "ie_regime_breakdown.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_png}")
    else:
        print("  (Not enough data per regime to plot - test period mostly in one regime)")

    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
