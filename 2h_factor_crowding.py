"""
2h_factor_crowding.py — Factor Crowding, Decay & Contrarian Diagnostic
=======================================================================
Addresses Kieran's four concerns:

  1. FACTOR CROWDING — correlation matrix (IS data only), cluster highly
     correlated factors (r > 0.70), flag redundant pairs to prune before retraining.

  2. 7-DAY DECAY — within-month IC decay at days 1/3/7/10 from existing
     factor_ic_decay_daily.csv.  "Do we need a 7-day refresh or monthly is fine?"

  3. CONTRARIAN FACTORS — IS vs OOS for the sign-flip group.
     Are short-term reversal signals actually useful contrarian bets or noise?

  4. QUINTILE DIRECTION AUDIT — flag factors where Q5-Q1 sign contradicts IC_IS.
     Shows the 'wrong direction' factors Kieran spotted.

STRICT IS / OOS SEPARATION:
  - All training metrics  → data ≤ TRAIN_END  (from config.py)
  - All OOS metrics       → data >  TRAIN_END
  - Never mixed.

Outputs:
  data/crowding_corr_matrix.csv       — pairwise IC correlation matrix (IS)
  data/crowding_redundant_pairs.csv   — pairs with |r| > 0.70
  data/crowding_7day_decay.csv        — day_1 / day_3 / day_7 IC per factor
  data/crowding_contrarian_oos.csv    — IS vs OOS for sign-flip factors
  data/crowding_quintile_audit.csv    — quintile direction audit
  figures/crowding_corr_heatmap.png
  figures/crowding_7day_decay.png
  figures/crowding_contrarian_oos.png
  figures/crowding_quintile_audit.png
"""

import warnings
warnings.filterwarnings("ignore")

import time as _time
_t0 = _time.time()

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import DATA_DIR as _cfg_dir, TRAIN_END, VALID_END, MACRO_COLS
    DATA_DIR = Path(_cfg_dir)
except ImportError:
    DATA_DIR   = Path("data")
    TRAIN_END  = "2022-12-31"
    VALID_END  = "2024-06-30"
    MACRO_COLS = []

FIG_DIR  = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_STOCKS       = 20
CROWD_THRESHOLD  = 0.70   # |r| above this = crowded pair
PANEL_PATH       = DATA_DIR / "panel_monthly_enriched.parquet"

NON_FEATURE_COLS = {
    "date", "ticker", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
}

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("FACTOR CROWDING & DECAY DIAGNOSTIC")
print("=" * 65)
print(f"\nLoading panel → {PANEL_PATH}")

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])

# Strict IS / OOS split — never mix
IS_END = pd.Timestamp(TRAIN_END)
OOS_START = IS_END + pd.DateOffset(days=1)

panel_is  = panel[panel["date"] <= IS_END].copy()
panel_oos = panel[panel["date"] >= OOS_START].copy()

print(f"  IS  : {panel_is['date'].min().date()} → {panel_is['date'].max().date()}  "
      f"({len(panel_is):,} rows, {panel_is['date'].nunique()} months)")
print(f"  OOS : {panel_oos['date'].min().date()} → {panel_oos['date'].max().date()}  "
      f"({len(panel_oos):,} rows, {panel_oos['date'].nunique()} months)")

# Factor list (exclude macro — macro are same for all stocks within a month, zero CS variance)
all_factor_cols = [c for c in panel.columns if c not in NON_FEATURE_COLS]
cs_factors = [c for c in all_factor_cols if c not in MACRO_COLS]
print(f"\n  CS factors : {len(cs_factors)}")
print(f"  Macro cols : {len(MACRO_COLS)}  (excluded from crowding — zero CS variance)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FACTOR CROWDING (IS DATA ONLY)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("SECTION 1 — Factor Crowding (IS data only)")
print("─" * 65)

# Build monthly factor pivot: mean factor value per stock per month (IS only)
# Then compute pairwise Spearman correlation of the factor columns
# across the full IS panel (stock-months as observations)

def compute_factor_corr_matrix(panel_is, factors):
    """
    Pairwise Spearman correlation of factor values across stock-months (IS).
    Uses a pivot of mean factor per stock-month to be memory-efficient.
    """
    sub = panel_is[factors].copy()
    # Sample if very large to keep runtime reasonable (> 100k rows)
    if len(sub) > 60_000:
        sub = sub.sample(60_000, random_state=42)
    # Spearman = Pearson of ranks
    ranked = sub.rank(pct=True)
    corr = ranked.corr(method="pearson")
    return corr

print("  Computing pairwise factor correlation (IS)...")
corr_mat = compute_factor_corr_matrix(panel_is, cs_factors)
corr_mat.to_csv(DATA_DIR / "crowding_corr_matrix.csv")
print(f"  Saved → data/crowding_corr_matrix.csv  shape={corr_mat.shape}")

# Identify redundant pairs (|r| > threshold)
print(f"\n  Flagging pairs with |r| > {CROWD_THRESHOLD}...")
upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
pairs = (upper.stack()
              .reset_index()
              .rename(columns={"level_0": "factor_a", "level_1": "factor_b", 0: "corr"}))
pairs["abs_corr"] = pairs["corr"].abs()
redundant = pairs[pairs["abs_corr"] > CROWD_THRESHOLD].sort_values("abs_corr", ascending=False)
redundant.to_csv(DATA_DIR / "crowding_redundant_pairs.csv", index=False)

print(f"  Redundant pairs found: {len(redundant)}")
print(f"  Top 15 most-crowded pairs:")
print(redundant.head(15).to_string(index=False))

# Hierarchical clustering heatmap
print("\n  Plotting correlation heatmap with clustering...")
try:
    dist = 1 - corr_mat.abs()
    np.fill_diagonal(dist.values, 0)
    dist_condensed = squareform(dist.values.clip(0))
    link = linkage(dist_condensed, method="ward")
    order = dendrogram(link, no_plot=True)["leaves"]

    mat_ordered = corr_mat.iloc[order, order]
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        mat_ordered, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=mat_ordered.columns,
        yticklabels=mat_ordered.index,
        square=True, linewidths=0,
        cbar_kws={"shrink": 0.6, "label": "Spearman ρ"},
    )
    ax.set_title(
        f"Factor Correlation Matrix — IS only (≤{TRAIN_END})\n"
        f"{len(cs_factors)} CS factors | Hierarchically clustered | Threshold = {CROWD_THRESHOLD}",
        fontsize=13, pad=12
    )
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", rotation=0,  labelsize=6)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "crowding_corr_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved → figures/crowding_corr_heatmap.png")
except Exception as e:
    print(f"  [WARN] heatmap failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 7-DAY DECAY (from factor_ic_decay_daily.csv)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("SECTION 2 — 7-Day Intra-Month Factor Decay")
print("─" * 65)
print("  (Uses factor_ic_decay_daily.csv — IC at days 1/2/3/5/7/10 ahead)")

decay_path = DATA_DIR / "factor_ic_decay_daily.csv"
if not decay_path.exists():
    print("  [WARN] factor_ic_decay_daily.csv not found — skipping Section 2")
    decay_daily = None
else:
    decay_daily = pd.read_csv(decay_path, index_col="factor")
    # Columns: day_1, day_2, day_3, day_5, day_10, day_15, day_20, ...
    # We want day_1, day_3, day_7 (or day_5 if day_7 absent)
    day_cols_want = ["day_1", "day_3", "day_5", "day_10"]
    day_cols_avail = [c for c in day_cols_want if c in decay_daily.columns]
    decay_7d = decay_daily[day_cols_avail].copy()

    # Also load lag-0 IC from factor_ic_summary for context
    ic_summary = pd.read_csv(DATA_DIR / "factor_ic_summary.csv", index_col="factor")
    decay_7d["IC_day0"] = ic_summary["ic_mean"].reindex(decay_7d.index)

    # Decay ratio: IC_day5 / IC_day0 — >0.5 = slow decay, <0.2 = very fast
    if "day_5" in decay_7d.columns:
        decay_7d["decay_5d_ratio"] = (
            decay_7d["day_5"] / decay_7d["IC_day0"].replace(0, np.nan)
        ).round(3)

    # Flag fast decayers: |IC_day5| < 30% of |IC_day0| and |IC_day0| > 0.01
    decay_7d["fast_decay"] = (
        (decay_7d["IC_day0"].abs() > 0.01) &
        (decay_7d.get("decay_5d_ratio", pd.Series(np.nan, index=decay_7d.index)).abs() < 0.3)
    )

    decay_7d = decay_7d.sort_values("IC_day0", key=abs, ascending=False)
    decay_7d.to_csv(DATA_DIR / "crowding_7day_decay.csv")
    print(f"  Saved → data/crowding_7day_decay.csv")
    print(f"\n  Top 20 factors by |IC_day0| with 7-day decay:")
    print(decay_7d.head(20).to_string())

    n_fast = decay_7d["fast_decay"].sum()
    print(f"\n  Fast-decaying factors (decay within 5 days): {n_fast}")
    fast_list = decay_7d[decay_7d["fast_decay"]].index.tolist()
    if fast_list:
        print("  ", fast_list[:20])

    # Plot: bar chart of IC_day0, day_3, day_5 for top 30 factors
    top30 = decay_7d.head(30)
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(top30))
    w = 0.25
    ax.bar(x - w, top30["IC_day0"], w, label="Day 0 (monthly IC)", color="#2196F3", alpha=0.85)
    if "day_3" in top30.columns:
        ax.bar(x,     top30["day_3"],   w, label="Day 3",  color="#FF9800", alpha=0.85)
    if "day_5" in top30.columns:
        ax.bar(x + w, top30["day_5"],   w, label="Day 5",  color="#F44336", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top30.index, rotation=90, fontsize=7)
    ax.set_ylabel("IC")
    ax.set_title("7-Day Factor Decay — IC at Day 0 / 3 / 5 (IS data)\n"
                 "Rapid sign-flip or collapse by Day 5 = factor needs daily refresh",
                 fontsize=12)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "crowding_7day_decay.png", dpi=150)
    plt.close(fig)
    print("  Saved → figures/crowding_7day_decay.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONTRARIAN FACTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("SECTION 3 — Contrarian Factor Analysis (IS vs OOS)")
print("─" * 65)
print("  Strict split: IS ≤ TRAIN_END, OOS > TRAIN_END — NEVER mixed")

oos_ic_path = DATA_DIR / "factor_oos_ic.csv"
if not oos_ic_path.exists():
    print("  [WARN] factor_oos_ic.csv not found — run 2a first")
    sign_flip_factors = []
else:
    oos_ic = pd.read_csv(oos_ic_path, index_col="factor")

    # Sign-flip factors = contrarian IS, different direction OOS
    flip_df = oos_ic[oos_ic["sign_flip"] == True].copy()
    stable_df = oos_ic[oos_ic["sign_flip"] == False].copy()

    print(f"\n  Sign-flip factors (IS → OOS direction reversed): {len(flip_df)}")
    print(f"  Stable factors  (same direction IS and OOS):     {len(stable_df)}")

    # Recompute fresh IS IC for sign-flip factors to confirm
    def monthly_ic_series(panel_sub, factor, target="fwd_ret_1m"):
        results = {}
        for dt, grp in panel_sub.groupby("date"):
            sub = grp[[factor, target]].dropna()
            if len(sub) < MIN_STOCKS:
                continue
            corr, _ = spearmanr(sub[factor], sub[target])
            if not np.isnan(corr):
                results[dt] = corr
        return pd.Series(results)

    # Pick top 20 flip factors by |IC_OOS|
    flip_top = flip_df.reindex(
        flip_df["ic_oos"].abs().sort_values(ascending=False).index
    ).head(20)

    contrarian_records = []
    for fac in flip_top.index:
        if fac not in panel_is.columns:
            continue
        ic_is_series  = monthly_ic_series(panel_is,  fac)
        ic_oos_series = monthly_ic_series(panel_oos, fac)
        contrarian_records.append({
            "factor":         fac,
            "ic_is":          ic_is_series.mean().round(4),
            "icir_is":        (ic_is_series.mean() / ic_is_series.std()).round(3) if ic_is_series.std() > 1e-8 else np.nan,
            "pct_pos_is":     (ic_is_series > 0).mean().round(3),
            "ic_oos":         ic_oos_series.mean().round(4) if len(ic_oos_series) > 0 else np.nan,
            "icir_oos":       (ic_oos_series.mean() / ic_oos_series.std()).round(3) if len(ic_oos_series) > 3 and ic_oos_series.std() > 1e-8 else np.nan,
            "pct_pos_oos":    (ic_oos_series > 0).mean().round(3) if len(ic_oos_series) > 0 else np.nan,
            "n_is_months":    len(ic_is_series),
            "n_oos_months":   len(ic_oos_series),
            "interpretation": "SHORT-TERM REVERSAL" if "ret_1" in fac or "ret_2" in fac or "ret_1w" in fac
                              else "TECHNICAL OSC"   if any(x in fac for x in ["rsi","macd","bollinger","price_to_ma"])
                              else "OTHER",
        })

    contrarian_df = pd.DataFrame(contrarian_records).set_index("factor")
    contrarian_df.to_csv(DATA_DIR / "crowding_contrarian_oos.csv")
    print(f"\n  Contrarian factor IS vs OOS (top {len(contrarian_df)} by |IC_OOS|):")
    print(contrarian_df.to_string())
    print(f"\n  Saved → data/crowding_contrarian_oos.csv")

    sign_flip_factors = flip_df.index.tolist()

    # Plot: IS vs OOS IC for contrarian factors
    if len(contrarian_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(contrarian_df))
        w = 0.35
        bars_is  = ax.bar(x - w/2, contrarian_df["ic_is"],  w, label="IS IC",  color="#1976D2", alpha=0.85)
        bars_oos = ax.bar(x + w/2, contrarian_df["ic_oos"], w, label="OOS IC", color="#E53935", alpha=0.85)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(contrarian_df.index, rotation=90, fontsize=8)
        ax.set_ylabel("IC (Spearman)")
        ax.set_title(
            f"Contrarian Factors — Sign Reversal IS→OOS\n"
            f"IS ≤ {TRAIN_END} (blue)  |  OOS > {TRAIN_END} (red)\n"
            f"Flipped sign = factor is a REVERSAL signal, not momentum",
            fontsize=11
        )
        ax.legend()
        plt.tight_layout()
        fig.savefig(FIG_DIR / "crowding_contrarian_oos.png", dpi=150)
        plt.close(fig)
        print("  Saved → figures/crowding_contrarian_oos.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — QUINTILE DIRECTION AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("SECTION 4 — Quintile Direction Audit")
print("─" * 65)
print("  Flag factors where Q5-Q1 spread sign contradicts IC_IS sign.")
print("  These are the 'wrong direction' factors Kieran spotted.\n")

quint_path = DATA_DIR / "factor_quintile_returns.csv"
ic_sum_path = DATA_DIR / "factor_ic_summary.csv"

if not quint_path.exists() or not ic_sum_path.exists():
    print("  [WARN] factor_quintile_returns.csv or factor_ic_summary.csv not found — run 2a first")
else:
    quint_df = pd.read_csv(quint_path)
    ic_sum   = pd.read_csv(ic_sum_path)

    # factor_quintile_returns.csv: columns factor, Q1..Q5, spread (Q5-Q1 mean return)
    # Merge with IC
    if "factor" in quint_df.columns and "factor" in ic_sum.columns:
        merged_q = quint_df.merge(ic_sum[["factor", "ic_mean", "icir"]], on="factor", how="left")

        if "spread" in merged_q.columns:
            # Direction mismatch: IC > 0 but spread < 0, or IC < 0 but spread > 0
            merged_q["ic_sign"]     = np.sign(merged_q["ic_mean"])
            merged_q["spread_sign"] = np.sign(merged_q["spread"])
            merged_q["direction_ok"] = merged_q["ic_sign"] == merged_q["spread_sign"]
            merged_q["issue"] = "OK"
            merged_q.loc[~merged_q["direction_ok"] & (merged_q["ic_mean"].abs() > 0.01), "issue"] = "WRONG_DIR"

            wrong_dir = merged_q[merged_q["issue"] == "WRONG_DIR"].sort_values("ic_mean", key=abs, ascending=False)
            ok_factors = merged_q[merged_q["issue"] == "OK"]

            print(f"  Factors with correct quintile direction  : {len(ok_factors)}")
            print(f"  Factors with WRONG quintile direction    : {len(wrong_dir)}")

            if len(wrong_dir) > 0:
                print("\n  WRONG DIRECTION factors (IC sign ≠ Q5-Q1 sign):")
                display_cols = ["factor", "ic_mean", "icir", "spread", "Q1", "Q3", "Q5"]
                display_cols = [c for c in display_cols if c in wrong_dir.columns]
                print(wrong_dir[display_cols].to_string(index=False))

                print("\n  ROOT CAUSES:")
                for _, row in wrong_dir.iterrows():
                    fac   = row["factor"]
                    ic    = row.get("ic_mean", np.nan)
                    spr   = row.get("spread", np.nan)
                    if "ret_1" in fac or "ret_2" in fac or "ret_1w" in fac:
                        note = "Short-term reversal: IC from Spearman (rank) vs arithmetic mean quintile diverge when outliers present"
                    elif any(x in fac for x in ["vol", "var", "cvar", "dd", "atr", "kurt", "skew"]):
                        note = "Risk factor: negative IC is expected (high risk → low return). Spread confirms but sign is reversed."
                    else:
                        note = "Investigate: possible outlier contamination in Q1/Q5 returns or CS-rank inversion"
                    print(f"    {fac:<30} IC={ic:+.4f}  spread={spr:+.4f}  → {note}")

            merged_q.to_csv(DATA_DIR / "crowding_quintile_audit.csv", index=False)
            print(f"\n  Saved → data/crowding_quintile_audit.csv")

            # Plot: scatter IC_mean vs spread, colour by direction_ok
            fig, ax = plt.subplots(figsize=(10, 7))
            ok_mask   = merged_q["direction_ok"]
            err_mask  = ~merged_q["direction_ok"]

            ax.scatter(merged_q.loc[ok_mask,  "ic_mean"],
                       merged_q.loc[ok_mask,  "spread"],
                       c="#2196F3", alpha=0.7, label="Direction OK", s=40)
            ax.scatter(merged_q.loc[err_mask, "ic_mean"],
                       merged_q.loc[err_mask, "spread"],
                       c="#F44336", marker="X", s=80, alpha=0.9, label="WRONG direction")

            for _, row in merged_q[err_mask].iterrows():
                ax.annotate(row["factor"],
                            xy=(row["ic_mean"], row["spread"]),
                            fontsize=6.5, ha="center", va="bottom",
                            color="#B71C1C")

            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.axvline(0, color="grey", lw=0.8, ls="--")
            # Perfect direction line
            xlim = ax.get_xlim()
            ax.plot(xlim, xlim, "k--", lw=0.8, alpha=0.4, label="Perfect alignment")

            ax.set_xlabel("IC_IS (Spearman)", fontsize=11)
            ax.set_ylabel("Quintile Spread Q5−Q1  (annualised %)", fontsize=11)
            ax.set_title(
                "Quintile Direction Audit — IS Data Only\n"
                "Points should be in Q2/Q4 (IC and spread same sign). Red X = wrong direction.",
                fontsize=12
            )
            ax.legend(fontsize=9)
            plt.tight_layout()
            fig.savefig(FIG_DIR / "crowding_quintile_audit.png", dpi=150)
            plt.close(fig)
            print("  Saved → figures/crowding_quintile_audit.png")
        else:
            print("  [WARN] 'spread' column missing from factor_quintile_returns.csv")
    else:
        print("  [WARN] Unexpected column structure in quintile / IC summary files")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("CROWDING & DECAY DIAGNOSTIC COMPLETE")
print("=" * 65)
print(f"\n  Data outputs:")
print(f"    data/crowding_corr_matrix.csv       — {len(cs_factors)}×{len(cs_factors)} correlation matrix")
print(f"    data/crowding_redundant_pairs.csv   — {len(redundant) if 'redundant' in dir() else '?'} pairs with |r| > {CROWD_THRESHOLD}")
print(f"    data/crowding_7day_decay.csv        — IC day0/3/5/10 per factor")
print(f"    data/crowding_contrarian_oos.csv    — IS vs OOS for sign-flip factors")
print(f"    data/crowding_quintile_audit.csv    — quintile direction vs IC sign")

if "redundant" in dir() and len(redundant) > 0:
    print(f"\n  ⚠ Factor crowding: {len(redundant)} redundant pairs (|r| > {CROWD_THRESHOLD})")
    # Most crowded single factors
    from collections import Counter
    flat = redundant["factor_a"].tolist() + redundant["factor_b"].tolist()
    top_crowded = Counter(flat).most_common(10)
    print(f"  Factors appearing in most redundant pairs (candidates to DROP):")
    for fac, cnt in top_crowded:
        print(f"    {fac:<35} appears in {cnt} redundant pairs")

elapsed = _time.time() - _t0
print(f"\nDone in {elapsed:.1f}s")
