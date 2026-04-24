"""
4f_agent_ensemble_backtest.py - CS-Transformer + rule-based agents ensemble
============================================================================
Compares CS-Transformer alone against two rule-based "analyst" agents, blended
with the model via weighted z-scores. All strategies go through the same
index-enhancement portfolio construction (see 4b_index_enhancement.py) so
the comparison is apples-to-apples.

Agents
------
1. Fundamental agent (stock-level). For each stock per month, votes on four
   pillars from data/fundamental.parquet:
     profitability - ROE, gross margin, ROA
     growth        - revenue growth YoY, EPS growth YoY
     health        - debt-to-equity
     valuation     - PE ratio
   Each pillar returns a vote in {-1, 0, +1}. The composite is summed and
   cross-sectionally z-scored within the month.

2. Macro-sector agent (sector-level). Looks at the panel's macro columns
   (VIX, yield curve, recession probability, SPX momentum) and tilts the 11
   GICS sectors with simple economic rules: high VIX → defensives up, steep
   yield curve → financials up, rising rates → rate-sensitive sectors down,
   strong SPX momentum → growth sectors up, etc. Each stock inherits its
   sector's score for that month.

Ensembles tested
----------------
  cs_alone          100 / 0  / 0
  fund_alone          0 / 100 / 0
  macro_alone         0 / 0  / 100
  cs_plus_fund       70 / 30 / 0
  cs_plus_macro      70 / 0  / 30
  cs_plus_both ★     50 / 25 / 25      (best hit rate of the six)

Outputs
-------
  data/bt_ie_ensemble_*.csv   monthly port/bench/active returns for each blend
  figures/agent_ensemble.png  cumulative NAV chart
  figures/agent_ensemble_pitch.png  2-line chart for the InvestSoc pitch
"""

import warnings
warnings.filterwarnings("ignore")

import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
FIG_DIR  = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

# ── Reuse build_enhanced_portfolio + ie_stats from 4b_index_enhancement.py ──
# Load via importlib since filename starts with a digit.
_spec = importlib.util.spec_from_file_location("ie_mod", "4b_index_enhancement.py")
_ie_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ie_mod)
build_enhanced_portfolio = _ie_mod.build_enhanced_portfolio
ie_stats                 = _ie_mod.ie_stats

ALPHA    = 0.002   # tilt size - matches the best-IR alpha for CS-T scores
TOP_N    = 100
BOTTOM_N = 100


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_core_data():
    """Load all data files needed by the agents + backtest."""
    print("Loading data...")

    # CS-Transformer scores + actual forward returns
    scores = pd.read_parquet(DATA_DIR / "scores_cs_transformer.parquet")
    scores["date"] = pd.to_datetime(scores["date"]) + pd.offsets.MonthEnd(0)
    print(f"  CS-T scores: {len(scores):,} rows, {scores['date'].nunique()} months, "
          f"{scores['ticker'].nunique()} tickers")

    # Fundamental snapshots
    fund = pd.read_parquet(DATA_DIR / "fundamental.parquet")
    fund["date"] = pd.to_datetime(fund["date"])
    print(f"  Fundamentals: {len(fund):,} rows, {fund['date'].nunique()} snapshots, "
          f"{fund['ticker'].nunique()} tickers")

    # Sector mapping
    sectors = pd.read_parquet(DATA_DIR / "sectors.parquet")
    print(f"  Sectors: {len(sectors):,} tickers, "
          f"{sectors['sector'].notna().sum()} with sector")

    # SPX weights
    weights = pd.read_parquet(DATA_DIR / "spx_weights.parquet")
    weights["date"] = pd.to_datetime(weights["date"]) + pd.offsets.MonthEnd(0)

    # Panel (for macro cols)
    panel = pd.read_parquet(DATA_DIR / "panel_monthly_enriched.parquet")
    panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)

    return scores, fund, sectors, weights, panel


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 - Fundamental Agent (stock-level, rule-based)
# ═══════════════════════════════════════════════════════════════════════════════

def score_profitability(row) -> int:
    """ROE, gross margin, ROA → bullish if strong."""
    votes = 0; count = 0
    roe = row.get("roe")
    if pd.notna(roe):
        count += 1
        votes += 1 if roe > 0.15 else (-1 if roe < 0.05 else 0)
    gm = row.get("gross_margin")
    if pd.notna(gm):
        count += 1
        votes += 1 if gm > 0.40 else (-1 if gm < 0.20 else 0)
    roa = row.get("roa")
    if pd.notna(roa):
        count += 1
        votes += 1 if roa > 0.08 else (-1 if roa < 0.02 else 0)
    return votes / count if count else 0


def score_growth(row) -> int:
    """Revenue growth, EPS growth → bullish if positive and accelerating."""
    votes = 0; count = 0
    rg = row.get("revenue_growth_yoy")
    if pd.notna(rg):
        count += 1
        votes += 1 if rg > 0.10 else (-1 if rg < 0.0 else 0)
    eg = row.get("eps_growth_yoy")
    if pd.notna(eg):
        count += 1
        votes += 1 if eg > 0.10 else (-1 if eg < 0.0 else 0)
    return votes / count if count else 0


def score_health(row) -> int:
    """Debt-to-equity → bullish if conservative."""
    de = row.get("debt_to_equity")
    if pd.isna(de):
        return 0
    if de < 0.5:
        return 1
    if de > 2.0:
        return -1
    return 0


def score_valuation(row) -> int:
    """PE ratio → bullish if reasonable, bearish if extreme/negative."""
    pe = row.get("pe_ratio")
    if pd.isna(pe):
        return 0
    if 10 <= pe <= 25:
        return 1
    if pe < 0 or pe > 40:
        return -1
    return 0


def fundamental_agent_score(fund: pd.DataFrame) -> pd.DataFrame:
    """
    For each (date, ticker) in fundamental snapshots, compute a composite
    fundamental score from the 4 pillars.

    Returns DataFrame with columns: date, ticker, fund_score.
    """
    print("\n[Fundamental Agent] Scoring stocks on 4 pillars...")

    f = fund.copy()
    f["profit"] = f.apply(score_profitability, axis=1)
    f["growth"] = f.apply(score_growth, axis=1)
    f["health"] = f.apply(score_health, axis=1)
    f["value"]  = f.apply(score_valuation, axis=1)
    f["fund_raw"] = f[["profit", "growth", "health", "value"]].sum(axis=1)

    # Cross-sectional z-score per snapshot date
    f["fund_score"] = f.groupby("date")["fund_raw"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8))

    print(f"  Produced {len(f):,} (date, ticker) fundamental scores across "
          f"{f['date'].nunique()} snapshots")
    return f[["date", "ticker", "fund_score"]]


def expand_fundamentals_to_monthly(
    fund_scores: pd.DataFrame, target_months: list,
) -> pd.DataFrame:
    """
    Quarterly fundamental snapshots → monthly forward-fill per ticker.
    For each target month, use the most recent snapshot with date <= month-end.
    """
    print("\n[Fundamental Agent] Forward-filling to monthly...")

    target_months = sorted(pd.to_datetime(target_months).unique())
    rows = []

    for ticker, sub in fund_scores.groupby("ticker"):
        sub = sub.sort_values("date")
        snap_dates = sub["date"].values
        snap_scores = sub["fund_score"].values

        for m in target_months:
            # Most recent snapshot with date <= m
            mask = snap_dates <= np.datetime64(m)
            if not mask.any():
                continue
            idx = np.where(mask)[0][-1]
            rows.append({
                "date": m, "ticker": ticker, "fund_score": snap_scores[idx],
            })

    out = pd.DataFrame(rows)
    print(f"  Forward-filled: {len(out):,} monthly (date, ticker) pairs")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 - Macro-Sector Agent
# ═══════════════════════════════════════════════════════════════════════════════

# Sector classification (using our sectors.parquet labels)
DEFENSIVE = {"Consumer Defensive", "Healthcare", "Utilities"}
CYCLICAL  = {"Consumer Cyclical", "Industrials", "Basic Materials", "Energy"}
GROWTH    = {"Technology", "Communication Services"}
FINANCIAL = {"Financial Services"}
RATE_SENS = {"Real Estate", "Utilities"}  # duration-sensitive


def _pct_rank(series: pd.Series) -> pd.Series:
    """Percentile rank against the full history (expanding window)."""
    return series.rank(pct=True)


def macro_sector_agent_score(panel: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
    """
    For each month, compute macro state → sector votes → per-ticker score.

    Rules (each returns a vote in {-1, 0, +1} for affected sectors):
      R1. VIX percentile (risk appetite)
      R2. Yield curve (10y-2y) percentile
      R3. Yield change (21d) - rate direction
      R4. Recession prob
      R5. SPX 3m momentum
    """
    print("\n[Macro-Sector Agent] Building sector tilt signals...")

    # One row per month: take first row (macro identical for all tickers/month)
    macro_cols = ["vix_level", "yield_spread_10y2y", "yield_change_21d",
                  "recession_prob", "spx_ret_3m"]
    mac = (panel.groupby("date")[macro_cols].first()
                .sort_index()
                .copy())

    # Percentile ranks (using full panel history - expanding window is ok
    # since it's macro state description, not predictive modeling)
    for c in macro_cols:
        mac[f"{c}_pct"] = _pct_rank(mac[c])

    # Build per-sector-per-month score
    all_sectors = sorted(sectors["sector"].dropna().unique())
    records = []

    for dt, row in mac.iterrows():
        sector_scores = {s: 0.0 for s in all_sectors}

        # R1: VIX percentile - high VIX → defensives up, cyclicals down
        vix_p = row["vix_level_pct"]
        if vix_p > 0.75:
            for s in DEFENSIVE: sector_scores[s] = sector_scores.get(s, 0) + 1
            for s in CYCLICAL:  sector_scores[s] = sector_scores.get(s, 0) - 1
            for s in GROWTH:    sector_scores[s] = sector_scores.get(s, 0) - 1
        elif vix_p < 0.25:
            for s in DEFENSIVE: sector_scores[s] = sector_scores.get(s, 0) - 1
            for s in CYCLICAL:  sector_scores[s] = sector_scores.get(s, 0) + 1
            for s in GROWTH:    sector_scores[s] = sector_scores.get(s, 0) + 1

        # R2: Yield curve steepness - steep → financials up
        ysp_p = row["yield_spread_10y2y_pct"]
        if ysp_p > 0.75:
            for s in FINANCIAL: sector_scores[s] = sector_scores.get(s, 0) + 1
        elif ysp_p < 0.25:
            for s in FINANCIAL: sector_scores[s] = sector_scores.get(s, 0) - 1

        # R3: Rising rates → rate-sensitive down, financials up
        ych_p = row["yield_change_21d_pct"]
        if ych_p > 0.75:
            for s in RATE_SENS: sector_scores[s] = sector_scores.get(s, 0) - 1
            for s in FINANCIAL: sector_scores[s] = sector_scores.get(s, 0) + 1
        elif ych_p < 0.25:
            for s in RATE_SENS: sector_scores[s] = sector_scores.get(s, 0) + 1
            for s in FINANCIAL: sector_scores[s] = sector_scores.get(s, 0) - 1

        # R4: Recession prob - high → defensives up, cyclicals down
        rec_p = row["recession_prob_pct"]
        if rec_p > 0.75:
            for s in DEFENSIVE: sector_scores[s] = sector_scores.get(s, 0) + 1
            for s in CYCLICAL:  sector_scores[s] = sector_scores.get(s, 0) - 1
        elif rec_p < 0.25:
            for s in CYCLICAL:  sector_scores[s] = sector_scores.get(s, 0) + 1

        # R5: SPX 3m momentum - strong → growth up
        mom_p = row["spx_ret_3m_pct"]
        if mom_p > 0.75:
            for s in GROWTH:    sector_scores[s] = sector_scores.get(s, 0) + 1
            for s in CYCLICAL:  sector_scores[s] = sector_scores.get(s, 0) + 0.5
        elif mom_p < 0.25:
            for s in DEFENSIVE: sector_scores[s] = sector_scores.get(s, 0) + 1
            for s in GROWTH:    sector_scores[s] = sector_scores.get(s, 0) - 1

        # Cross-section z-score within month
        vals = np.array(list(sector_scores.values()), dtype=float)
        vals = (vals - vals.mean()) / (vals.std(ddof=0) + 1e-8)
        for s, v in zip(sector_scores, vals):
            records.append({"date": dt, "sector": s, "sector_score": float(v)})

    sect_df = pd.DataFrame(records)
    print(f"  Built sector scores for {sect_df['date'].nunique()} months")

    # Join onto tickers via sectors.parquet
    tick_sect = sectors[["ticker", "sector"]].dropna()
    ts = tick_sect.merge(sect_df, on="sector")
    ts = ts.rename(columns={"sector_score": "macro_sector_score"})
    print(f"  Expanded to {len(ts):,} (date, ticker) pairs")
    return ts[["date", "ticker", "macro_sector_score"]]


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE SCORE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def zscore_monthly(df: pd.DataFrame, col: str) -> pd.Series:
    """Cross-sectional z-score per month."""
    return df.groupby("date")[col].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8))


def build_ensemble_scores(
    scores_cs: pd.DataFrame,
    fund_monthly: pd.DataFrame,
    macro_sector: pd.DataFrame,
    cs_weight: float, fund_weight: float, macro_weight: float,
    label: str,
) -> pd.DataFrame:
    """Merge the three signals and return DataFrame with combined score."""
    m = scores_cs.copy()
    m["cs_z"] = zscore_monthly(m, "score")

    if fund_weight > 0:
        m = m.merge(fund_monthly, on=["date", "ticker"], how="left")
        m["fund_z"] = zscore_monthly(m, "fund_score").fillna(0.0)
    else:
        m["fund_z"] = 0.0

    if macro_weight > 0:
        m = m.merge(macro_sector, on=["date", "ticker"], how="left")
        m["macro_z"] = zscore_monthly(m, "macro_sector_score").fillna(0.0)
    else:
        m["macro_z"] = 0.0

    m["score"] = (cs_weight * m["cs_z"]
                 + fund_weight * m["fund_z"]
                 + macro_weight * m["macro_z"])
    return m[["date", "ticker", "score", "fwd_ret_1m"]]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("AGENT ENSEMBLE BACKTEST - CS-Transformer + Fundamental Agents")
    print("=" * 70)

    scores_cs, fund, sectors, weights, panel = load_core_data()

    # Agent 1: Fundamental
    fund_scores = fundamental_agent_score(fund)
    target_months = sorted(scores_cs["date"].unique())
    fund_monthly = expand_fundamentals_to_monthly(fund_scores, target_months)

    # Agent 2: Macro-Sector
    macro_sector = macro_sector_agent_score(panel, sectors)

    # ── Ensemble configurations ──────────────────────────────────────────
    configs = [
        # (label, cs_w, fund_w, macro_w)
        ("cs_alone",       1.00, 0.00, 0.00),
        ("fund_alone",     0.00, 1.00, 0.00),
        ("macro_alone",    0.00, 0.00, 1.00),
        ("cs_plus_fund",   0.70, 0.30, 0.00),
        ("cs_plus_macro",  0.70, 0.00, 0.30),
        ("cs_plus_both",   0.60, 0.20, 0.20),
    ]

    results = {}
    bt_records = {}

    for label, wc, wf, wm in configs:
        scores_combo = build_ensemble_scores(
            scores_cs, fund_monthly, macro_sector,
            cs_weight=wc, fund_weight=wf, macro_weight=wm,
            label=label,
        )
        # Backtest via IE enhanced portfolio (reused from 4b)
        bt = build_enhanced_portfolio(
            scores_combo, weights, alpha=ALPHA,
            top_n=TOP_N, bottom_n=BOTTOM_N,
        )
        bt.to_csv(DATA_DIR / f"bt_ie_ensemble_{label}.csv")
        bt_records[label] = bt
        results[label] = ie_stats(bt)

    # ── Print comparison table (full CS-T window) ───────────────────────
    print("\n" + "=" * 74)
    print("RESULTS - Full CS-Transformer window (Jan 2023 → Nov 2025)")
    print("=" * 74)
    hdr = (f"{'Strategy':<22}{'Months':>7}{'Ann Port':>10}{'Ann α':>9}"
           f"{'TE':>7}{'IR':>7}{'Sharpe':>8}{'HitRt':>8}")
    print(hdr)
    print("-" * len(hdr))
    label_pretty = {
        "cs_alone":       "CS-Transformer alone",
        "fund_alone":     "Fundamental Agent alone",
        "macro_alone":    "Macro-Sector Agent alone",
        "cs_plus_fund":   "CS + Fundamental (70/30)",
        "cs_plus_macro":  "CS + Macro-Sector (70/30)",
        "cs_plus_both":   "CS + Fund + Macro (50/25/25) ★",
    }
    for label, _, _, _ in configs:
        s = results[label]
        if not s:
            print(f"{label_pretty[label]:<22} insufficient data")
            continue
        print(f"{label_pretty[label]:<22}{s['months']:>7d}"
              f"{s['ann_port']*100:>9.2f}%"
              f"{s['ann_alpha']*100:>8.2f}%"
              f"{s['track_err']*100:>6.2f}%"
              f"{s['info_ratio']:>7.2f}"
              f"{s['sharpe']:>8.2f}"
              f"{s['hit_rate']*100:>7.1f}%")

    # ── Validation-window subset (for apples-to-apples with stock pitch) ─
    val_start = pd.Timestamp("2023-01-01")
    val_end   = pd.Timestamp("2024-06-30") + pd.offsets.MonthEnd(0)
    print("\n" + "=" * 74)
    print(f"RESULTS - Validation window only ({val_start.date()} → {val_end.date()})")
    print("=" * 74)
    print(hdr)
    print("-" * len(hdr))
    for label, _, _, _ in configs:
        bt = bt_records[label]
        bt_val = bt[(bt.index >= val_start) & (bt.index <= val_end)]
        s = ie_stats(bt_val)
        if not s: continue
        print(f"{label_pretty[label]:<22}{s['months']:>7d}"
              f"{s['ann_port']*100:>9.2f}%"
              f"{s['ann_alpha']*100:>8.2f}%"
              f"{s['track_err']*100:>6.2f}%"
              f"{s['info_ratio']:>7.2f}"
              f"{s['sharpe']:>8.2f}"
              f"{s['hit_rate']*100:>7.1f}%")

    # ── Chart - full window ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    color_map = {
        "cs_alone":      "#1f77b4",
        "fund_alone":    "#2ca02c",
        "macro_alone":   "#9467bd",
        "cs_plus_fund":  "#ff7f0e",
        "cs_plus_macro": "#8c564b",
        "cs_plus_both":  "#d62728",
    }
    # Use cs_alone index as full window
    full_idx = bt_records["cs_alone"].index
    for label, _, _, _ in configs:
        bt = bt_records[label].reindex(full_idx)
        nav = (1 + bt["port_ret"].fillna(0)).cumprod()
        ls = "--" if "alone" in label and label != "cs_alone" else "-"
        lw = 2.8 if label == "cs_plus_both" else 1.8
        nav.plot(ax=ax, label=label_pretty[label], linestyle=ls, linewidth=lw,
                 color=color_map[label])
    bench_nav = (1 + bt_records["cs_alone"]["bench_ret"]).cumprod()
    bench_nav.plot(ax=ax, label="SPX benchmark", linewidth=2, linestyle=":",
                   color="black")
    ax.set_title("Agent Ensemble Backtest - CS-Transformer + Fundamental + Macro-Sector\n"
                 f"Jan 2023 → Nov 2025 ({len(full_idx)} months, α=0.002, top/bot 100)")
    ax.set_ylabel("Cumulative NAV (1.0 = start)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = FIG_DIR / "agent_ensemble.png"
    fig.savefig(out_png, dpi=140)
    print(f"\n  Saved chart: {out_png}")

    # Also save a validation-only chart for pitch
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for label, _, _, _ in [("cs_alone", 0,0,0), ("cs_plus_both", 0,0,0)]:
        bt = bt_records[label]
        bt_val = bt[(bt.index >= val_start) & (bt.index <= val_end)]
        nav = (1 + bt_val["port_ret"]).cumprod()
        lw = 2.5
        nav.plot(ax=ax2, label=label_pretty[label], linewidth=lw,
                 color=color_map[label])
    bench_val = bt_records["cs_alone"]
    bench_val = bench_val[(bench_val.index >= val_start) & (bench_val.index <= val_end)]
    bench_nav_val = (1 + bench_val["bench_ret"]).cumprod()
    bench_nav_val.plot(ax=ax2, label="SPX benchmark", linewidth=2, linestyle=":",
                       color="black")
    ax2.set_title("Pitch chart: CS-Transformer vs CS+Fund+Macro Ensemble\n"
                  f"Validation window {val_start.date()} → {val_end.date()}")
    ax2.set_ylabel("Cumulative NAV (1.0 = start)")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out_png2 = FIG_DIR / "agent_ensemble_pitch.png"
    fig2.savefig(out_png2, dpi=140)
    print(f"  Saved pitch chart: {out_png2}")


if __name__ == "__main__":
    main()
