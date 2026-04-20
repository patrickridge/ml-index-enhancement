"""
4e_stock_pitch_backtest.py — InvestSoc Fundamental Stock Pitch Backtest
=========================================================================
Backtest of the 6 fundamental analyst picks:
  Mercado Libre (MELI), CME Group (CME), Salesforce (CRM),
  Delta Airlines (DAL), Maersk (AMKBY), Edwards Lifesciences (EW)

Strategy: a +1pp tilt over SPX for each pick. MELI and AMKBY are not S&P 500
constituents, so they are added as pure +1% overlays. The remaining SPX weights
are proportionally underweighted so total portfolio weight stays at 1.0.

Window: validation period (2023-01-31 → 2024-06-30), matching the one used
for the CS-Transformer model in config.py.

Comparisons:
  • SPX cap-weighted benchmark (^GSPC via yfinance)
  • CS-Transformer IE strategy (from data/bt_ie_cs_transformer.csv)
  • Bonus: average ML rank percentile of the 6 picks over the window

Outputs:
  data/bt_ie_stock_pitch.csv                 monthly port/bench/active returns
  figures/stock_pitch_cumulative.png         cumulative NAV chart (3 lines)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths / config ───────────────────────────────────────────────────────────
DATA_DIR  = Path("data")
FIG_DIR   = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

# Validation window from config.py (2023-01-01 → 2024-06-30)
START = pd.Timestamp("2023-01-01")
END   = pd.Timestamp("2024-06-30")

# Tickers: 4 in panel + 2 to fetch from yfinance
PICKS_IN_PANEL = {
    "CME":  "CME.O",
    "CRM":  "CRM.N",
    "DAL":  "DAL.N",
    "EW":   "EW.N",
}
# Try candidates in order until one returns data
YFINANCE_PICKS = {
    "MELI":   ["MELI"],
    "Maersk": ["AMKBY", "AMKAF", "MAERSK-B.CO"],
}

TILT = 0.01  # +1pp overweight per pick

print("=" * 70)
print("INVESTSOC STOCK PITCH BACKTEST")
print("=" * 70)
print(f"Window: {START.date()} → {END.date()} (validation period)")
print(f"Picks: {list(PICKS_IN_PANEL) + list(YFINANCE_PICKS)}")


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE STATS (mirrors 4d_benchmark_spx.py & 4b_index_enhancement.py)
# ═══════════════════════════════════════════════════════════════════════════════

def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 3:
        return dict(months=len(r), ann=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    ann    = (1 + r).prod() ** (12 / len(r)) - 1
    vol    = r.std(ddof=1) * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else np.nan
    nav    = (1 + r).cumprod()
    maxdd  = (nav / nav.cummax() - 1).min()
    return dict(months=len(r), ann=ann, vol=vol, sharpe=sharpe, maxdd=maxdd)


def ie_stats(port: pd.Series, bench: pd.Series) -> dict:
    """Active-return stats: tracking error, IR, alpha, hit rate."""
    df = pd.concat([port.rename("p"), bench.rename("b")], axis=1).dropna()
    if len(df) < 3:
        return {}
    active = df["p"] - df["b"]
    n = len(active)
    ann_alpha = (1 + active).prod() ** (12 / n) - 1
    te        = active.std(ddof=1) * np.sqrt(12)
    ir        = ann_alpha / te if te > 0 else np.nan
    hit_rate  = (active > 0).mean()
    active_nav = (1 + active).cumprod()
    max_act_dd = (active_nav / active_nav.cummax() - 1).min()
    return dict(
        months=n, ann_alpha=ann_alpha, track_err=te, info_ratio=ir,
        hit_rate=hit_rate, max_active_dd=max_act_dd,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Monthly returns for the 6 picks
# ═══════════════════════════════════════════════════════════════════════════════

def month_end_returns_from_prices(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """Daily close → month-end close → monthly return."""
    sub = prices[prices["ticker"] == ticker][["date", "close"]].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    sub = sub.set_index("date").sort_index()
    me  = sub["close"].resample("ME").last()
    return me.pct_change().dropna()


def fetch_yfinance_monthly(ticker_candidates: list, start, end) -> tuple:
    """Try each candidate ticker; return (series, actual_ticker_used)."""
    try:
        import yfinance as yf
    except ImportError:
        print("  [ERROR] yfinance not installed; pip install yfinance")
        return pd.Series(dtype=float), None

    for tk in ticker_candidates:
        try:
            df = yf.download(
                tk,
                start=(start - pd.DateOffset(months=2)).strftime("%Y-%m-%d"),
                end=(end + pd.DateOffset(days=7)).strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True,
            )
            if df is None or df.empty:
                continue
            # yfinance may return multi-index cols; flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close_col = "Close" if "Close" in df.columns else "Adj Close"
            me = df[close_col].resample("ME").last()
            rets = me.pct_change().dropna()
            if len(rets) >= 3:
                return rets, tk
        except Exception as e:
            print(f"    [WARN] {tk} fetch failed: {e}")
            continue
    return pd.Series(dtype=float), None


def build_monthly_returns() -> pd.DataFrame:
    """Returns monthly-return DataFrame indexed by month-end, columns = pretty names."""
    print("\nStep 1: Building monthly returns for the 6 picks...")

    # From prices.parquet
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    ret_dict = {}
    for pretty, tk in PICKS_IN_PANEL.items():
        r = month_end_returns_from_prices(prices, tk)
        ret_dict[pretty] = r
        print(f"  {pretty:7s} ({tk:10s}): {len(r)} months from panel")

    # From yfinance
    for pretty, candidates in YFINANCE_PICKS.items():
        r, actual = fetch_yfinance_monthly(candidates, START, END)
        ret_dict[pretty] = r
        src = f"yfinance ({actual})" if actual else "FAILED"
        print(f"  {pretty:7s} ({candidates[0]:10s}): {len(r)} months from {src}")

    # Align
    rets = pd.DataFrame(ret_dict).sort_index()
    rets.index = pd.to_datetime(rets.index)

    # Align to month-end — normalize to last-day-of-month
    rets.index = rets.index + pd.offsets.MonthEnd(0)
    # Deduplicate any duplicate month-end indices
    rets = rets[~rets.index.duplicated(keep="last")]

    # Restrict to validation window
    rets = rets[(rets.index >= START) & (rets.index <= END + pd.offsets.MonthEnd(0))]

    print(f"\n  Coverage in {START.date()} → {END.date()}:")
    for col in rets.columns:
        n = rets[col].notna().sum()
        exp = len(rets)
        print(f"    {col:7s}: {n}/{exp} months "
              f"({'✓' if n == exp else '⚠ missing ' + str(exp-n)})")
    return rets


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build the 1%-tilt over SPX portfolio
# ═══════════════════════════════════════════════════════════════════════════════

def build_tilt_returns(picks_rets: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly portfolio:
      - Each pick in SPX gets +1pp over its cap weight
      - Each pick NOT in SPX (MELI, Maersk) gets +1pp standalone overlay
      - Remainder of SPX proportionally underweighted to sum to 1.0
      - Forward 1m return from scores parquet for all SPX stocks

    Returns DataFrame indexed by month-end with columns port_ret, bench_ret, active_ret.
    """
    print("\nStep 2: Building 1%-tilt portfolio...")

    # SPX weights (monthly, month-end dates) + fwd_ret_1m for SPX stocks
    spx_w = pd.read_parquet(DATA_DIR / "spx_weights.parquet")
    spx_w["date"] = pd.to_datetime(spx_w["date"]) + pd.offsets.MonthEnd(0)

    # Stock returns: use scores parquet's fwd_ret_1m (total returns, per-ticker per month)
    scores = pd.read_parquet(DATA_DIR / "scores_cs_transformer.parquet")
    scores["date"] = pd.to_datetime(scores["date"]) + pd.offsets.MonthEnd(0)
    rets_panel = scores[["date", "ticker", "fwd_ret_1m"]].copy()

    # Mapping pretty → panel ticker
    panel_map = PICKS_IN_PANEL  # pretty -> panel_ticker
    non_spx_picks = list(YFINANCE_PICKS.keys())  # MELI, Maersk

    records = []
    for m in sorted(picks_rets.index.unique()):
        w_snap = spx_w[spx_w["date"] == m].copy()
        r_snap = rets_panel[rets_panel["date"] == m].copy()

        if w_snap.empty or r_snap.empty:
            continue

        # Baseline SPX weights normalized to 1.0 (safety)
        w_snap["w"] = w_snap["spx_weight"] / w_snap["spx_weight"].sum()

        # Merge returns onto weights
        wr = w_snap.merge(r_snap[["ticker", "fwd_ret_1m"]], on="ticker", how="left")
        wr["fwd_ret_1m"] = wr["fwd_ret_1m"].fillna(0.0)

        # Benchmark = pure SPX cap-weighted return
        bench_ret = float((wr["w"] * wr["fwd_ret_1m"]).sum())

        # ── Build tilted portfolio ─────────────────────────────────────────
        wr_tilt = wr.copy()

        # Apply +1pp tilt to in-SPX picks
        in_spx_overweight = 0.0
        for pretty, panel_tk in panel_map.items():
            mask = wr_tilt["ticker"] == panel_tk
            if mask.any():
                wr_tilt.loc[mask, "w"] = wr_tilt.loc[mask, "w"] + TILT
                in_spx_overweight += TILT
            else:
                # Pick not in SPX this month — treat as overlay instead
                in_spx_overweight += 0.0

        # Non-SPX overlays (MELI, Maersk): add rows with +1pp each
        overlay_rows = []
        overlay_total = 0.0
        for pretty in non_spx_picks:
            r = picks_rets.loc[m, pretty] if pretty in picks_rets.columns else np.nan
            if pd.notna(r):
                overlay_rows.append({"ticker": pretty, "w": TILT, "fwd_ret_1m": float(r)})
                overlay_total += TILT
            # If the ticker is missing this month, we DON'T add it (weight=0)

        # Also add in-panel picks that weren't in SPX this month as overlays
        for pretty, panel_tk in panel_map.items():
            if not (wr_tilt["ticker"] == panel_tk).any():
                r = picks_rets.loc[m, pretty] if pretty in picks_rets.columns else np.nan
                if pd.notna(r):
                    overlay_rows.append({"ticker": panel_tk, "w": TILT, "fwd_ret_1m": float(r)})
                    overlay_total += TILT

        # Rescale non-pick SPX weights so total = 1.0
        # Total overweight applied = in_spx_overweight + overlay_total
        total_tilt = in_spx_overweight + overlay_total
        pick_tickers_in_spx = [panel_map[p] for p in panel_map if (wr["ticker"] == panel_map[p]).any()]
        non_pick_mask = ~wr_tilt["ticker"].isin(pick_tickers_in_spx)
        non_pick_sum = wr_tilt.loc[non_pick_mask, "w"].sum()
        scale = (1.0 - total_tilt - in_spx_overweight) / non_pick_sum if non_pick_sum > 0 else 1.0
        # Wait — correct math:
        # Before tilt, sum(w) = 1.0. After adding TILT to each of k in-SPX picks, sum = 1 + k*TILT.
        # After adding overlays, sum = 1 + (k + m)*TILT where m = overlay count.
        # To restore sum=1, rescale non-pick rows by: (1 - TILT*(k+m)) / non_pick_sum_original.
        # But the picks' weights include TILT, so we should rescale only non-pick rows.
        # The sum of non-pick rows originally = 1 - sum(pick_original_weights_in_spx).
        # Let S = sum of pick_original_weights_in_spx. non_pick_original = 1 - S.
        # After tilt: picks = S + k*TILT. non_picks need to equal 1 - (S + k*TILT) - overlay_total.
        # So scale = (1 - S - k*TILT - overlay_total) / (1 - S).
        # Implementing directly:
        pick_weights_original = wr.loc[wr["ticker"].isin(pick_tickers_in_spx), "w"].sum()
        k = len(pick_tickers_in_spx)
        S = pick_weights_original
        desired_non_pick_sum = 1.0 - (S + k * TILT) - overlay_total
        current_non_pick_sum = 1.0 - S
        if current_non_pick_sum > 0:
            scale = desired_non_pick_sum / current_non_pick_sum
        else:
            scale = 1.0
        wr_tilt.loc[non_pick_mask, "w"] = wr_tilt.loc[non_pick_mask, "w"] * scale

        # Compute portfolio return
        port_ret = float((wr_tilt["w"] * wr_tilt["fwd_ret_1m"]).sum())
        port_ret += sum(row["w"] * row["fwd_ret_1m"] for row in overlay_rows)

        # Sanity: total weight
        total_w = wr_tilt["w"].sum() + sum(row["w"] for row in overlay_rows)

        records.append({
            "date":       m,
            "port_ret":   port_ret,
            "bench_ret":  bench_ret,
            "active_ret": port_ret - bench_ret,
            "total_w":    total_w,
            "n_overlays": len(overlay_rows),
        })

    bt = pd.DataFrame(records).set_index("date").sort_index()
    print(f"  Built {len(bt)} months, avg total weight={bt['total_w'].mean():.4f} "
          f"(should be ~1.0)")
    return bt[["port_ret", "bench_ret", "active_ret"]]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ML opinion on these 6 stocks
# ═══════════════════════════════════════════════════════════════════════════════

def ml_rank_percentile() -> float:
    """Average cross-sectional rank percentile of the 4 in-panel picks."""
    scores = pd.read_parquet(DATA_DIR / "scores_cs_transformer.parquet")
    scores["date"] = pd.to_datetime(scores["date"]) + pd.offsets.MonthEnd(0)
    scores = scores[(scores["date"] >= START) & (scores["date"] <= END + pd.offsets.MonthEnd(0))]

    # rank within each month (1 = worst, n = best)
    scores["rank_pct"] = scores.groupby("date")["score"].rank(pct=True)

    picks = list(PICKS_IN_PANEL.values())
    picks_ranks = scores[scores["ticker"].isin(picks)].copy()
    if picks_ranks.empty:
        return np.nan
    return float(picks_ranks["rank_pct"].mean())


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Step 1: monthly returns for the 6 picks
    picks_rets = build_monthly_returns()
    if picks_rets.empty:
        raise RuntimeError("No pick returns — aborting")

    # Step 2: 1% tilt portfolio
    bt = build_tilt_returns(picks_rets)
    bt.index.name = "date"
    bt.to_csv(DATA_DIR / "bt_ie_stock_pitch.csv")
    print(f"\n  Saved: data/bt_ie_stock_pitch.csv ({len(bt)} rows)")

    # Step 3: CS-Transformer IE for same window
    cs_bt = pd.read_csv(DATA_DIR / "bt_ie_cs_transformer.csv", parse_dates=["date"])
    cs_bt["date"] = cs_bt["date"] + pd.offsets.MonthEnd(0)
    cs_bt = cs_bt.set_index("date").sort_index()
    cs_bt = cs_bt[(cs_bt.index >= START) & (cs_bt.index <= END + pd.offsets.MonthEnd(0))]

    # Each strategy on its own full available period
    port_pitch = bt["port_ret"]
    bench_spx  = bt["bench_ret"]
    port_cs    = cs_bt["port_ret"]
    bench_cs   = cs_bt["bench_ret"]  # CS-T's own SPX benchmark (on its available months)

    # Alpha / IR / tracking error: computed on each strategy's own overlap with its benchmark
    ie_pitch = ie_stats(port_pitch, bench_spx)           # 18 months
    ie_cs    = ie_stats(port_cs,    bench_cs)            # 13 months

    # Standalone stats on each's own full period
    s_pitch = perf_stats(port_pitch)
    s_spx   = perf_stats(bench_spx)
    s_cs    = perf_stats(port_cs)

    avg_ml_rank = ml_rank_percentile()

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS — Validation Window 2023-01 → 2024-06")
    print("=" * 70)
    hdr = f"{'Strategy':<30} {'Ann Ret':>8} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>8}" \
          f" {'α vs SPX':>9} {'TE':>7} {'IR':>6}"
    print(hdr)
    print("-" * len(hdr))

    def row(label, s, ie=None):
        ann = s.get("ann", np.nan); vol = s.get("vol", np.nan)
        shp = s.get("sharpe", np.nan); dd = s.get("maxdd", np.nan)
        n   = s.get("months", 0)
        extras = ""
        if ie:
            extras = (f" {ie.get('ann_alpha', np.nan)*100:>8.2f}%"
                      f" {ie.get('track_err', np.nan)*100:>6.2f}%"
                      f" {ie.get('info_ratio', np.nan):>6.2f}")
        else:
            extras = f" {'—':>9} {'—':>7} {'—':>6}"
        print(f"{label:<30} {ann*100:>7.2f}% {vol*100:>6.2f}% "
              f"{shp:>7.2f} {dd*100:>7.2f}%" + extras + f"  [n={n}]")

    row(f"6-stock 1% Tilt over SPX",   s_pitch, ie_pitch)
    row(f"SPX (benchmark)",             s_spx,   None)
    row(f"CS-Transformer IE (ours)",    s_cs,    ie_cs)
    print(f"\n  Note: pitch portfolio & SPX benchmark cover 18 months; "
          f"CS-Transformer only has {s_cs.get('months', 0)} months available.")

    print(f"\nBonus: Average ML rank percentile of the 4 in-panel picks "
          f"(CME/CRM/DAL/EW): {avg_ml_rank*100:.1f}%")
    print(f"  (>50% = model liked them, <50% = model was bearish)")

    # ── Chart ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    nav_pitch = (1 + port_pitch).cumprod()
    nav_spx   = (1 + bench_spx).cumprod()
    nav_cs    = (1 + port_cs).cumprod()
    nav_pitch.plot(ax=ax, label="Fundamental 1% Tilt", linewidth=2)
    nav_cs.plot(ax=ax, label="CS-Transformer IE (ML)", linewidth=2)
    nav_spx.plot(ax=ax, label="SPX", linewidth=2, linestyle="--", color="grey")
    ax.set_title(f"InvestSoc Stock Pitch Backtest\n"
                 f"{START.date()} → {END.date()}  (pitch: 18 months, "
                 f"CS-T: {len(port_cs)} months)")
    ax.set_ylabel("Cumulative NAV (1.0 = start)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = FIG_DIR / "stock_pitch_cumulative.png"
    fig.savefig(out_png, dpi=140)
    print(f"\n  Saved chart: {out_png}")


if __name__ == "__main__":
    main()
