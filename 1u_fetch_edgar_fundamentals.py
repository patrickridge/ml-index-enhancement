"""
1u_fetch_edgar_fundamentals.py - Fundamentals from SEC EDGAR, point-in-time.

The panel declares twelve fundamental columns and every one is empty, so the
working factor library is entirely price and volume derived. On the rebuilt
data no factor survives BHY, and the most likely reason is not that the market
is efficient but that value and quality were never tested: the columns that
would carry them never had data in them.

EDGAR is the right source for US equities. It is free, needs no key, covers
full history rather than a rolling window, and - the part that matters - every
fact carries the date it was *filed*, not just the period it describes. Apple's
Q4 figures are not knowable in October; they become knowable when the 10-K is
filed in November. Using period-end would leak weeks of hindsight into every
row, which is exactly the class of bug this repo keeps finding.

So every value here is stamped with `filed` and merged backward onto month
ends. A month sees only what had actually been published by then.

Requires a contact string, because the SEC asks automated clients to identify
themselves and will refuse otherwise:

    export SEC_USER_AGENT="Your Name your@email.com"
    python 1u_fetch_edgar_fundamentals.py

Reads:  data/panel_monthly.parquet   (for the ticker list)
        data/spx_weights.parquet     (market caps, for the valuation ratios)
Writes: data/fundamental.parquet     (auto-detected by 1h, enables Cat 9)
"""

import json
import os
import time

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
OUT_PATH = DATA_DIR / "fundamental.parquet"
CIK_CACHE = DATA_DIR / "edgar_cik_map.json"
FACT_CACHE = DATA_DIR / "edgar_facts"

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# The SEC publishes a 10 requests/second guideline, but that is not what binds
# here: companyfacts payloads run to several megabytes each, so their limiter
# trips on bandwidth long before request count. At 0.12s every single request
# came back 429, and once tripped the block persists rather than easing. Two a
# second is slower than necessary on paper and actually works.
SLEEP = 0.5
MAX_RETRIES = 5

UA = os.environ.get("SEC_USER_AGENT")

# XBRL concept names, most-preferred first. Filers are inconsistent about which
# tag they use, particularly for revenue, so each field tries several.
CONCEPTS = {
    "net_income":  ["NetIncomeLoss",
                    "ProfitLoss"],
    "revenue":     ["RevenueFromContractWithCustomerExcludingAssessedTax",
                    "Revenues",
                    "SalesRevenueNet",
                    "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit": ["GrossProfit"],
    "assets":      ["Assets"],
    "equity":      ["StockholdersEquity",
                    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liabilities": ["Liabilities"],
    "eps":         ["EarningsPerShareDiluted",
                    "EarningsPerShareBasicAndDiluted"],
    "op_income":   ["OperatingIncomeLoss"],
    "dep_amort":   ["DepreciationDepletionAndAmortization",
                    "DepreciationAmortizationAndAccretionNet"],
    "cfo":         ["NetCashProvidedByUsedInOperatingActivities",
                    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "buybacks":    ["PaymentsForRepurchaseOfCommonStock"],
}

# Flows are summed over four quarters; stocks are a snapshot at the date.
FLOW_FIELDS = {"net_income", "revenue", "gross_profit", "op_income",
               "dep_amort", "cfo", "buybacks", "eps"}


_session = None


def fetch_json(url: str) -> dict:
    """
    GET and decode one EDGAR endpoint.

    Uses requests rather than urllib because the stock macOS Python does not
    point urllib at a certificate bundle, so every https call fails on cert
    verification. requests carries its own via certifi, and handles gzip.
    """
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
        _session.headers.update({
            "User-Agent": UA,
            "Accept-Encoding": "gzip, deflate",
        })
    delay = SLEEP
    for attempt in range(MAX_RETRIES):
        r = _session.get(url, timeout=30)
        if r.status_code == 429:
            # Back off rather than keep hammering. Honour Retry-After when the
            # server sends one, otherwise double each time. Giving up quietly
            # here is how a run ends with 628 failures and nothing cached.
            wait = float(r.headers.get("Retry-After", 0)) or delay * (2 ** attempt)
            wait = min(wait, 120)
            print(f"    429, waiting {wait:.0f}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("rate limited after retries")


def clean_ticker(tk: str) -> str:
    """AAPL.O -> AAPL, BRK.B -> BRK-B. EDGAR uses the hyphenated class form."""
    tk = tk.strip().replace("_", ".")
    parts = [p for p in tk.split(".") if p]
    if len(parts) > 1 and parts[-1].upper() in ("N", "O", "OQ", "A", "BAT"):
        parts = parts[:-1]
    return "-".join(p.upper() for p in parts) if len(parts) > 1 else parts[0].upper()


def load_cik_map() -> dict:
    if CIK_CACHE.exists():
        return json.loads(CIK_CACHE.read_text())
    print("Fetching ticker -> CIK map from SEC...")
    data = fetch_json(TICKERS_URL)
    m = {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
    CIK_CACHE.write_text(json.dumps(m))
    print(f"  {len(m):,} tickers mapped")
    return m


def extract_series(facts: dict, tags: list, is_flow: bool) -> pd.DataFrame:
    """
    Pull one concept out of a companyfacts blob.

    Returns date-stamped observations where `date` is the FILING date, which is
    when the number became public. Keeps the latest filing for a given period so
    restatements supersede the original rather than duplicating it.
    """
    units = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        if tag not in units:
            continue
        for unit_name, rows in units[tag].get("units", {}).items():
            if unit_name not in ("USD", "USD/shares"):
                continue
            recs = [{"filed": r["filed"], "end": r["end"], "val": r["val"],
                     "start": r.get("start"), "form": r.get("form", "")}
                    for r in rows if r.get("filed") and r.get("val") is not None]
            if not recs:
                continue
            df = pd.DataFrame(recs)
            df["filed"] = pd.to_datetime(df["filed"])
            df["end"] = pd.to_datetime(df["end"])
            df["start"] = pd.to_datetime(df["start"])
            df = df[df["form"].str.startswith(("10-K", "10-Q"))]
            if df.empty:
                continue

            if is_flow:
                # EDGAR carries both the quarter and the full year for flow
                # concepts, and a 10-K reports both at once. Summing four of a
                # mixed set double-counts: Apple came out at $780bn of revenue
                # against a real figure near $420bn. Keep quarters only, then
                # four of them make a trailing year.
                dur = (df["end"] - df["start"]).dt.days
                df = df[dur.between(80, 100)]
                if len(df) < 4:
                    return pd.DataFrame(columns=["filed", "end", "val"])

            df = (df.sort_values("filed")
                    .drop_duplicates(subset=["end"], keep="last")
                    .sort_values("end"))
            return df[["filed", "end", "val"]]
    return pd.DataFrame(columns=["filed", "end", "val"])


def build_ticker_frame(facts: dict) -> pd.DataFrame:
    """One row per reporting period, each stamped with when it was filed."""
    out, filed_cols = None, []
    for field, tags in CONCEPTS.items():
        is_flow = field in FLOW_FIELDS
        s = extract_series(facts, tags, is_flow)
        if s.empty:
            continue
        s = s.rename(columns={"val": field, "filed": f"_filed_{field}"})
        if is_flow:
            # Four quarters make a trailing year, comparable across companies
            # with different year ends.
            s[field] = s[field].rolling(4, min_periods=4).sum()
        filed_cols.append(f"_filed_{field}")
        # Join on the period, not on (period, filing date). Different concepts
        # in the same filing can carry different filed stamps, and an outer
        # join on both scatters one period across several half-empty rows.
        out = s if out is None else out.merge(s, on="end", how="outer")
    if out is None or out.empty:
        return pd.DataFrame()

    # A row is only usable once every part of it has been published, so take
    # the latest filing date across the fields. Conservative by construction,
    # which is the direction to err in.
    out["filed"] = out[filed_cols].max(axis=1)
    out = out.drop(columns=filed_cols).dropna(subset=["filed"])
    return out.sort_values("filed").reset_index(drop=True)


def main():
    if not UA:
        raise SystemExit(
            "SEC_USER_AGENT is not set. The SEC requires automated clients to\n"
            "identify themselves with a contact string, for example:\n\n"
            '    export SEC_USER_AGENT="Your Name your@email.com"\n')

    FACT_CACHE.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(DATA_DIR / "panel_monthly.parquet",
                            columns=["date", "ticker"])
    panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)
    tickers = sorted(panel["ticker"].unique())

    cik_map = load_cik_map()
    resolved = {t: cik_map.get(clean_ticker(t)) for t in tickers}
    found = {t: c for t, c in resolved.items() if c}
    print(f"{len(found)}/{len(tickers)} tickers resolved to a CIK\n")

    frames, failed = [], []
    for i, (tkr, cik) in enumerate(found.items(), 1):
        cache = FACT_CACHE / f"{cik}.json"
        try:
            if cache.exists():
                facts = json.loads(cache.read_text())
            else:
                facts = fetch_json(FACTS_URL.format(cik=cik))
                cache.write_text(json.dumps(facts))
                time.sleep(SLEEP)
            df = build_ticker_frame(facts)
            if df.empty:
                failed.append(tkr); continue
            df["ticker"] = tkr
            frames.append(df)
        except Exception as e:
            failed.append(tkr)
            if len(failed) <= 5:
                print(f"  {tkr}: {type(e).__name__}")
        if i % 50 == 0 or i == len(found):
            print(f"  {i}/{len(found)}  ok={len(frames)}  failed={len(failed)}")

    if not frames:
        raise SystemExit("Nothing fetched. Check SEC_USER_AGENT and network.")

    fund = pd.concat(frames, ignore_index=True)
    print(f"\nRaw filings: {len(fund):,} rows, {fund['ticker'].nunique()} tickers")

    # Attach each filing to the month ends it was knowable in. merge_asof on the
    # filing date is what enforces point-in-time: a month picks up the most
    # recent filing at or before its end, never a later one.
    fund = fund.sort_values("filed")
    month_ends = panel.sort_values("date")
    merged = pd.merge_asof(
        month_ends, fund.rename(columns={"filed": "date"}),
        on="date", by="ticker", direction="backward",
        tolerance=pd.Timedelta(days=550),   # a filing older than ~18m is stale
    )

    # Valuation ratios need a market cap, which 1c already computes.
    wpath = DATA_DIR / "spx_weights.parquet"
    if wpath.exists():
        caps = pd.read_parquet(wpath)[["date", "ticker", "mktcap"]]
        caps["date"] = pd.to_datetime(caps["date"]) + pd.offsets.MonthEnd(0)
        merged = merged.merge(caps, on=["date", "ticker"], how="left")
    else:
        merged["mktcap"] = np.nan

    def safe(num, den):
        den = den.replace(0, np.nan)
        return num / den

    out = merged[["date", "ticker"]].copy()
    out["pe_ratio"]    = safe(merged["mktcap"], merged.get("net_income"))
    out["pb_ratio"]    = safe(merged["mktcap"], merged.get("equity"))
    out["ps_ratio"]    = safe(merged["mktcap"], merged.get("revenue"))
    ebitda             = merged.get("op_income", 0) + merged.get("dep_amort", 0)
    ev                 = merged["mktcap"] + merged.get("liabilities", 0)
    out["ev_ebitda"]   = safe(ev, ebitda)
    out["roe"]         = safe(merged.get("net_income"), merged.get("equity"))
    out["roa"]         = safe(merged.get("net_income"), merged.get("assets"))
    out["gross_margin"] = safe(merged.get("gross_profit"), merged.get("revenue"))
    out["debt_to_equity"] = safe(merged.get("liabilities"), merged.get("equity"))
    # Sloan (1996): accruals are the gap between reported profit and the cash
    # that actually arrived. A wide gap is low earnings quality.
    out["earnings_quality"] = safe(
        merged.get("cfo", np.nan) - merged.get("net_income", np.nan),
        merged.get("assets"))
    out["buyback_yield"] = safe(merged.get("buybacks"), merged["mktcap"])

    for col, base in [("revenue_growth_yoy", "revenue"), ("eps_growth_yoy", "eps")]:
        if base in merged.columns:
            g = merged.groupby("ticker")[base]
            out[col] = g.pct_change(12)
        else:
            out[col] = np.nan

    # Ratios built from near-zero denominators explode; clip rather than let one
    # row dominate a cross-sectional rank.
    for c in out.columns:
        if c in ("date", "ticker"):
            continue
        s = out[c].replace([np.inf, -np.inf], np.nan)
        lo, hi = s.quantile(0.005), s.quantile(0.995)
        out[c] = s.clip(lo, hi)

    out.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved -> {OUT_PATH}")
    print(f"{len(out):,} rows | {out['ticker'].nunique()} tickers\n")
    print(f"{'column':<22}{'coverage':>10}{'distinct/mo':>13}")
    ref = out.copy()
    for c in [c for c in out.columns if c not in ("date", "ticker")]:
        cov = out[c].notna().mean() * 100
        dm = ref.groupby("date")[c].nunique().mean()
        print(f"  {c:<20}{cov:>9.1f}%{dm:>13.0f}")
    if failed:
        print(f"\n{len(failed)} tickers returned nothing: {', '.join(failed[:10])}"
              f"{' ...' if len(failed) > 10 else ''}")
    print("\nNext: rerun 1h_feature_engineering.py to pick up Cat 9.")


if __name__ == "__main__":
    main()
