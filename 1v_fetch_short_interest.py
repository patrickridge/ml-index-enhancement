"""
1v_fetch_short_interest.py - Consolidated short interest from FINRA.

Four short-interest columns are declared in the panel and all four are empty,
so short interest has never been tested here. It is worth testing: it is one of
the better-documented cross-sectional anomalies, and unlike most of the factor
zoo it has an economic story that does not require anyone to be irrational -
shorting is costly and constrained, so a large short position is informative
about who is willing to pay to hold it.

FINRA publishes consolidated short interest for all US-listed equities twice a
month, free and without a key. Server-side filtering on symbol means the whole
S&P 500 costs about thirty requests rather than paging through millions of rows.

Point-in-time. Short interest is measured at a settlement date and published
about eight calendar days later, so the settlement date is not when the number
became knowable. Every row is stamped with settlement + PUBLICATION_LAG and
merged backward onto month ends, the same discipline 1u applies to filings. Use
the settlement date directly and you hand the model a week of hindsight.

Coverage starts 2017-12, which is where FINRA's API history begins. Earlier
months get NaN rather than a fabricated value.

Usage:
    python 1v_fetch_short_interest.py

Reads:  data/panel_monthly.parquet, data/spx_weights.parquet, data/prices.parquet
Writes: data/short_interest.parquet   (auto-detected by 1h)
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
OUT_PATH = DATA_DIR / "short_interest.parquet"

API_URL = ("https://api.finra.org/data/group/otcMarket/name/"
           "consolidatedShortInterest")
BATCH = 20          # symbols per request; ~200 rows each, well under the limit
PAGE = 5000
SLEEP = 0.4

# FINRA settles on the 15th and month end, and publishes about eight calendar
# days later. Erring long is the safe direction: it can only delay information,
# never leak it early.
PUBLICATION_LAG = pd.Timedelta(days=8)

UA = os.environ.get("FINRA_USER_AGENT",
                    "ml-index-enhancement research (github.com/patrickridge)")


def clean_ticker(tk: str) -> str:
    """AAPL.O -> AAPL, BRK.B -> BRK-B, matching FINRA's symbol spelling."""
    tk = tk.strip().replace("_", ".")
    parts = [p for p in tk.split(".") if p]
    if len(parts) > 1 and parts[-1].upper() in ("N", "O", "OQ", "A", "BAT"):
        parts = parts[:-1]
    return "-".join(p.upper() for p in parts) if len(parts) > 1 else parts[0].upper()


def fetch_batch(session, symbols: list) -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        body = {
            "limit": PAGE,
            "offset": offset,
            "domainFilters": [{"fieldName": "symbolCode", "values": symbols}],
        }
        r = session.post(API_URL, json=body, timeout=45)
        if r.status_code == 429:
            time.sleep(5)
            continue
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < PAGE:
            break
        offset += PAGE
    return pd.DataFrame(rows)


def main():
    import requests

    panel = pd.read_parquet(DATA_DIR / "panel_monthly.parquet",
                            columns=["date", "ticker"])
    panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)
    tickers = sorted(panel["ticker"].unique())
    sym_map = {t: clean_ticker(t) for t in tickers}
    symbols = sorted(set(sym_map.values()))

    print(f"{len(tickers)} tickers -> {len(symbols)} FINRA symbols")

    session = requests.Session()
    session.headers.update({
        "User-Agent": UA,
        "Content-Type": "application/json",
        "Accept": "application/json",
    })

    frames = []
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        try:
            df = fetch_batch(session, batch)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  batch {i // BATCH + 1}: {type(e).__name__}")
        done = min(i + BATCH, len(symbols))
        if done % 100 < BATCH or done == len(symbols):
            got = sum(len(f) for f in frames)
            print(f"  {done}/{len(symbols)} symbols | {got:,} rows")
        time.sleep(SLEEP)

    if not frames:
        raise SystemExit("Nothing fetched from FINRA.")

    si = pd.concat(frames, ignore_index=True)
    si["settlementDate"] = pd.to_datetime(si["settlementDate"])
    for c in ["currentShortPositionQuantity", "previousShortPositionQuantity",
              "averageDailyVolumeQuantity", "daysToCoverQuantity"]:
        si[c] = pd.to_numeric(si.get(c), errors="coerce")

    print(f"\nRaw: {len(si):,} rows | {si['symbolCode'].nunique()} symbols | "
          f"{si['settlementDate'].min().date()} to "
          f"{si['settlementDate'].max().date()}")

    # The date the number became public, not the date it describes.
    si["available"] = si["settlementDate"] + PUBLICATION_LAG

    # Shares outstanding, to turn a raw short count into a percentage. 1c's
    # market cap divided by price is the same implied share count it used.
    px = pd.read_parquet(DATA_DIR / "prices.parquet",
                         columns=["date", "ticker", "close"])
    px["date"] = pd.to_datetime(px["date"])
    px = (px.sort_values("date").groupby("ticker").resample("ME", on="date")
            .last().drop(columns=["ticker", "date"], errors="ignore")
            .reset_index())
    caps = pd.read_parquet(DATA_DIR / "spx_weights.parquet")[
        ["date", "ticker", "mktcap"]]
    caps["date"] = pd.to_datetime(caps["date"]) + pd.offsets.MonthEnd(0)
    px["date"] = pd.to_datetime(px["date"]) + pd.offsets.MonthEnd(0)
    shares = caps.merge(px, on=["date", "ticker"], how="inner")
    shares["shares_out"] = shares["mktcap"] / shares["close"].replace(0, np.nan)
    shares = shares[["date", "ticker", "shares_out"]]

    # Map FINRA symbols back onto panel tickers, then asof-merge on the date
    # the figure was publishable.
    rev = pd.DataFrame({"ticker": list(sym_map.keys()),
                        "symbolCode": list(sym_map.values())})
    si = si.merge(rev, on="symbolCode", how="inner")
    si = si.sort_values("available")

    merged = pd.merge_asof(
        panel.sort_values("date"),
        si[["available", "ticker", "currentShortPositionQuantity",
            "previousShortPositionQuantity", "averageDailyVolumeQuantity",
            "daysToCoverQuantity"]].rename(columns={"available": "date"}),
        on="date", by="ticker", direction="backward",
        tolerance=pd.Timedelta(days=75),   # two missed prints and it is stale
    )
    merged = merged.merge(shares, on=["date", "ticker"], how="left")

    cur = merged["currentShortPositionQuantity"]
    prev = merged["previousShortPositionQuantity"]

    out = merged[["date", "ticker"]].copy()
    out["short_pct_float"] = cur / merged["shares_out"].replace(0, np.nan)
    out["short_interest_ratio"] = merged["daysToCoverQuantity"]
    out["short_change_2w"] = (cur - prev) / prev.replace(0, np.nan)
    # Crowded and hard to exit: a large position relative to the float that
    # would also take many days to cover.
    out["short_squeeze_risk"] = (out["short_pct_float"]
                                 * merged["daysToCoverQuantity"])

    for c in [c for c in out.columns if c not in ("date", "ticker")]:
        s = out[c].replace([np.inf, -np.inf], np.nan)
        out[c] = s.clip(s.quantile(0.005), s.quantile(0.995))

    out.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved -> {OUT_PATH}")
    cov = out[out["date"] >= "2018-01-01"]
    print(f"{'column':<24}{'coverage':>10}{'from 2018':>11}{'distinct/mo':>13}")
    for c in [c for c in out.columns if c not in ("date", "ticker")]:
        print(f"  {c:<22}{out[c].notna().mean() * 100:>9.1f}%"
              f"{cov[c].notna().mean() * 100:>10.1f}%"
              f"{cov.groupby('date')[c].nunique().mean():>13.0f}")
    print("\nFINRA history starts 2017-12, so earlier months are NaN by")
    print("construction rather than filled with anything invented.")
    print("\nNext: rerun 1h_feature_engineering.py.")


if __name__ == "__main__":
    main()
