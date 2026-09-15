"""
1s_refetch_prices.py - Re-fetch prices and volume on a real scale.

0_data_audit.py found two faults in data/prices.parquet. Returns are correct,
so everything derived from returns is sound, but:

  1. Each ticker's price series is multiplied by an arbitrary constant. The
     proof does not depend on knowing any real price: a uniform scale factor
     shifts log-prices without spreading them, and the cross-sectional log
     standard deviation is 2.79 against a realistic 0.85. Backing out the
     excess gives sd(log k) = 2.66. So the factor varies per ticker, which
     corrupts every feature built on price levels - log_mktcap, size_proxy,
     dollar_vol_*, amihud_illiq_*, kyle_lambda - and those are most of what
     survives BHY.

  2. Volume is all-or-nothing by ticker: 192 of 697 have it, 505 never do.
     The liquidity factors are therefore partly an indicator of whether the
     fetch succeeded rather than a measure of liquidity.

Writes a new file rather than touching prices.parquet, so the old and new can
be compared before anything is swapped. Nothing downstream changes until you
run 1t_validate_refetch.py and then move the file into place deliberately.

Delisted names will not come back, and that is correct: they are the zombies
from Notes item 22 and they should not be in the panel. Coverage is reported
per ticker so the gap is visible rather than silent.

Usage:
    python 1s_refetch_prices.py            # all tickers
    ML_REFETCH_LIMIT=20 python 1s_refetch_prices.py   # smoke test

Reads:  data/prices.parquet   (for the ticker list and date range)
Writes: data/prices_refetched.parquet
        data/refetch_coverage.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
IN_PATH  = DATA_DIR / "prices.parquet"
OUT_PATH = DATA_DIR / "prices_refetched.parquet"
COV_PATH = DATA_DIR / "refetch_coverage.csv"

CHUNK  = 40           # tickers per yfinance call
PAUSE  = 1.0          # seconds between chunks, to stay polite
LIMIT  = int(os.environ.get("ML_REFETCH_LIMIT", 0))


# .N, .O and .OQ identify an exchange and are dropped. Anything else after the
# dot is a share class and has to be kept: BRK.B is Berkshire class B, which
# Yahoo spells BRK-B, and dropping the suffix asks for "BRK", which is not a
# ticker at all. Getting this wrong silently loses the name.
EXCHANGE_SUFFIXES = {"N", "O", "OQ", "A_", "K", "P", "Z", "BAT"}


def to_yahoo(ric: str) -> str:
    """
    Reuters RIC to Yahoo symbol.

    AAPL.O -> AAPL, BRK.B -> BRK-B, BRK_B.N -> BRK-B. The exchange code has to
    come off before the share class is read, or BRK_B.N reduces to BRK.
    """
    parts = [p for p in ric.strip().replace("_", ".").split(".") if p]

    if len(parts) > 1 and parts[-1].upper() in EXCHANGE_SUFFIXES:
        parts = parts[:-1]

    if len(parts) > 1:
        return f"{parts[0].upper()}-{parts[-1].upper()}"

    # Trailing lowercase letter is the older RIC spelling of a share class,
    # so check it before uppercasing throws the case away.
    base = parts[0]
    if len(base) > 1 and base[-1].islower() and base[:-1].isupper():
        return f"{base[:-1]}-{base[-1].upper()}"
    return base.upper()


def main():
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("pip install yfinance")

    old = pd.read_parquet(IN_PATH)
    old["date"] = pd.to_datetime(old["date"])
    tickers = sorted(old["ticker"].unique())
    if LIMIT:
        tickers = tickers[:LIMIT]

    start = old["date"].min().strftime("%Y-%m-%d")
    end   = (old["date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    print(f"{len(tickers)} tickers | {start} to {end}")
    print(f"Writing to {OUT_PATH.name}, leaving {IN_PATH.name} untouched\n")

    sym_map = {t: to_yahoo(t) for t in tickers}
    frames, cov = [], []

    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        syms  = [sym_map[t] for t in batch]
        try:
            raw = yf.download(syms, start=start, end=end, auto_adjust=True,
                              progress=False, group_by="ticker",
                              threads=True, timeout=30)
        except Exception as e:
            print(f"  chunk {i//CHUNK + 1}: FAILED ({type(e).__name__})")
            for t in batch:
                cov.append({"ticker": t, "symbol": sym_map[t], "rows": 0,
                            "status": "fetch_error"})
            continue

        got = 0
        for t in batch:
            sym = sym_map[t]
            try:
                sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
                sub = sub.dropna(subset=["Close"])
            except (KeyError, TypeError):
                sub = pd.DataFrame()

            if sub.empty:
                cov.append({"ticker": t, "symbol": sym, "rows": 0,
                            "status": "no_data"})
                continue

            df = pd.DataFrame({
                "date":   pd.to_datetime(sub.index),
                "ticker": t,
                "open":   sub["Open"].values,
                "high":   sub["High"].values,
                "low":    sub["Low"].values,
                "close":  sub["Close"].values,
                "volume": sub["Volume"].values,
            })
            frames.append(df)
            got += 1
            cov.append({"ticker": t, "symbol": sym, "rows": len(df),
                        "status": "ok",
                        "vol_pct": float(df["volume"].gt(0).mean())})

        done = min(i + CHUNK, len(tickers))
        print(f"  {done:>4}/{len(tickers)}  chunk ok: {got}/{len(batch)}")
        time.sleep(PAUSE)

    if not frames:
        raise SystemExit("Nothing fetched. Check network access.")

    out = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    out.to_parquet(OUT_PATH, index=False)

    cov_df = pd.DataFrame(cov)
    cov_df.to_csv(COV_PATH, index=False)

    n_ok = int((cov_df["status"] == "ok").sum())
    print(f"\nSaved {OUT_PATH} | {len(out):,} rows | {n_ok}/{len(tickers)} tickers")
    print(f"Saved {COV_PATH}")

    # The headline check, and it needs no external price to interpret.
    last_yr = out[out["date"].dt.year == out["date"].dt.year.max()]
    px = last_yr.sort_values("date").groupby("ticker")["close"].last()
    px = px[px > 0]
    print(f"\nCross-sectional log-sd of price: {np.log(px).std():.2f}")
    print( "  old file 2.79 | realistic ~0.85")
    print(f"  median ${px.median():,.0f} | p95 ${px.quantile(.95):,.0f}")

    vol_ok = float(out.groupby("ticker")["volume"].apply(
        lambda s: s.gt(0).mean()).gt(0.9).mean())
    print(f"\nTickers with volume: {vol_ok:.0%}  (old file 28%)")
    print("\nNext: python 1t_validate_refetch.py")


if __name__ == "__main__":
    main()
