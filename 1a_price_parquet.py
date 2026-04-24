"""
1a_price_parquet.py - Parse the legacy Wind OHLC workbook into prices.parquet.

The original data.xlsx is a wide sheet: each ticker occupies a block of
columns separated by empty columns. We detect the column blocks, parse a
(date, O, H, L, C) panel for each ticker, and concatenate into one tidy
long-format parquet.

Inputs:
  data/data.xlsx          - raw Wind export (deprecated; historical only)
Outputs:
  data/prices.parquet     - long-format OHLC panel used by every later stage
"""

import pandas as pd
import numpy as np
from pathlib import Path


def find_blocks(df: pd.DataFrame, min_block_width: int = 10):
    sub = df.iloc[:, 1:]  # exclude date col
    is_empty_col = sub.isna().all(axis=0).to_numpy()

    blocks = []
    start = None
    for j, empty in enumerate(is_empty_col):
        if not empty and start is None:
            start = j
        if empty and start is not None:
            end = j - 1
            if (end - start + 1) >= min_block_width:
                blocks.append((start + 1, end + 1))  # shift back to df col index
            start = None

    if start is not None:
        end = len(is_empty_col) - 1
        if (end - start + 1) >= min_block_width:
            blocks.append((start + 1, end + 1))

    return blocks


def parse_dates(series: pd.Series) -> pd.Series:

    d = pd.to_datetime(series, errors="coerce")
    if d.isna().mean() > 0.2:
        s = series.astype(str).str.replace(r"\s+", "", regex=True)
        d = pd.to_datetime(s, errors="coerce", format="%Y%m%d")
    return d


def clean_tickers(tickers: pd.Series) -> pd.Series:
    t = tickers.astype(str).str.strip()
    t = t.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return t


def parse_block(df: pd.DataFrame, date_col: int, block_start: int, block_end: int, value_name: str):
    # dates: from row 6 onward (pandas index 5 onward)
    dates_raw = df.iloc[5:, date_col]
    dates = parse_dates(dates_raw)

    # tickers: row 5 (pandas index 4)
    tickers = clean_tickers(df.iloc[4, block_start:block_end + 1])

    # values: row 6 onward
    vals = df.iloc[5:, block_start:block_end + 1].copy()

    # build wide with explicit column names
    wide = pd.DataFrame(vals.to_numpy(), columns=tickers, index=dates)
    wide.index.name = "date"

    # drop bad date rows
    wide = wide[~wide.index.isna()]

    # drop columns where ticker is NaN
    wide = wide.loc[:, ~wide.columns.isna()]

    # if duplicate tickers exist, keep the first occurrence (rare but possible)
    if wide.columns.duplicated().any():
        wide = wide.loc[:, ~wide.columns.duplicated()]

    # ensure numeric
    wide = wide.apply(pd.to_numeric, errors="coerce")

    # melt to long (stable; always gives 'ticker' column)
    long = (
        wide.reset_index()
            .melt(id_vars="date", var_name="ticker", value_name=value_name)
    )

    # drop missing values for this field
    long = long.dropna(subset=[value_name])

    return long


def main():
    input_path = Path("data.xlsx")   # change to your file path
    output_path = Path("data/prices.parquet")

    df = pd.read_excel(input_path, sheet_name=0, header=None, engine="openpyxl")

    blocks = find_blocks(df, min_block_width=10)
    if len(blocks) < 4:
        raise ValueError(f"Expected 4 blocks (O/H/L/C), but found {len(blocks)} blocks: {blocks}")

    block_names = ["open", "high", "low", "close"]
    parsed = []

    for name, (s, e) in zip(block_names, blocks[:4]):
        print(f"Parsing block {name}: cols {s}..{e}")
        parsed.append(parse_block(df, date_col=0, block_start=s, block_end=e, value_name=name))

    # merge with outer join
    out = parsed[0]
    for other in parsed[1:]:
        # safety: ensure keys exist
        for k in ["date", "ticker"]:
            if k not in other.columns:
                raise ValueError(f"Missing key column '{k}' in {other.columns.tolist()}")
        out = out.merge(other, on=["date", "ticker"], how="outer")

    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    out = out.dropna(subset=["open", "high", "low", "close"], how="all")

    # remove impossible negatives
    for c in ["open", "high", "low", "close"]:
        out.loc[out[c] < 0, c] = np.nan

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved: {output_path} | rows={len(out):,} | unique tickers={out['ticker'].nunique():,}")
    print(out.head())

if __name__ == "__main__":
    main()
