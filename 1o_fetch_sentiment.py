"""
1o_fetch_sentiment.py — Fetch News Sentiment Data
====================================================
Fetches company news from Finnhub (free tier: 60 calls/min) and scores
headlines using VADER sentiment analysis (open source, no API key needed).

Features produced (per stock per month):
  - sentiment_mean_30d:     mean VADER compound score of headlines (30d)
  - sentiment_std_30d:      std of sentiment scores (30d) — controversy signal
  - sentiment_momentum_30d: change in mean sentiment vs prior 30d
  - news_volume_30d:        number of news articles (30d, log-transformed)

Academic basis:
  Tetlock (2007) — media pessimism predicts downward market pressure
  Loughran & McDonald (2011) — finance-specific sentiment outperforms generic
  Da, Engelberg & Gao (2015) — news sentiment predicts short-term returns
  Hutto & Gilbert (2014) — VADER: robust social media sentiment analysis

Dependencies:
  pip install finnhub-python vaderSentiment

Output:
  data/sentiment.parquet — monthly panel of news sentiment features

Rate limits: Finnhub free tier allows 60 calls/min.
Run time: ~30-60 min (rate-limited to 60 req/min, ~500 tickers).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import time as _time
import os

_t0 = _time.time()
DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "sentiment.parquet"

print("=" * 65)
print("NEWS SENTIMENT DATA FETCH (Finnhub + VADER)")
print("=" * 65)

# ── Check dependencies ────────────────────────────────────────────────────────
try:
    import finnhub
except ImportError:
    print("[ERROR] finnhub-python not installed. Run: pip install finnhub-python")
    raise SystemExit(1)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("[ERROR] vaderSentiment not installed. Run: pip install vaderSentiment")
    raise SystemExit(1)

# ── Config ────────────────────────────────────────────────────────────────────
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
if not FINNHUB_API_KEY:
    print("[WARN] FINNHUB_API_KEY not set in environment. Set it with:")
    print("  export FINNHUB_API_KEY='your_key_here'")
    print("  Get a free key at: https://finnhub.io/register")
    raise SystemExit(1)

try:
    from config import START_DATE
    start = START_DATE
except ImportError:
    start = "2010-01-01"

# Load S&P 500 tickers
try:
    constit = pd.read_parquet(DATA_DIR / "constituents.parquet")
    tickers = sorted(constit["ticker"].unique().tolist())
    print(f"Loaded {len(tickers)} tickers from constituents.parquet")
except FileNotFoundError:
    print("[WARN] constituents.parquet not found — cannot proceed")
    raise SystemExit(1)

# ── Finnhub client ────────────────────────────────────────────────────────────
client = finnhub.Client(api_key=FINNHUB_API_KEY)
analyzer = SentimentIntensityAnalyzer()

RATE_LIMIT_DELAY = 1.05  # ~60 calls/min → 1s per call with margin


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH NEWS AND SCORE SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_news_for_ticker(ticker: str, from_date: str, to_date: str) -> list:
    """Fetch company news from Finnhub for a single ticker."""
    try:
        news = client.company_news(ticker, _from=from_date, to=to_date)
        return news if isinstance(news, list) else []
    except Exception:
        return []


def score_headlines(articles: list) -> pd.DataFrame:
    """Score article headlines with VADER sentiment."""
    records = []
    for art in articles:
        headline = art.get("headline", "")
        if not headline:
            continue

        scores = analyzer.polarity_scores(headline)
        records.append({
            "datetime": pd.Timestamp(art.get("datetime", 0), unit="s"),
            "headline": headline,
            "compound": scores["compound"],
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
        })

    return pd.DataFrame(records)


def fetch_all_sentiment() -> pd.DataFrame:
    """
    Fetch news and compute sentiment for all tickers.

    Finnhub free tier limits: 1 year of news per request.
    We fetch in yearly chunks from start_date to today.
    """
    print(f"\nFetching news sentiment for {len(tickers)} tickers...")
    print(f"  Rate limit: ~60 requests/min (1s delay per call)")

    all_scored = []
    n_tickers = len(tickers)
    total_articles = 0

    # Fetch recent news (last 12 months) — Finnhub free tier limitation
    today = pd.Timestamp.today()
    from_date = (today - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    for i, ticker in enumerate(tickers):
        if (i + 1) % 25 == 0:
            elapsed = _time.time() - _t0
            print(f"  Progress: {i+1}/{n_tickers} | "
                  f"articles so far: {total_articles:,} | {elapsed:.0f}s")

        articles = fetch_news_for_ticker(ticker, from_date, to_date)
        _time.sleep(RATE_LIMIT_DELAY)

        if not articles:
            continue

        scored = score_headlines(articles)
        if scored.empty:
            continue

        scored["ticker"] = ticker
        all_scored.append(scored)
        total_articles += len(scored)

    print(f"\n  Total articles scored: {total_articles:,}")

    if not all_scored:
        return pd.DataFrame()

    return pd.concat(all_scored, ignore_index=True)


def compute_sentiment_features(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate scored headlines into monthly features per ticker.
    """
    if scored.empty:
        return pd.DataFrame()

    scored = scored.copy()
    scored["datetime"] = pd.to_datetime(scored["datetime"])
    scored["date"] = scored["datetime"].dt.date

    records = []

    for ticker in scored["ticker"].unique():
        tk = scored[scored["ticker"] == ticker].sort_values("datetime")

        # Generate monthly features
        months = pd.date_range(
            tk["datetime"].min().replace(day=1),
            pd.Timestamp.today(),
            freq="MS",
        )

        for month_start in months:
            month_end = month_start + pd.offsets.MonthEnd(0)

            # 30-day lookback
            mask_30d = (
                (tk["datetime"] >= month_end - pd.Timedelta(days=30))
                & (tk["datetime"] <= month_end)
            )
            articles_30d = tk[mask_30d]

            # Prior 30 days (for momentum)
            mask_prior = (
                (tk["datetime"] >= month_end - pd.Timedelta(days=60))
                & (tk["datetime"] < month_end - pd.Timedelta(days=30))
            )
            articles_prior = tk[mask_prior]

            if len(articles_30d) == 0:
                continue

            mean_30d = articles_30d["compound"].mean()
            std_30d = articles_30d["compound"].std() if len(articles_30d) > 1 else 0.0
            volume_30d = len(articles_30d)

            # Sentiment momentum: current mean - prior mean
            prior_mean = articles_prior["compound"].mean() if len(articles_prior) > 0 else mean_30d
            momentum = mean_30d - prior_mean

            records.append({
                "ticker": ticker,
                "date": month_end,
                "sentiment_mean_30d": mean_30d,
                "sentiment_std_30d": std_30d,
                "sentiment_momentum_30d": momentum,
                "news_volume_30d": np.log1p(volume_30d),
            })

    result = pd.DataFrame(records)
    if not result.empty:
        result["date"] = pd.to_datetime(result["date"])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    scored = fetch_all_sentiment()
    features = compute_sentiment_features(scored)

    if features.empty:
        print("\n[WARN] No sentiment features computed — creating empty parquet")
        features = pd.DataFrame(columns=[
            "ticker", "date", "sentiment_mean_30d", "sentiment_std_30d",
            "sentiment_momentum_30d", "news_volume_30d",
        ])

    features.to_parquet(OUT_PATH, index=False)
    elapsed = _time.time() - _t0
    print(f"\nSaved: {OUT_PATH} | {len(features):,} rows | {elapsed:.0f}s")
    if not features.empty:
        print(f"Tickers with data: {features['ticker'].nunique()}")
        print(f"Date range: {features['date'].min()} → {features['date'].max()}")
        print(f"\nSample features:")
        print(features.describe().round(3))
