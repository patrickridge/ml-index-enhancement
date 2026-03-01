"""
config.py — shared constants for the main pipeline
(2_lgbm_backtest.py, 3_pca_rp_backtest.py, 4_benchmark_spx.py)
"""
from pathlib import Path

DATA_DIR = Path("data")

# ── Date splits ───────────────────────────────────────────────────────────────
# With 20 years of data (~2004-2024):
#   Option A — keep GFC cut (cleaner regime, ~168 months from 2010):
#     START_DATE = "2010-01-01", TRAIN_END = "2020-12-31", VALID_END = "2022-12-31"
#   Option B — use all 20 years (~240 months, includes pre-GFC):
#     START_DATE = "2004-01-01", TRAIN_END = "2018-12-31", VALID_END = "2021-12-31"
#
# Current: Option A (recommended — avoids GFC regime in training)
START_DATE    = "2010-01-01"   # cut pre-GFC data
TRAIN_END     = "2020-12-31"   # ~132 months training (extended with 20yr data)
VALID_END     = "2022-12-31"   # ~24 months validation; test = 2023+ (~24 months)

# ── Portfolio construction ────────────────────────────────────────────────────
TOP_N         = 50             # long-only portfolio size
BOTTOM_N      = 50             # short leg size
LONG_FRAC     = 0.10           # top/bottom fraction for long-short

# ── Walk-forward retraining ───────────────────────────────────────────────────
RETRAIN_EVERY = 12             # retrain every N months (expanding window)

# ── PCA Risk Parity ───────────────────────────────────────────────────────────
VOL_LOOKBACK  = 252            # trading days for covariance estimation
MIN_OBS       = 150            # minimum daily observations per ticker
N_PCA         = 10             # PCA components to keep
RIDGE         = 1e-3           # ridge regularization on covariance

# ── LightGBM ─────────────────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=5,
    min_child_samples=60,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
)
