"""
config.py — shared constants for the main pipeline
(2_lgbm_backtest.py, 2b_nn_backtest.py, 3_pca_rp_backtest.py, 4_benchmark_spx.py)
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

# ── FT-Transformer (2b_nn_backtest.py) ───────────────────────────────────────
# Architecture: each of the N input features is embedded into d_model-dim vectors,
# then processed by multi-head self-attention to learn feature interactions.
# CPU training time: ~45-60 min (3 folds × ~15-20 min each). GPU: ~15-25 min total.
TRANSFORMER_PARAMS = dict(
    d_model=64,         # embedding dimension per feature
    n_heads=4,          # attention heads (must divide d_model)
    n_layers=3,         # transformer encoder layers
    dropout=0.1,        # regularisation
    lr=1e-3,            # Adam learning rate
    weight_decay=1e-4,  # L2 regularisation
    epochs=80,          # max training epochs (was 150 — converges well before 80)
    patience=10,        # early stopping patience (was 20 — halves worst-case time)
    batch_size=512,     # mini-batch size
)

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
    verbosity=-1,          # suppress -inf/-nan split-gain warnings
)

# ── 100+ Factor Pipeline additions ───────────────────────────────────────────
# Feature variants
USE_ORTHOGONALIZED_FEATURES = False   # True → load panel_monthly_orthogonalized.parquet
PCA_VARIANCE_THRESHOLD      = 0.80    # variance explained threshold for 1b_orthogonalize.py

# RMT covariance denoising (fixes beta=2.07 in 3_pca_rp_backtest.py)
USE_RMT_COV_DENOISING = True

# Macro factor columns — NOT cross-sectionally ranked (same value for all stocks per month)
MACRO_COLS = [
    "vix_level", "vix_change_21d", "yield_10y", "yield_spread_10y2y",
    "yield_change_21d", "dollar_index", "credit_proxy_change",
    "market_trend_spx", "market_vol_regime",
]

# Cross-sectional transformer (2c_cs_transformer.py)
TRANSFORMER_CS_PARAMS = dict(
    d_model=128, n_heads_s1=4, n_layers_s1=2,
    n_heads_s2=4, n_layers_s2=2,
    dropout=0.1, lr=5e-4, weight_decay=1e-4,
    epochs=100, patience=15, max_stocks=520,
)
