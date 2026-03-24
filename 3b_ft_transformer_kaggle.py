"""
2b_nn_backtest_kaggle.py  — GPU-ready standalone version
=========================================================
Identical logic to 2b_nn_backtest.py but with NO dependency on config.py.
All hyperparameters are defined as plain variables at the top — easy to tweak on Kaggle.

╔══════════════════════════════════════════════════════════════════════╗
║  KAGGLE SETUP (5 steps)                                              ║
║                                                                      ║
║  1. Create a Kaggle Dataset called "investsoc-ml-data" and upload:   ║
║       • panel_monthly_enriched.parquet  (~12 MB)                     ║
║       • scores_lgbm.parquet            (~300 KB)  [optional]         ║
║                                                                      ║
║  2. In your Kaggle notebook:                                         ║
║       Add Data → Your Datasets → investsoc-ml-data                   ║
║                                                                      ║
║  3. Enable GPU:                                                      ║
║       Settings (right panel) → Accelerator → GPU T4 x2               ║
║                                                                      ║
║  4. Paste this entire file into a code cell and run.                 ║
║     Or upload this file and run: !python 2b_nn_backtest_kaggle.py    ║
║                                                                      ║
║  5. When done, download from Output:                                 ║
║       scores_transformer.parquet                                     ║
║       bt_transformer.csv                                             ║
║       bt_transformer_ls.csv                                          ║
║     Copy these into your local data/ folder.                         ║
╚══════════════════════════════════════════════════════════════════════╝

Expected GPU time:  ~15–25 min on Kaggle T4
Expected CPU time:  ~45–90 min (not recommended for 80+ features)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time as _time; _t0 = _time.time()

# ── Paths (override via env vars for local testing) ───────────────────────────
DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "/kaggle/input/datasets/patrickridge/investsoc-ml-data"))
OUT_DIR  = Path(os.environ.get("ML_OUT_DIR",  "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_IN     = DATA_DIR / "panel_monthly_enriched.parquet"
LGBM_SCORES  = DATA_DIR / "scores_lgbm.parquet"   # optional — for comparison print
OUT_SCORES   = OUT_DIR  / "scores_transformer.parquet"
OUT_BT_LO    = OUT_DIR  / "bt_transformer.csv"
OUT_BT_LS    = OUT_DIR  / "bt_transformer_ls.csv"

# ── Date splits ───────────────────────────────────────────────────────────────
START_DATE  = "2010-01-01"
TRAIN_END   = "2020-12-31"
VALID_END   = "2022-12-31"

# ── Portfolio construction ────────────────────────────────────────────────────
TOP_N       = 100
LONG_FRAC   = 0.20
RETRAIN_EVERY = 12

# ── FT-Transformer hyperparameters (tweak freely on Kaggle) ──────────────────
D_MODEL      = 64
N_HEADS      = 4
N_LAYERS     = 3
DROPOUT      = 0.1
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 80
PATIENCE     = 10
BATCH_SIZE   = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Performance helpers ────────────────────────────────────────────────────────
def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 6:
        return dict(months=len(r), ann=float("nan"), vol=float("nan"),
                    sharpe=float("nan"), maxdd=float("nan"))
    ann    = (1 + r).prod() ** (12 / len(r)) - 1
    vol    = r.std(ddof=1) * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else float("nan")
    nav    = (1 + r).cumprod()
    maxdd  = (nav / nav.cummax() - 1).min()
    return dict(months=len(r), ann=ann, vol=vol, sharpe=sharpe, maxdd=maxdd)


def print_stats(label: str, r: pd.Series):
    s = perf_stats(r)
    print(f"  {label}")
    print(f"    months={s['months']} | ann={s['ann']*100:.2f}% | "
          f"vol={s['vol']*100:.2f}% | sharpe={s['sharpe']:.2f} | maxdd={s['maxdd']*100:.2f}%")


def long_only_ret(df_month: pd.DataFrame, top_n: int) -> float:
    sub = df_month.dropna(subset=["score", "fwd_ret_1m"])
    if len(sub) < top_n:
        return float("nan")
    return sub.nlargest(top_n, "score")["fwd_ret_1m"].mean()


def long_short_ret(df_month: pd.DataFrame, frac: float) -> float:
    sub = df_month.dropna(subset=["score", "fwd_ret_1m"])
    n = len(sub)
    if n < 20:
        return float("nan")
    k = max(1, int(np.floor(n * frac)))
    sub_s = sub.sort_values("score", ascending=False)
    return sub_s.head(k)["fwd_ret_1m"].mean() - sub_s.tail(k)["fwd_ret_1m"].mean()


# ── FT-Transformer model ───────────────────────────────────────────────────────
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int, n_heads: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens  = self.tokenizer(x)
        cls     = self.cls_token.expand(x.size(0), -1, -1)
        tokens  = torch.cat([cls, tokens], dim=1)
        out     = self.transformer(tokens)
        cls_out = out[:, 0, :]
        return self.head(cls_out).squeeze(-1)


# ── Training ───────────────────────────────────────────────────────────────────
def make_tensors(df: pd.DataFrame, feat_cols: list):
    X = torch.tensor(df[feat_cols].fillna(0).values, dtype=torch.float32)
    y = torch.tensor(df["fwd_ret_1m"].values, dtype=torch.float32)
    return X, y


def train_model(X_tr, y_tr, X_va, y_va, n_features) -> FTTransformer:
    model = FTTransformer(
        n_features=n_features,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT,
    ).to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(X_va.to(DEVICE)), y_va.to(DEVICE)
            ).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"    Early stop at epoch {epoch + 1} (best val MSE: {best_val_loss:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model: FTTransformer, X: torch.Tensor, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            preds.append(model(X[i:i + batch_size].to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print("Loading enriched panel...")

    panel = pd.read_parquet(PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= START_DATE].reset_index(drop=True)

    exclude   = {"date", "ticker", "fwd_ret_1m"}
    feat_cols = [c for c in panel.columns if c not in exclude]
    n_features = len(feat_cols)

    print(f"Features: {n_features} | Rows: {len(panel):,} | Tickers: {panel['ticker'].nunique()}")

    months      = sorted(panel["date"].unique())
    train_end   = pd.Timestamp(TRAIN_END)
    valid_end   = pd.Timestamp(VALID_END)

    train_months = [m for m in months if m <= train_end]
    valid_months = [m for m in months if train_end < m <= valid_end]
    test_months  = [m for m in months if m > valid_end]

    print(f"Train: {train_months[0].date()} → {train_months[-1].date()} ({len(train_months)} months)")
    print(f"Valid: {valid_months[0].date()} → {valid_months[-1].date()} ({len(valid_months)} months)")
    print(f"Test:  {test_months[0].date()} → {test_months[-1].date()} ({len(test_months)} months)")

    tr = panel[panel["date"].isin(train_months)].dropna(subset=feat_cols + ["fwd_ret_1m"])
    va = panel[panel["date"].isin(valid_months)].dropna(subset=feat_cols + ["fwd_ret_1m"])
    X_tr, y_tr = make_tensors(tr, feat_cols)
    X_va, y_va = make_tensors(va, feat_cols)

    print("\nFitting initial FT-Transformer...")
    model = train_model(X_tr, y_tr, X_va, y_va, n_features)

    all_scores = []
    for i, m in enumerate(test_months):
        if i > 0 and i % RETRAIN_EVERY == 0:
            seen_months = train_months + valid_months + test_months[:i]
            tr_exp = panel[panel["date"].isin(seen_months)].dropna(
                subset=feat_cols + ["fwd_ret_1m"]
            )
            va_new = panel[panel["date"].isin(
                test_months[max(0, i - RETRAIN_EVERY):i]
            )].dropna(subset=feat_cols + ["fwd_ret_1m"])

            if len(va_new) > 0:
                print(f"  Retraining at month {i} ({m.date()})...")
                X_tr_e, y_tr_e = make_tensors(tr_exp, feat_cols)
                X_va_e, y_va_e = make_tensors(va_new, feat_cols)
                model = train_model(X_tr_e, y_tr_e, X_va_e, y_va_e, n_features)

        sub = panel[panel["date"] == m].copy()
        if len(sub) == 0:
            continue
        X_sub = torch.tensor(sub[feat_cols].fillna(0).values, dtype=torch.float32)
        sub["score"] = predict(model, X_sub)
        all_scores.append(sub[["date", "ticker", "score", "fwd_ret_1m"]])

    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_parquet(OUT_SCORES, index=False)
    print(f"\nSaved scores: {OUT_SCORES} | rows={len(scores):,}")

    lo_rets = scores.groupby("date").apply(long_only_ret, top_n=TOP_N).rename("port_ret")
    lo_rets = lo_rets.dropna().reset_index()
    lo_rets.to_csv(OUT_BT_LO, index=False)

    ls_rets = scores.groupby("date").apply(long_short_ret, frac=LONG_FRAC).rename("ls_ret")
    ls_rets = ls_rets.dropna().reset_index()
    ls_rets.to_csv(OUT_BT_LS, index=False)

    print("\n" + "=" * 65)
    print("FT-TRANSFORMER  (test period results)")
    print("=" * 65)
    print_stats(f"Long-Only Top{TOP_N} (equal weight)", lo_rets["port_ret"])
    print_stats(f"Long-Short top/bot {int(LONG_FRAC*100)}%", ls_rets["ls_ret"])

    try:
        lgbm_scores = pd.read_parquet(LGBM_SCORES)
        lgbm_lo = lgbm_scores.groupby("date").apply(long_only_ret, top_n=TOP_N).dropna()
        lgbm_ls = lgbm_scores.groupby("date").apply(long_short_ret, frac=LONG_FRAC).dropna()
        lg_lo = perf_stats(lgbm_lo)
        lg_ls = perf_stats(lgbm_ls)
        tr_lo = perf_stats(lo_rets["port_ret"])
        tr_ls = perf_stats(ls_rets["ls_ret"])
        print("\n" + "=" * 65)
        print("COMPARISON: LightGBM  vs  FT-Transformer")
        print("=" * 65)
        print(f"{'Strategy':<30} {'Sharpe':>8}  {'Ann Ret':>8}  {'MaxDD':>8}")
        print("-" * 65)
        print(f"{'LGBM Long-Only':<30} {lg_lo['sharpe']:>8.2f}  {lg_lo['ann']*100:>7.1f}%  {lg_lo['maxdd']*100:>7.1f}%")
        print(f"{'Transformer Long-Only':<30} {tr_lo['sharpe']:>8.2f}  {tr_lo['ann']*100:>7.1f}%  {tr_lo['maxdd']*100:>7.1f}%")
        print("-" * 65)
        print(f"{'LGBM Long-Short':<30} {lg_ls['sharpe']:>8.2f}  {lg_ls['ann']*100:>7.1f}%  {lg_ls['maxdd']*100:>7.1f}%")
        print(f"{'Transformer Long-Short':<30} {tr_ls['sharpe']:>8.2f}  {tr_ls['ann']*100:>7.1f}%  {tr_ls['maxdd']*100:>7.1f}%")
        print("=" * 65)
    except FileNotFoundError:
        pass

    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
