"""
3e_hp_sweep.py — Hyperparameter sweep for the CS-Transformer
=============================================================
Runs the full training pipeline across N config variants and records the
test-period performance of each. Result: a single CSV you can use to
pick the best config before doing a proper Kaggle retrain.

Variants tested (edit SWEEP below to add/remove):
  1. baseline       — current config
  2. deeper_3x3     — 3 layers in each stage (more capacity)
  3. wider_192      — d_model=192 (more capacity, less depth)
  4. dropout_20     — 0.2 dropout (more regularisation)
  5. ic_loss        — IC loss instead of MSE
  6. ensemble_3     — 3-seed ensemble (averaged predictions)

How to run
----------
Locally (CPU, slow — each variant ~30-90 min):
    python 3e_hp_sweep.py

On Kaggle (GPU, fast — each variant ~15-25 min):
    Paste the whole file into a code cell. It'll need access to 3c_cs_transformer.py
    which isn't directly possible on Kaggle. See the KAGGLE note below.

Output: data/sweep_results.csv with one row per variant (IR, alpha, Sharpe, etc).
Variant scores saved to data/scores_cs_transformer_<name>.parquet for audit.

⚠ Kaggle note:
  This local orchestrator imports from 3c_cs_transformer.py. To run the full
  sweep on Kaggle GPU, you'd need to either:
    (a) Upload 3c_cs_transformer.py + config.py into the Kaggle dataset and
        import them in the notebook, OR
    (b) Copy-paste the entire 3c file above this sweep code and call the
        functions directly (simpler for a one-off sweep).

  For now, this script is primarily a LOCAL diagnostic tool. Expect ~8-15
  hours total for all six variants on CPU.
"""

import copy
import importlib.util
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Import 3c's internals (filename starts with digit → importlib) ───────────
_spec = importlib.util.spec_from_file_location(
    "cs_mod", Path(__file__).parent / "3c_cs_transformer.py"
)
cs_mod = importlib.util.module_from_spec(_spec)
sys.modules["cs_mod"] = cs_mod
_spec.loader.exec_module(cs_mod)

DEVICE = cs_mod.DEVICE
DATA_DIR = cs_mod.DATA_DIR

# Data loading helpers
build_monthly_cross_sections = cs_mod.build_monthly_cross_sections
train_cs_model               = cs_mod.train_cs_model
grpo_finetune_cs_model       = cs_mod.grpo_finetune_cs_model
train_ensemble               = cs_mod.train_ensemble
predict_cs_ensemble          = cs_mod.predict_cs_ensemble
long_only_ret                = cs_mod.long_only_ret
long_short_ret               = cs_mod.long_short_ret
perf_stats                   = cs_mod.perf_stats

# Config (mutable — we'll override per variant)
TRANSFORMER_CS_PARAMS = cs_mod.TRANSFORMER_CS_PARAMS
RL_FINETUNE_PARAMS    = cs_mod.RL_FINETUNE_PARAMS

# ── Sweep definition ─────────────────────────────────────────────────────────
# Each entry: name + dict of keys to override in TRANSFORMER_CS_PARAMS.
# To disable RL for a specific variant, set "rl_method" to None in rl_overrides.
SWEEP = [
    {"name": "baseline",     "overrides": {}},
    {"name": "deeper_3x3",   "overrides": {"n_layers_s1": 3, "n_layers_s2": 3}},
    {"name": "wider_192",    "overrides": {"d_model": 192}},
    {"name": "dropout_20",   "overrides": {"dropout": 0.2}},
    {"name": "ic_loss",      "overrides": {"loss_type": "ic"}},
    {"name": "ensemble_3",   "overrides": {"ensemble_seeds": 3}},
]

RESULTS_CSV = DATA_DIR / "sweep_results.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_variant(variant_name, overrides, panel, stock_feat_cols, macro_cols,
                    train_months, valid_months, test_months, cs_map):
    """Train + evaluate one config variant. Returns dict of metrics."""
    print("\n" + "═" * 74)
    print(f"VARIANT: {variant_name}   overrides: {overrides}")
    print("═" * 74)

    # Snapshot then override config
    original_params = dict(TRANSFORMER_CS_PARAMS)
    TRANSFORMER_CS_PARAMS.update(overrides)

    p = TRANSFORMER_CS_PARAMS
    n_macro = len(macro_cols)
    n_stock_features = len(stock_feat_cols)

    train_cs = [cs_map[m] for m in train_months if m in cs_map]
    valid_cs = [cs_map[m] for m in valid_months if m in cs_map]

    t0 = _time.time()
    try:
        rl_method = RL_FINETUNE_PARAMS["method"]
        n_seeds = p.get("ensemble_seeds", 1)
        models = train_ensemble(
            train_cs, valid_cs, n_stock_features,
            n_macro=n_macro, corr_matrix=None,
            n_seeds=n_seeds, rl_method=rl_method,
        )

        # Walk-forward predictions (single retrain at the default cadence)
        all_scores = []
        for i, m in enumerate(test_months):
            if m not in cs_map:
                continue
            cs = cs_map[m]
            scores = predict_cs_ensemble(models, cs, n_macro=n_macro)
            sub = (panel[panel["date"] == m]
                   .dropna(subset=["fwd_ret_1m"])
                   .head(cs["n_valid"])
                   .copy())
            sub["score"] = scores
            all_scores.append(sub[["date", "ticker", "score", "fwd_ret_1m"]])

        scores_df = pd.concat(all_scores, ignore_index=True)
        scores_df.to_parquet(
            DATA_DIR / f"scores_cs_transformer_{variant_name}.parquet",
            index=False,
        )

        # Compute IR / alpha / Sharpe / max DD for long-only top-100
        lo_rets = (scores_df.groupby("date")
                   .apply(long_only_ret, top_n=100)
                   .rename("port_ret").dropna().reset_index())
        lo_stats = perf_stats(lo_rets["port_ret"])

        ls_rets = (scores_df.groupby("date")
                   .apply(long_short_ret, frac=0.20)
                   .rename("ls_ret").dropna().reset_index())
        ls_stats = perf_stats(ls_rets["ls_ret"])

        runtime = _time.time() - t0
        result = {
            "variant": variant_name,
            "overrides": str(overrides),
            "lo_ann": lo_stats["ann"],
            "lo_sharpe": lo_stats["sharpe"],
            "lo_maxdd": lo_stats["maxdd"],
            "ls_ann": ls_stats["ann"],
            "ls_sharpe": ls_stats["sharpe"],
            "ls_maxdd": ls_stats["maxdd"],
            "runtime_min": runtime / 60,
            "status": "ok",
        }
        print(f"\n  [✓] {variant_name} done in {runtime/60:.1f} min | "
              f"LO Sharpe={lo_stats['sharpe']:.2f} ann={lo_stats['ann']*100:+.1f}%")

    except Exception as e:
        runtime = _time.time() - t0
        result = {
            "variant": variant_name,
            "overrides": str(overrides),
            "runtime_min": runtime / 60,
            "status": f"error: {type(e).__name__}: {e}",
        }
        print(f"\n  [✗] {variant_name} failed after {runtime/60:.1f} min — {e}")

    # Restore config
    TRANSFORMER_CS_PARAMS.clear()
    TRANSFORMER_CS_PARAMS.update(original_params)

    return result


def main():
    print("=" * 74)
    print("HYPERPARAMETER SWEEP — CS-Transformer")
    print(f"Device: {DEVICE}")
    print(f"Variants: {len(SWEEP)}")
    print("=" * 74)

    # ── Load panel once ───────────────────────────────────────────────────
    print("\nLoading panel...")
    panel = pd.read_parquet(cs_mod.PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= cs_mod.START_DATE].reset_index(drop=True)

    # Zombie filter
    spx_weights_path = DATA_DIR / "spx_weights.parquet"
    if spx_weights_path.exists():
        spx_w = pd.read_parquet(spx_weights_path)
        spx_w["date"] = pd.to_datetime(spx_w["date"]) + pd.offsets.MonthEnd(0)
        panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)
        valid_keys = set(zip(spx_w["date"], spx_w["ticker"]))
        mask = list(zip(panel["date"], panel["ticker"]))
        panel = panel[[k in valid_keys for k in mask]].reset_index(drop=True)

    exclude = {"date", "ticker", "fwd_ret_1m"}
    all_feat_cols = [c for c in panel.columns if c not in exclude]
    macro_cols = [c for c in cs_mod.MACRO_COLS if c in panel.columns]
    stock_feat_cols = [c for c in all_feat_cols if c not in macro_cols]

    months = sorted(panel["date"].unique())
    train_end = pd.Timestamp(cs_mod.TRAIN_END)
    valid_end = pd.Timestamp(cs_mod.VALID_END)
    train_months = [m for m in months if m <= train_end]
    valid_months = [m for m in months if train_end < m <= valid_end]
    test_months  = [m for m in months if m > valid_end]

    print(f"  Panel: {len(panel):,} rows · {len(stock_feat_cols)} stock features "
          f"· {len(macro_cols)} macro")
    print(f"  Train={len(train_months)} Valid={len(valid_months)} "
          f"Test={len(test_months)}")

    # Build cross-sections once and share across variants
    print("\nBuilding monthly cross-sections...")
    p = TRANSFORMER_CS_PARAMS
    all_cs = build_monthly_cross_sections(
        panel, stock_feat_cols, max_stocks=p["max_stocks"], macro_cols=macro_cols,
    )
    cs_map = {cs["date"]: cs for cs in all_cs}

    # ── Run sweep ─────────────────────────────────────────────────────────
    results = []
    t_start = _time.time()
    for i, variant in enumerate(SWEEP, 1):
        print(f"\n[{i}/{len(SWEEP)}] ──────────────────────────────────────────")
        r = run_one_variant(
            variant["name"], variant["overrides"],
            panel, stock_feat_cols, macro_cols,
            train_months, valid_months, test_months, cs_map,
        )
        results.append(r)

        # Save after each variant (so partial results survive crashes)
        pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
        print(f"  → saved {RESULTS_CSV} ({len(results)} variants done)")

    total_min = (_time.time() - t_start) / 60

    # ── Summary table ─────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    if not ok.empty:
        ok = ok.sort_values("lo_sharpe", ascending=False)

    print("\n" + "=" * 74)
    print(f"SWEEP COMPLETE — {total_min:.1f} min total ({len(results)} variants)")
    print("=" * 74)
    if not ok.empty:
        print(f"{'Variant':<16}{'LO Ann':>9}{'LO Sharpe':>10}"
              f"{'LO MaxDD':>10}{'LS Sharpe':>10}{'Runtime':>10}")
        print("-" * 74)
        for _, r in ok.iterrows():
            print(f"{r['variant']:<16}"
                  f"{r['lo_ann']*100:>8.2f}%"
                  f"{r['lo_sharpe']:>10.2f}"
                  f"{r['lo_maxdd']*100:>9.2f}%"
                  f"{r['ls_sharpe']:>10.2f}"
                  f"{r['runtime_min']:>9.1f}m")
    print(f"\nFull table: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
