"""
4g_post_retrain_report.py - Post-retrain one-shot report
Run this after dropping fresh Kaggle scores into data/scores_cs_transformer.parquet.

What it does:
  1. Re-runs 4f_agent_ensemble_backtest.py to regenerate the bt_ie_ensemble_*.csv
     files and figures/agent_ensemble.png using the new CS-T scores.
  2. Prints a side-by-side "before vs after" table so you can see how the
     retrain changed each strategy's alpha and IR.
  3. Flags any strategies that got worse - worth a second look before
     declaring success.

Prereqs:
  - data/scores_cs_transformer.parquet has been refreshed with the Kaggle output
  - Old bt_ie_ensemble_*.csv files still exist from the previous run (so we can
    diff). If not, the report will still run - it just won't show a diff.

Usage:
  python 4g_post_retrain_report.py
"""
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
STRATEGIES = [
    "cs_alone", "fund_alone", "macro_alone",
    "cs_plus_fund", "cs_plus_macro", "cs_plus_both",
]

# Archive the previous ensemble backtest files before overwriting
print("=" * 70)
print("POST-RETRAIN REPORT - CS-Transformer Ensemble")
print("=" * 70)

before = {}
archive_dir = DATA_DIR / "pre_retrain"
archive_dir.mkdir(exist_ok=True)
for s in STRATEGIES:
    old = DATA_DIR / f"bt_ie_ensemble_{s}.csv"
    if old.exists():
        archived = archive_dir / f"bt_ie_ensemble_{s}.csv"
        shutil.copy(old, archived)
        before[s] = pd.read_csv(archived, parse_dates=["date"]).set_index("date")

if before:
    print(f"  Archived {len(before)} previous backtests → {archive_dir}/\n")
else:
    print("  No previous backtests found - report will show 'after' only.\n")


# Re-run 4f with the new scores
print("Running 4f_agent_ensemble_backtest.py ...\n")
result = subprocess.run(
    ["python", "4f_agent_ensemble_backtest.py"],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print("[ERROR] 4f failed:")
    print(result.stderr[-2000:])
    raise SystemExit(1)
# Print only the RESULTS section from 4f's output
for line in result.stdout.splitlines():
    if any(tag in line for tag in ("RESULTS", "Strategy", "alpha", "Ensemble",
                                    "cs_alone", "fund_alone", "macro_alone",
                                    "cs_plus", "SPX", "=" * 10, "-" * 10)):
        print(line)


# Load new backtest results
after = {}
for s in STRATEGIES:
    new = DATA_DIR / f"bt_ie_ensemble_{s}.csv"
    if new.exists():
        after[s] = pd.read_csv(new, parse_dates=["date"]).set_index("date")


def stats(bt, months=None):
    r = bt["active_ret"]
    if months is not None:
        r = r.head(months)
    n = len(r)
    if n < 6:
        return {"months": n}
    ann_alpha = (1 + r).prod() ** (12 / n) - 1
    te = r.std(ddof=1) * np.sqrt(12)
    ir = ann_alpha / te if te > 0 else float("nan")
    hit = (r > 0).mean()
    return {"months": n, "alpha": ann_alpha, "ir": ir, "hit": hit}


# Side-by-side diff
if before:
    print("\n" + "=" * 78)
    print("BEFORE vs AFTER - retrain impact per strategy")
    print("=" * 78)
    hdr = (f"{'Strategy':<18}{'Alpha':>10}{'IR':>8}{'Hit':>8}"
           f"{'Δ Alpha':>10}{'Δ IR':>8}{'Δ Hit':>8}")
    print(hdr)
    print("-" * len(hdr))
    regressions = []
    for s in STRATEGIES:
        if s not in before or s not in after:
            continue
        b = stats(before[s])
        a = stats(after[s])
        if "alpha" not in b or "alpha" not in a:
            continue
        d_alpha = a["alpha"] - b["alpha"]
        d_ir = a["ir"] - b["ir"]
        d_hit = a["hit"] - b["hit"]
        flag = "  ⚠" if d_alpha < -0.005 else ""
        print(f"{s:<18}{a['alpha']*100:>9.2f}%{a['ir']:>8.2f}{a['hit']*100:>7.1f}%"
              f"{d_alpha*100:>+9.2f}%{d_ir:>+8.2f}{d_hit*100:>+7.1f}%{flag}")
        if d_alpha < -0.005:
            regressions.append(s)

    if regressions:
        print(f"\n  ⚠ Regressions (>0.5% alpha drop): {', '.join(regressions)}")
        print("    Investigate before pushing the new scores to the deck.")
    else:
        print("\n  No regressions. Safe to update the deck with the new numbers.")
else:
    print("\n(No 'before' data - skipped diff.)")


# Quick sanity: does the chart file exist?
chart = Path("figures/agent_ensemble.png")
if chart.exists():
    print(f"\n  Chart refreshed: {chart}")
else:
    print(f"\n  [WARN] {chart} not regenerated - check 4f output")

print("\nDone.")
