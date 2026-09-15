# Running the CS-Transformer on Kaggle

The local `3c_cs_transformer.py` is CPU-bound and slow. `3d_cs_transformer_kaggle.py`
is the same model with config inlined so it runs standalone on a Kaggle GPU.

There are two distinct runs. Job A is a straight retrain. Job B extends the
scoring window so the RL overlay in `5c` can use transformer scores instead of
factor-combo scores. Job B needs more GPU time and is only worth doing if you
want those two halves of the project measured on the same signal.

## Data setup (once)

Both jobs read from the same Kaggle Dataset.

1. On kaggle.com, go to **Datasets → New Dataset**, name it `investsoc-ml-data`.
2. Upload from your local `data/`:

   | File | Size | Why |
   |---|---|---|
   | `panel_monthly_enriched.parquet` | ~12 MB | features and forward returns |
   | `spx_weights.parquet` | ~3 MB | index membership; the zombie filter needs it |

   Upload both. Without `spx_weights.parquet` the script prints a warning and
   trains on delisted tickers, which is the bug that invalidated the previous
   results.

3. Check where Kaggle actually mounted it before running anything. It nests
   uploads under your username, so the path is usually

       /kaggle/input/datasets/<your-username>/investsoc-ml-data

   not `/kaggle/input/investsoc-ml-data`. Confirm with:

   ```python
   import os
   for root, dirs, files in os.walk("/kaggle/input"):
       for f in files:
           print(os.path.join(root, f))
   ```

   Set `ML_DATA_DIR` to whatever directory that prints. Getting this wrong
   gives a `FileNotFoundError` on `pd.read_parquet(PANEL_IN)`.

Re-upload `panel_monthly_enriched.parquet` whenever you rebuild it with
`1h_feature_engineering.py`, or the GPU run will train on a stale panel.

## Job A: retrain on current code (~90 min)

Fixes the staleness problem. The scores currently in `data/` predate the Macro
FiLM upgrade, so they test a model that no longer matches the code.

1. **New Notebook**, then **Add Data → Your Datasets → investsoc-ml-data**
2. **Settings → Accelerator → GPU T4 x2**
3. Cell 1:

```python
import os
os.environ["ML_DATA_DIR"] = "/kaggle/input/datasets/patrickridge/investsoc-ml-data"
os.environ["ML_OUT_DIR"]  = "/kaggle/working"
```

4. Cell 2: paste the entire contents of `3d_cs_transformer_kaggle.py`
5. Run all. Check the log shows the filter working:

```
Zombie filter: 147,733 → ~100,000 rows (... non-member pairs dropped, weight floor 5e-06)
```

   If it says `[WARN] spx_weights.parquet not found`, stop and fix the upload.

6. Download from **Output** into your local `data/`:
   - `scores_cs_transformer.parquet`
   - `scores_cs_transformer_model.pt`
   - `bt_cs_transformer.csv`
   - `bt_cs_transformer_ls.csv`

   The checkpoint matters. Without it a score file has nothing behind it that
   says which model produced it, which is exactly how `scores_lgbm.parquet`
   ended up as an unreproducible benchmark for six months.

7. Locally: `python 4b_index_enhancement.py`

This gives 17 OOS months, same as before, but of a model that matches the code
and was trained on clean data.

## Job B: full-history scores (several hours)

Only needed to put the RL overlay on transformer scores. `5c` walks forward over
~143 months and cannot use a signal that only exists for the last 17.

Same as Job A, but Cell 1 becomes:

```python
import os
os.environ["ML_DATA_DIR"]  = "/kaggle/input/investsoc-ml-data"
os.environ["ML_OUT_DIR"]   = "/kaggle/working"
os.environ["ML_OOS_START"] = "2013-12-31"   # score 2014 onward
```

The model fits on 2010-2013, scores 2014-2015, refits on everything through
2015, scores 2016-2017, and so on. That is five-plus full training cycles rather
than one, hence the runtime.

Kaggle caps sessions at 12 hours. If it looks like overrunning, raise
`ML_OOS_START` to `2017-12-31` for roughly 96 months of coverage, which is still
enough for a meaningful walk-forward.

Afterwards, locally:

```bash
ML_OOS_START=2013-12-31 python 3f_lgbm_baseline.py   # tree baseline on the same months
python 4b_index_enhancement.py                       # index enhancement on the new scores
python 4h_bootstrap_ci.py                            # confidence intervals on both
python 5c_walk_forward.py                            # RL overlay, now on transformer scores
```

Run 3f with the same `ML_OOS_START` as the Kaggle job. It is CPU-only and takes
a few minutes. Skipping it leaves the tree baseline on a different window from
the transformer, which is the mismatch that let the retracted +0.56 stand for
six months. Matching windows is the point of the whole exercise.

## Checks worth doing on the output

Before trusting anything the run produces:

```python
import pandas as pd
s = pd.read_parquet("data/scores_cs_transformer.parquet")
s["date"] = pd.to_datetime(s["date"])

print(s["date"].min(), "->", s["date"].max(), s["date"].nunique(), "months")
print("max |fwd_ret_1m|:", s["fwd_ret_1m"].abs().max())
print("CPWR rows:", (s["ticker"] == "CPWR").sum())
```

Expect the max absolute monthly return around 1.0 or below, and zero CPWR rows.
Anything above 2.0 means the filter did not apply and the run should be redone.
See `Notes/Improvements.md` item 22 for why this check exists.
