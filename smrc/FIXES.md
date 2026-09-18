# Fixes applied after the first full Kaggle run

## FIX 1 — mutual information was computed under a false assumption
**File:** `src/pruning.py` (`_mutual_info_scorer`), `src/config.py` (`MI_SUBSAMPLE`)

The first run emitted ~20,000 warnings:
`Clustering metrics expects discrete values but received continuous values`.

Cause: `mutual_info_classif(X, y, discrete_features=True)` was called on raw
continuous TF-IDF weights, so the scores were invalid and the sweep was slow.

The obvious repair (`discrete_features=False`) is impossible here because
sklearn raises `Sparse matrix 'X' can't have continuous features`, and
densifying 120k x 20k is not feasible.

**Repair:** score MI on **binarised term presence** (`X > 0`) with
`discrete_features=True`. The flag is then a true statement about the input, the
matrix stays sparse, it is fast, and it matches the classical text
feature-selection formulation of Yang and Pedersen (1997), which the report
already cites.

## FIX 2 — weight pruning reported no size saving
**File:** `src/metrics.py` (`deployable_size_kb`, `weight_density`),
`src/config.py` (`SPARSE_STORAGE_THRESHOLD`), `src/plots.py` (new figure)

Every `weight_prune` row logged 625.85 KB regardless of sparsity, because
zeroing entries of a **dense** numpy array does not shrink its pickle. Magnitude
pruning therefore looked useless on the efficiency index.

**Repair:** measure the size a real deployment would ship. When a weight array is
mostly zeros, serialise it as CSR and report the smaller of the dense and sparse
footprints. Verified at the real 20,000-feature scale:

| sparsity | dense | deployable | saving |
|----------|-------|------------|--------|
| 0.70 | 625.7 KB | 282.2 KB | 2.2x |
| 0.90 | 625.7 KB | 94.7 KB | 6.6x |
| 0.95 | 625.7 KB | 47.8 KB | 13.1x |
| 0.99 | 625.7 KB | 10.3 KB | 60.8x |

New logged columns: `model_size_dense_kb` (transparency) and `weight_density`.
`model_size_kb` is now the deployable figure and drives the efficiency index.
New figure: `trackA_weight_sparsity_size.png`.

## Re-running only what changed
```bash
python scripts/rerun_fixes.py          # both fixes
python scripts/rerun_fixes.py mi       # only the mutual_info sweeps
python scripts/rerun_fixes.py prune    # only weight pruning
python -m src.analysis
python scripts/make_figures.py
```
It backs up the old CSV to `results/results_before_fixes.csv` and purges the
superseded rows, so you do not re-run the full pipeline.
