"""Re-run ONLY the experiments affected by the two fixes.

Saves you the full ~98 minute pipeline. Re-runs:
  FIX 1 - the mutual_info feature-selection sweeps (previously computed with
          discrete_features=True on continuous TF-IDF, which was invalid).
  FIX 2 - the weight-pruning sweeps (previously reported no size saving because
          a zeroed dense array pickles to the same size).

Everything else (baselines, chi2/L1 sweeps, doc-freq, Track B) is unaffected by
the fixes and does not need re-running.

Usage from the repo root:
    python scripts/rerun_fixes.py            # both fixes
    python scripts/rerun_fixes.py mi         # only the mutual_info sweeps
    python scripts/rerun_fixes.py prune      # only the weight pruning
"""
from __future__ import annotations
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # repo root, for `src`
sys.path.insert(0, str(_HERE))          # scripts dir, for `run_track_a`

import pandas as pd

from src import config, data
from src.utils import set_seed
from run_track_a import run_feature_pruning, run_weight_pruning


def drop_stale_rows(variants_selectors):
    """Remove superseded rows so the CSV holds one clean set of results.

    variants_selectors: list of (variant, selector_or_None) to purge.
    The old file is backed up alongside as results_prefix.csv.
    """
    path = config.RESULTS_CSV
    if not path.exists():
        print("[rerun] no existing results.csv, nothing to purge")
        return
    df = pd.read_csv(path)
    backup = config.RESULTS_DIR / "results_before_fixes.csv"
    df.to_csv(backup, index=False)
    before = len(df)

    mask = pd.Series(False, index=df.index)
    for variant, selector in variants_selectors:
        m = df.get("variant") == variant
        if selector is not None and "selector" in df.columns:
            m = m & (df["selector"] == selector)
        mask = mask | m
    df = df[~mask]
    df.to_csv(path, index=False)
    print(f"[rerun] purged {before - len(df)} stale rows "
          f"(backup -> {backup.name})")


def main(which: str = "all", prefer: str = "auto"):
    set_seed(config.SEED)
    do_mi = which in ("all", "mi")
    do_prune = which in ("all", "prune")

    purge = []
    if do_mi:
        purge.append(("feature_prune", "mutual_info"))
    if do_prune:
        purge.append(("weight_prune", None))
    drop_stale_rows(purge)

    ds = data.load(prefer=prefer)
    Xtr, Xte, _ = ds.vectorise()

    if do_mi:
        print("\n===== FIX 1: re-running mutual_info sweeps =====")
        run_feature_pruning(ds, Xtr, Xte, selectors=["mutual_info"])

    if do_prune:
        print("\n===== FIX 2: re-running weight pruning =====")
        run_weight_pruning(ds, Xtr, Xte)

    print(f"\nRe-run complete -> {config.RESULTS_CSV}")
    print("Next: python -m src.analysis && python scripts/make_figures.py")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    prefer = sys.argv[2] if len(sys.argv) > 2 else "auto"
    main(which, prefer)
