"""Repeat the HEADLINE configurations across several random seeds.

Why this exists
---------------
The main sweep runs on a single seed, which means small differences between
methods cannot be distinguished from run-to-run noise. In particular the
chi-square versus mutual-information comparison turns on gaps of a few
thousandths of a point, which a single run cannot resolve.

This script re-runs only the configurations the report actually makes claims
about, once per seed in config.SEEDS, and logs every run to a separate file so
the original single-seed sweep stays intact. Use scripts/summarise_multiseed.py
afterwards to compute means and standard deviations.

Scope is controlled by the MULTISEED_* settings in src/config.py.

Usage from the repo root:
    python scripts/run_multiseed.py                 # everything, all seeds
    python scripts/run_multiseed.py baselines       # just the baselines
    python scripts/run_multiseed.py features        # just the selector sweep
    python scripts/run_multiseed.py weights         # just weight pruning
    python scripts/run_multiseed.py quant           # just quantisation
    python scripts/run_multiseed.py all synthetic   # smoke run on toy data
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src import config, data
from src.baselines import MODEL_FACTORIES, build_search, get_coef
from src.metrics import full_evaluation
from src.pruning import (achieved_sparsity, apply_pruned_weights,
                         prune_features, prune_weights)
from src.quantisation import (apply_quantised_weights, count_parameters,
                              theoretical_size_kb)
from src.utils import log_result, print_row, set_seed

OUT = config.MULTISEED_CSV


def _log(row):
    log_result(row, path=OUT)
    print_row(row)


def run_baselines(ds, Xtr, Xte, seed):
    print(f"\n--- baselines (seed {seed}) ---")
    for key in ("naive_bayes", "logreg", "linear_svm"):
        s = build_search(key, seed=seed)
        s.fit(Xtr, ds.y_train)
        _log({"seed": seed, "model": key, "track": "baseline", "variant": "tuned",
              "best_params": str(s.best_params_), "n_features": Xtr.shape[1],
              **full_evaluation(s.best_estimator_, Xte, ds.y_test)})


def run_features(ds, Xtr, Xte, seed):
    """The selector comparison. This is the main reason multi-seed exists."""
    print(f"\n--- feature selection (seed {seed}) ---")
    for key in config.MULTISEED_FEATURE_MODELS:
        for sel in config.MULTISEED_SELECTORS:
            for keep in config.MULTISEED_KEEP_FRACTIONS:
                model, Xte_red, n_kept = prune_features(
                    MODEL_FACTORIES[key], Xtr, ds.y_train, Xte, sel, keep, seed)
                _log({"seed": seed, "model": key, "track": "A",
                      "variant": "feature_prune", "selector": sel,
                      "keep_fraction": keep, "n_features": n_kept,
                      **full_evaluation(model, Xte_red, ds.y_test)})


def run_weights(ds, Xtr, Xte, seed):
    print(f"\n--- weight pruning (seed {seed}) ---")
    for key in config.MULTISEED_WEIGHT_MODELS:
        s = build_search(key, seed=seed)
        s.fit(Xtr, ds.y_train)
        base = s.best_estimator_
        coef = get_coef(base)
        for sp in config.MULTISEED_SPARSITIES:
            pc = prune_weights(coef, sp)
            _log({"seed": seed, "model": key, "track": "A",
                  "variant": "weight_prune", "weight_sparsity": sp,
                  "achieved_sparsity": achieved_sparsity(pc),
                  "n_features": Xtr.shape[1],
                  **full_evaluation(apply_pruned_weights(base, pc), Xte, ds.y_test)})


def run_quant(ds, Xtr, Xte, seed):
    print(f"\n--- quantisation (seed {seed}) ---")
    for key in config.MULTISEED_QUANT_MODELS:
        s = build_search(key, seed=seed)
        s.fit(Xtr, ds.y_train)
        base = s.best_estimator_
        n_params = count_parameters(base)
        for bits in config.MULTISEED_QUANT_BITS:
            m = apply_quantised_weights(base, bits, config.QUANT_SCHEME)
            _log({"seed": seed, "model": key, "track": "B", "variant": "quantise",
                  "bits": bits, "scheme": config.QUANT_SCHEME, "n_params": n_params,
                  "theoretical_size_kb": theoretical_size_kb(n_params, bits),
                  **full_evaluation(m, Xte, ds.y_test)})


STAGES = {"baselines": run_baselines, "features": run_features,
          "weights": run_weights, "quant": run_quant}


def main(which: str = "all", prefer: str = "auto"):
    stages = list(STAGES) if which == "all" else [which]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"unknown stage '{s}'. Options: all, {', '.join(STAGES)}")

    print(f"Seeds: {config.SEEDS}")
    print(f"Stages: {stages}")
    print(f"Logging to: {OUT}")

    ds = data.load(prefer=prefer)
    t0 = time.perf_counter()

    for seed in config.SEEDS:
        print(f"\n{'=' * 60}\nSEED {seed}\n{'=' * 60}")
        set_seed(seed)
        # Re-vectorise per seed. TF-IDF itself is deterministic, but doing it
        # inside the loop keeps each seed's run genuinely self-contained.
        Xtr, Xte, _ = ds.vectorise()
        for s in stages:
            STAGES[s](ds, Xtr, Xte, seed)

    mins = (time.perf_counter() - t0) / 60
    print(f"\nDone in {mins:.1f} minutes. Results -> {OUT}")
    print("Next: python scripts/summarise_multiseed.py")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    prefer = sys.argv[2] if len(sys.argv) > 2 else "auto"
    main(which, prefer)
