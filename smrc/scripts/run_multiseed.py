"""Repeat the headline configurations under several random seeds.

Only the configurations the report makes claims about are repeated. Results go
to results_multiseed.csv, so the main results file is never touched.
"""
import _common  # noqa: F401  (must come first)

import time

from src import config, data
from src.baselines import MODEL_FACTORIES, fit_tuned, get_weights
from src.metrics import evaluate
from src.pruning import achieved_sparsity, prune_features, prune_weights, with_weights
from src.quantisation import quantise_model
from src.utils import log_result, print_row, set_seed


def _log(row):
    log_result(row, config.MULTISEED_CSV)
    print_row(row)


def baselines(ds, X_train, X_test, seed):
    for key in ("naive_bayes", "logreg", "linear_svm"):
        model, params = fit_tuned(key, X_train, ds.y_train, seed)
        _log({"seed": seed, "model": key, "track": "baseline", "variant": "tuned",
              "best_params": str(params), **evaluate(model, X_test, ds.y_test)})


def features(ds, X_train, X_test, seed):
    for key in config.FEATURE_MODELS:
        for method in config.FEATURE_SELECTORS:
            for keep in config.MULTISEED_KEEP_FRACTIONS:
                model, X_test_small, n_kept = prune_features(
                    MODEL_FACTORIES[key], X_train, ds.y_train, X_test, method, keep, seed)
                _log({"seed": seed, "model": key, "track": "A", "variant": "feature_prune",
                      "selector": method, "keep_fraction": keep, "n_features": n_kept,
                      **evaluate(model, X_test_small, ds.y_test)})


def weights(ds, X_train, X_test, seed):
    for key in config.WEIGHT_MODELS:
        base, _ = fit_tuned(key, X_train, ds.y_train, seed)
        w = get_weights(base)
        for sparsity in config.MULTISEED_SPARSITIES:
            pruned = prune_weights(w, sparsity)
            _log({"seed": seed, "model": key, "track": "A", "variant": "weight_prune",
                  "weight_sparsity": sparsity, "achieved_sparsity": achieved_sparsity(pruned),
                  **evaluate(with_weights(base, pruned), X_test, ds.y_test)})


def quant(ds, X_train, X_test, seed):
    for key in config.QUANT_MODELS:
        base, _ = fit_tuned(key, X_train, ds.y_train, seed)
        for bits in config.MULTISEED_QUANT_BITS:
            _log({"seed": seed, "model": key, "track": "B", "variant": "quantise",
                  "bits": bits,
                  **evaluate(quantise_model(base, bits, config.QUANT_SCHEME), X_test, ds.y_test)})


STAGES = {"baselines": baselines, "features": features, "weights": weights, "quant": quant}


def main():
    args = _common.parse_args(__doc__, stages=list(STAGES))
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    print(f"Seeds {config.SEEDS}, stages {stages}")

    ds = data.load(args.data)
    X_train, X_test, _ = ds.vectorise()
    start = time.perf_counter()
    for seed in config.SEEDS:
        print(f"\nSeed {seed}")
        set_seed(seed)
        for name in stages:
            STAGES[name](ds, X_train, X_test, seed)

    minutes = (time.perf_counter() - start) / 60
    print(f"\nDone in {minutes:.1f} minutes. Logged to {config.MULTISEED_CSV}")


if __name__ == "__main__":
    main()
