"""Track A — feature selection + doc-frequency filtering + weight pruning."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data
from src.baselines import MODEL_FACTORIES, build_search, get_coef
from src.metrics import full_evaluation
from src.pruning import (achieved_sparsity, apply_pruned_weights,
                         prune_features, prune_weights)
from src.utils import log_result, print_row, set_seed

WEIGHT_PRUNE_MODELS = ("logreg", "linear_svm")
FEATURE_PRUNE_MODELS = ("naive_bayes", "logreg")


def run_feature_pruning(ds, Xtr, Xte):
    print("\n### 1) Statistical feature selection ###")
    for model_key in FEATURE_PRUNE_MODELS:
        factory = MODEL_FACTORIES[model_key]
        for selector in config.FEATURE_SELECTORS:
            for keep in config.FEATURE_KEEP_FRACTIONS:
                model, Xte_red, n_kept = prune_features(
                    factory, Xtr, ds.y_train, Xte, selector, keep)
                row = {"model": model_key, "track": "A", "variant": "feature_prune",
                       "selector": selector, "keep_fraction": keep,
                       "n_features": n_kept, "seed": config.SEED,
                       **full_evaluation(model, Xte_red, ds.y_test)}
                log_result(row); print_row(row)


def run_docfreq_filtering(ds):
    print("\n### 2) Document-frequency filtering (min_df / max_df) ###")
    for (min_df, max_df) in config.DOCFREQ_FILTERS:
        Xtr, Xte, _ = ds.vectorise(min_df=min_df, max_df=max_df)
        for model_key in FEATURE_PRUNE_MODELS:
            search = build_search(model_key, seed=config.SEED)
            search.fit(Xtr, ds.y_train)
            best = search.best_estimator_
            row = {"model": model_key, "track": "A", "variant": "docfreq_filter",
                   "min_df": min_df, "max_df": max_df,
                   "n_features": Xtr.shape[1], "seed": config.SEED,
                   **full_evaluation(best, Xte, ds.y_test)}
            log_result(row); print_row(row)


def run_weight_pruning(ds, Xtr, Xte):
    print("\n### 3) Weight pruning ###")
    for model_key in WEIGHT_PRUNE_MODELS:
        search = build_search(model_key, seed=config.SEED)
        search.fit(Xtr, ds.y_train)
        base = search.best_estimator_
        coef = get_coef(base)
        for sparsity in config.WEIGHT_PRUNE_SPARSITIES:
            pruned_coef = prune_weights(coef, sparsity)
            model = apply_pruned_weights(base, pruned_coef)
            row = {"model": model_key, "track": "A", "variant": "weight_prune",
                   "weight_sparsity": sparsity,
                   "achieved_sparsity": achieved_sparsity(pruned_coef),
                   "n_features": Xtr.shape[1], "seed": config.SEED,
                   **full_evaluation(model, Xte, ds.y_test)}
            log_result(row); print_row(row)


def main(prefer: str = "auto"):
    set_seed(config.SEED)
    ds = data.load(prefer=prefer)
    Xtr, Xte, _ = ds.vectorise()
    run_feature_pruning(ds, Xtr, Xte)
    run_docfreq_filtering(ds)
    run_weight_pruning(ds, Xtr, Xte)
    print(f"\nTrack A complete. Results -> {config.RESULTS_CSV}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "auto")
