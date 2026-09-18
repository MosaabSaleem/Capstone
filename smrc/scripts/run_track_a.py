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


def run_feature_pruning(ds, Xtr, Xte, selectors=None):
    print("\n### 1) Statistical feature selection ###")
    for key in FEATURE_PRUNE_MODELS:
        for sel in (selectors or config.FEATURE_SELECTORS):
            for keep in config.FEATURE_KEEP_FRACTIONS:
                model, Xte_red, n_kept = prune_features(
                    MODEL_FACTORIES[key], Xtr, ds.y_train, Xte, sel, keep)
                row = {"model": key, "track": "A", "variant": "feature_prune",
                       "selector": sel, "keep_fraction": keep,
                       "n_features": n_kept, "seed": config.SEED,
                       **full_evaluation(model, Xte_red, ds.y_test)}
                log_result(row); print_row(row)


def run_docfreq_filtering(ds):
    print("\n### 2) Document-frequency filtering (min_df / max_df) ###")
    for (min_df, max_df) in config.DOCFREQ_FILTERS:
        Xtr, Xte, _ = ds.vectorise(min_df=min_df, max_df=max_df)
        for key in FEATURE_PRUNE_MODELS:
            s = build_search(key, seed=config.SEED)
            s.fit(Xtr, ds.y_train)
            row = {"model": key, "track": "A", "variant": "docfreq_filter",
                   "min_df": min_df, "max_df": max_df,
                   "n_features": Xtr.shape[1], "seed": config.SEED,
                   **full_evaluation(s.best_estimator_, Xte, ds.y_test)}
            log_result(row); print_row(row)


def run_weight_pruning(ds, Xtr, Xte):
    print("\n### 3) Weight pruning ###")
    for key in WEIGHT_PRUNE_MODELS:
        s = build_search(key, seed=config.SEED)
        s.fit(Xtr, ds.y_train)
        base = s.best_estimator_
        coef = get_coef(base)
        for sp in config.WEIGHT_PRUNE_SPARSITIES:
            pc = prune_weights(coef, sp)
            model = apply_pruned_weights(base, pc)
            row = {"model": key, "track": "A", "variant": "weight_prune",
                   "weight_sparsity": sp, "achieved_sparsity": achieved_sparsity(pc),
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
    print(f"\nTrack A complete -> {config.RESULTS_CSV}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "auto")
