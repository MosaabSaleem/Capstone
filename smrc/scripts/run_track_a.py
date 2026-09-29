"""Track A: feature selection, document-frequency filtering and weight pruning."""
import _common  # noqa: F401  (must come first)

from src import config, data
from src.baselines import MODEL_FACTORIES, fit_tuned, get_weights
from src.metrics import evaluate
from src.pruning import achieved_sparsity, prune_features, prune_weights, with_weights
from src.utils import log_result, print_row, set_seed


def feature_selection(ds, X_train, X_test):
    print("\nFeature selection")
    for key in config.FEATURE_MODELS:
        for method in config.FEATURE_SELECTORS:
            for keep in config.FEATURE_KEEP_FRACTIONS:
                model, X_test_small, n_kept = prune_features(
                    MODEL_FACTORIES[key], X_train, ds.y_train, X_test, method, keep)
                row = {"seed": config.SEED, "model": key, "track": "A",
                       "variant": "feature_prune", "selector": method,
                       "keep_fraction": keep, "n_features": n_kept,
                       **evaluate(model, X_test_small, ds.y_test)}
                log_result(row)
                print_row(row)


def document_frequency(ds):
    print("\nDocument-frequency filtering")
    for min_df, max_df in config.DOCFREQ_FILTERS:
        X_train, X_test, _ = ds.vectorise(min_df=min_df, max_df=max_df)
        for key in config.FEATURE_MODELS:
            model, _ = fit_tuned(key, X_train, ds.y_train, config.SEED)
            row = {"seed": config.SEED, "model": key, "track": "A",
                   "variant": "docfreq_filter", "min_df": min_df, "max_df": max_df,
                   "n_features": X_train.shape[1], **evaluate(model, X_test, ds.y_test)}
            log_result(row)
            print_row(row)


def weight_pruning(ds, X_train, X_test):
    print("\nWeight pruning")
    for key in config.WEIGHT_MODELS:
        base, _ = fit_tuned(key, X_train, ds.y_train, config.SEED)
        weights = get_weights(base)
        for sparsity in config.WEIGHT_PRUNE_SPARSITIES:
            pruned = prune_weights(weights, sparsity)
            row = {"seed": config.SEED, "model": key, "track": "A",
                   "variant": "weight_prune", "weight_sparsity": sparsity,
                   "achieved_sparsity": achieved_sparsity(pruned),
                   **evaluate(with_weights(base, pruned), X_test, ds.y_test)}
            log_result(row)
            print_row(row)


def main():
    args = _common.parse_args(__doc__)
    set_seed(config.SEED)
    ds = data.load(args.data)
    X_train, X_test, _ = ds.vectorise()
    feature_selection(ds, X_train, X_test)
    document_frequency(ds)
    weight_pruning(ds, X_train, X_test)
    print(f"\nLogged to {config.RESULTS_CSV}")


if __name__ == "__main__":
    main()
