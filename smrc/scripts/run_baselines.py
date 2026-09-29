"""Tune and evaluate the three baseline classifiers."""
import _common  # noqa: F401  (must come first)

from src import config, data
from src.baselines import fit_tuned
from src.metrics import evaluate
from src.utils import log_result, print_row, set_seed


def main():
    args = _common.parse_args(__doc__)
    set_seed(config.SEED)
    ds = data.load(args.data)
    X_train, X_test, _ = ds.vectorise()
    print(f"train {X_train.shape}, test {X_test.shape}")

    for key in ("naive_bayes", "logreg", "linear_svm"):
        model, params = fit_tuned(key, X_train, ds.y_train, config.SEED)
        row = {"seed": config.SEED, "model": key, "track": "baseline", "variant": "tuned",
               "best_params": str(params), "n_features": X_train.shape[1],
               **evaluate(model, X_test, ds.y_test)}
        log_result(row)
        print_row(row)

    print(f"\nLogged to {config.RESULTS_CSV}")


if __name__ == "__main__":
    main()
