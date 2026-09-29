"""Measure how sparse the TF-IDF matrices are and how much memory CSR saves."""
import _common  # noqa: F401  (must come first)

from src import config, data
from src.sparsity import analyse, describe
from src.utils import log_result, set_seed


def main():
    args = _common.parse_args(__doc__)
    set_seed(config.SEED)
    ds = data.load(args.data)
    X_train, X_test, _ = ds.vectorise()

    train = analyse(X_train)
    print("Training matrix\n" + describe(train))
    print("\nTest matrix\n" + describe(analyse(X_test)))

    log_result({"model": "-", "track": "sparsity", "variant": "tfidf_train", **train.as_row()})
    print(f"\nLogged to {config.RESULTS_CSV}")


if __name__ == "__main__":
    main()
