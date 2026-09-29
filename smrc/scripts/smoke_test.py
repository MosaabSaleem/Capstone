"""Check that every stage of the pipeline runs, using a tiny synthetic dataset.

No download is needed and nothing is written to the results folders. If this
passes, the real runs should work too.
"""
import _common  # noqa: F401  (must come first)

import warnings

from src import config, data
from src.baselines import MODEL_FACTORIES, fit_tuned, get_weights
from src.metrics import dense_size_kb, deployable_size_kb, evaluate
from src.pruning import prune_features, prune_weights, with_weights
from src.quantisation import count_parameters, quantise_model
from src.sparsity import analyse
from src.utils import set_seed

EXPECTED_METRICS = ["accuracy", "macro_f1", "latency_ms", "peak_mem_mb", "rss_delta_mb",
                    "model_size_dense_kb", "model_size_kb", "weight_density",
                    "efficiency_index"]


def check(step: str, n: int, total: int = 8) -> None:
    print(f"[{n}/{total}] {step}")


def main() -> int:
    set_seed(config.SEED)

    check("synthetic data and TF-IDF", 1)
    ds = data.load("synthetic")
    X_train, X_test, _ = ds.vectorise(max_features=500, ngram_range=(1, 1), min_df=1)

    check("sparsity measurement", 2)
    report = analyse(X_train)
    assert report.dense_mb >= report.sparse_mb

    check("tuned baseline and every metric", 3)
    base, _ = fit_tuned("logreg", X_train, ds.y_train)
    metrics = evaluate(base, X_test, ds.y_test)
    missing = [m for m in EXPECTED_METRICS if m not in metrics]
    assert not missing, f"missing metrics: {missing}"

    check("feature selection with all three selectors", 4)
    for method in config.FEATURE_SELECTORS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, n_kept = prune_features(MODEL_FACTORIES["logreg"], X_train, ds.y_train,
                                          X_test, method, 0.5)
        noisy = [w for w in caught if "discrete" in str(w.message).lower()]
        assert not noisy, f"{method} raised discrete-value warnings"
        assert n_kept <= X_train.shape[1]

    check("weight pruning shrinks the deployable size", 5)
    pruned = with_weights(base, prune_weights(get_weights(base), 0.9))
    assert deployable_size_kb(pruned) < dense_size_kb(pruned)

    check("quantisation of a linear model", 6)
    evaluate(quantise_model(base, 8), X_test, ds.y_test)

    check("quantisation of Naive Bayes", 7)
    nb, _ = fit_tuned("naive_bayes", X_train, ds.y_train)
    assert count_parameters(nb) > 0
    evaluate(quantise_model(nb, 4), X_test, ds.y_test)

    check("multi-seed scripts import", 8)
    import importlib.util
    for name in ("run_multiseed", "summarise_multiseed"):
        path = _common.ROOT / "scripts" / f"{name}.py"
        assert importlib.util.spec_from_file_location(name, path) is not None

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
