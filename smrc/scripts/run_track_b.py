"""Track B — simulated quantisation for all three classical models (incl. NB)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data
from src.baselines import build_search
from src.metrics import full_evaluation
from src.quantisation import (apply_quantised_weights, count_parameters,
                              theoretical_size_kb)
from src.utils import log_result, print_row, set_seed

QUANT_MODELS = ("naive_bayes", "logreg", "linear_svm")


def main(prefer: str = "auto"):
    set_seed(config.SEED)
    ds = data.load(prefer=prefer)
    Xtr, Xte, _ = ds.vectorise()
    for model_key in QUANT_MODELS:
        search = build_search(model_key, seed=config.SEED)
        search.fit(Xtr, ds.y_train)
        base = search.best_estimator_
        n_params = count_parameters(base)
        row = {"model": model_key, "track": "B", "variant": "quantise",
               "bits": 32, "scheme": "none", "n_params": n_params,
               "theoretical_size_kb": theoretical_size_kb(n_params, 32),
               "seed": config.SEED, **full_evaluation(base, Xte, ds.y_test)}
        log_result(row); print_row(row)
        for bits in config.QUANT_BITS:
            model = apply_quantised_weights(base, bits, config.QUANT_SCHEME)
            row = {"model": model_key, "track": "B", "variant": "quantise",
                   "bits": bits, "scheme": config.QUANT_SCHEME, "n_params": n_params,
                   "theoretical_size_kb": theoretical_size_kb(n_params, bits),
                   "seed": config.SEED, **full_evaluation(model, Xte, ds.y_test)}
            log_result(row); print_row(row)
    print(f"\nTrack B complete. Results -> {config.RESULTS_CSV}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "auto")
