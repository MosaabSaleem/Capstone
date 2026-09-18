"""Track B — simulated quantisation for all three classical models."""
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
    for key in QUANT_MODELS:
        s = build_search(key, seed=config.SEED)
        s.fit(Xtr, ds.y_train)
        base = s.best_estimator_
        n = count_parameters(base)
        row = {"model": key, "track": "B", "variant": "quantise", "bits": 32,
               "scheme": "none", "n_params": n,
               "theoretical_size_kb": theoretical_size_kb(n, 32),
               "seed": config.SEED, **full_evaluation(base, Xte, ds.y_test)}
        log_result(row); print_row(row)
        for bits in config.QUANT_BITS:
            m = apply_quantised_weights(base, bits, config.QUANT_SCHEME)
            row = {"model": key, "track": "B", "variant": "quantise", "bits": bits,
                   "scheme": config.QUANT_SCHEME, "n_params": n,
                   "theoretical_size_kb": theoretical_size_kb(n, bits),
                   "seed": config.SEED, **full_evaluation(m, Xte, ds.y_test)}
            log_result(row); print_row(row)
    print(f"\nTrack B complete -> {config.RESULTS_CSV}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "auto")
