"""Week 1 deliverable: three tuned baselines with logged reference metrics."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data
from src.baselines import build_search
from src.metrics import full_evaluation
from src.sparsity import analyse, pretty
from src.utils import log_result, print_row, set_seed


def main(prefer: str = "auto"):
    set_seed(config.SEED)
    print("Loading data...")
    ds = data.load(prefer=prefer)
    print("Vectorising (TF-IDF)...")
    Xtr, Xte, vec = ds.vectorise()
    print(f"  train={Xtr.shape}  test={Xte.shape}")
    print(pretty(analyse(Xtr)))
    for model_key in ("naive_bayes", "logreg", "linear_svm"):
        print(f"\n=== {model_key}: nested CV search ===")
        search = build_search(model_key, seed=config.SEED)
        search.fit(Xtr, ds.y_train)
        best = search.best_estimator_
        print(f"  best params: {search.best_params_}")
        row = {"model": model_key, "track": "baseline", "variant": "tuned",
               "best_params": str(search.best_params_),
               "n_features": Xtr.shape[1], "seed": config.SEED,
               **full_evaluation(best, Xte, ds.y_test)}
        log_result(row); print_row(row)
    print(f"\nBaselines complete. Results -> {config.RESULTS_CSV}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "auto")
