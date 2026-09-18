"""Three tuned baselines with logged reference metrics."""
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
    ds = data.load(prefer=prefer)
    Xtr, Xte, vec = ds.vectorise()
    print(f"  train={Xtr.shape}  test={Xte.shape}")
    print(pretty(analyse(Xtr)))
    for key in ("naive_bayes", "logreg", "linear_svm"):
        print(f"\n=== {key}: nested CV search ===")
        s = build_search(key, seed=config.SEED)
        s.fit(Xtr, ds.y_train)
        print(f"  best params: {s.best_params_}")
        row = {"model": key, "track": "baseline", "variant": "tuned",
               "best_params": str(s.best_params_), "n_features": Xtr.shape[1],
               "seed": config.SEED,
               **full_evaluation(s.best_estimator_, Xte, ds.y_test)}
        log_result(row); print_row(row)
    print(f"\nBaselines complete -> {config.RESULTS_CSV}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "auto")
