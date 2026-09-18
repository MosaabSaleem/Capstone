"""Point 1 — quantify TF-IDF sparsity and CSR-vs-dense RAM savings."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, data
from src.sparsity import analyse, pretty
from src.utils import log_result, set_seed


def main(prefer: str = "auto"):
    set_seed(config.SEED)
    ds = data.load(prefer=prefer)
    Xtr, Xte, vec = ds.vectorise()
    print("TRAIN feature matrix:"); rtr = analyse(Xtr); print(pretty(rtr))
    print("\nTEST feature matrix:"); print(pretty(analyse(Xte)))
    log_result({"model": "-", "track": "sparsity", "variant": "tfidf_train",
                "n_features": Xtr.shape[1], **rtr.as_row()})
    print(f"\nLogged -> {config.RESULTS_CSV}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "auto")
