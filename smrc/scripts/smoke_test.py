"""End-to-end smoke test on tiny synthetic data (no downloads)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data
from src.baselines import MODEL_FACTORIES, build_search, get_coef
from src.metrics import full_evaluation
from src.pruning import apply_pruned_weights, prune_features, prune_weights
from src.quantisation import apply_quantised_weights, count_parameters
from src.sparsity import analyse
from src.utils import set_seed


def main() -> int:
    set_seed(config.SEED)
    print("[1/7] synthetic data + TF-IDF...")
    ds = data.load(prefer="synthetic")
    Xtr, Xte, _ = ds.vectorise(max_features=500, ngram_range=(1, 1), min_df=1)
    assert Xtr.shape[0] > 0 and Xte.shape[0] > 0

    print("[2/7] sparsity report (pt1)...")
    rep = analyse(Xtr)
    assert rep.dense_bytes >= rep.sparse_bytes
    print(f"      sparsity={rep.sparsity:.2%}  saving={rep.savings_ratio}x")

    print("[3/7] baseline (logreg) with CV...")
    search = build_search("logreg", seed=config.SEED)
    search.fit(Xtr, ds.y_train)
    base = search.best_estimator_
    m = full_evaluation(base, Xte, ds.y_test)
    assert 0.0 <= m["accuracy"] <= 1.0 and "efficiency_index" in m
    print(f"      acc={m['accuracy']}  eff_index={m['efficiency_index']}")

    print("[4/7] Track A feature pruning (pt2)...")
    model, Xte_red, n_kept = prune_features(
        MODEL_FACTORIES["logreg"], Xtr, ds.y_train, Xte, "chi2", 0.5)
    assert n_kept <= Xtr.shape[1]
    full_evaluation(model, Xte_red, ds.y_test)

    print("[5/7] Track A weight pruning...")
    pruned = prune_weights(get_coef(base), 0.9)
    wm = full_evaluation(apply_pruned_weights(base, pruned), Xte, ds.y_test)
    print(f"      90% sparsity acc={wm['accuracy']}")

    print("[6/7] Track B quantisation — linear AND Naive Bayes (pt4)...")
    q_lin = full_evaluation(apply_quantised_weights(base, 8), Xte, ds.y_test)
    nb = build_search("naive_bayes", seed=config.SEED)
    nb.fit(Xtr, ds.y_train)
    nb_base = nb.best_estimator_
    assert count_parameters(nb_base) > 0
    q_nb = full_evaluation(apply_quantised_weights(nb_base, 8), Xte, ds.y_test)
    print(f"      int8 logreg acc={q_lin['accuracy']}  int8 NB acc={q_nb['accuracy']}")

    print("[7/7] metric completeness...")
    for key in ("accuracy", "macro_f1", "latency_ms", "peak_mem_mb",
                "rss_delta_mb", "model_size_kb", "efficiency_index"):
        assert key in m, f"missing metric {key}"

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
