"""End-to-end smoke test on tiny synthetic data (no downloads)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src import config, data
from src.baselines import MODEL_FACTORIES, build_search, get_coef
from src.metrics import full_evaluation, deployable_size_kb, model_size_kb
from src.pruning import apply_pruned_weights, prune_features, prune_weights
from src.quantisation import apply_quantised_weights, count_parameters
from src.sparsity import analyse
from src.utils import set_seed


def main() -> int:
    set_seed(config.SEED)
    print("[1/8] synthetic data + TF-IDF...")
    ds = data.load(prefer="synthetic")
    Xtr, Xte, _ = ds.vectorise(max_features=500, ngram_range=(1, 1), min_df=1)

    print("[2/8] sparsity report (pt1)...")
    rep = analyse(Xtr)
    assert rep.dense_bytes >= rep.sparse_bytes
    print(f"      sparsity={rep.sparsity:.2%} saving={rep.savings_ratio}x")

    print("[3/8] baseline (logreg) with CV...")
    s = build_search("logreg", seed=config.SEED); s.fit(Xtr, ds.y_train)
    base = s.best_estimator_
    m = full_evaluation(base, Xte, ds.y_test)
    assert "efficiency_index" in m and "model_size_dense_kb" in m
    print(f"      acc={m['accuracy']} size={m['model_size_kb']}KB")

    print("[4/8] Track A feature pruning...")
    mdl, Xte_red, n_kept = prune_features(
        MODEL_FACTORIES["logreg"], Xtr, ds.y_train, Xte, "chi2", 0.5)
    assert n_kept <= Xtr.shape[1]
    full_evaluation(mdl, Xte_red, ds.y_test)

    print("[5/8] FIX 1: mutual_info selector runs without warnings...")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mdl2, Xte2, k2 = prune_features(
            MODEL_FACTORIES["logreg"], Xtr, ds.y_train, Xte, "mutual_info", 0.5)
        bad = [x for x in w if "discrete" in str(x.message).lower()]
    assert not bad, f"MI still warning: {bad}"
    print(f"      OK, kept {k2} features, no discrete-value warnings")

    print("[6/8] FIX 2: pruned model reports a real size saving...")
    pruned = prune_weights(get_coef(base), 0.9)
    pm = apply_pruned_weights(base, pruned)
    dense = model_size_kb(pm)
    deploy = deployable_size_kb(pm)
    assert deploy < dense, f"no saving: dense={dense} deploy={deploy}"
    print(f"      dense={dense}KB -> deployable={deploy}KB "
          f"({dense/deploy:.1f}x smaller)")

    print("[7/8] Track B quantisation (linear + NB)...")
    q = full_evaluation(apply_quantised_weights(base, 8), Xte, ds.y_test)
    nb = build_search("naive_bayes", seed=config.SEED); nb.fit(Xtr, ds.y_train)
    assert count_parameters(nb.best_estimator_) > 0
    qnb = full_evaluation(apply_quantised_weights(nb.best_estimator_, 8), Xte, ds.y_test)
    print(f"      int8 logreg={q['accuracy']} int8 NB={qnb['accuracy']}")

    print("[8/8] metric completeness...")
    for k in ("accuracy", "macro_f1", "latency_ms", "peak_mem_mb", "rss_delta_mb",
              "model_size_kb", "model_size_dense_kb", "weight_density",
              "efficiency_index"):
        assert k in m, f"missing {k}"

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
