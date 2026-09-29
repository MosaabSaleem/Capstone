"""End-to-end smoke test on tiny synthetic data (no downloads)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

    print("[2/8] sparsity report...")
    rep = analyse(Xtr)
    assert rep.dense_bytes >= rep.sparse_bytes

    print("[3/8] baseline with CV...")
    s = build_search("logreg", seed=config.SEED); s.fit(Xtr, ds.y_train)
    base = s.best_estimator_
    m = full_evaluation(base, Xte, ds.y_test)
    assert "efficiency_index" in m and "model_size_dense_kb" in m

    print("[4/8] feature pruning...")
    mdl, Xte_red, n_kept = prune_features(
        MODEL_FACTORIES["logreg"], Xtr, ds.y_train, Xte, "chi2", 0.5)
    assert n_kept <= Xtr.shape[1]

    print("[5/8] FIX: mutual_info runs without discrete-value warnings...")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prune_features(MODEL_FACTORIES["logreg"], Xtr, ds.y_train, Xte, "mutual_info", 0.5)
        bad = [x for x in w if "discrete" in str(x.message).lower()]
    assert not bad, f"MI still warning: {bad}"

    print("[6/8] FIX: pruned model reports a real size saving...")
    pm = apply_pruned_weights(base, prune_weights(get_coef(base), 0.9))
    dense, deploy = model_size_kb(pm), deployable_size_kb(pm)
    assert deploy < dense, f"no saving: {dense} -> {deploy}"
    print(f"      {dense}KB -> {deploy}KB ({dense/deploy:.1f}x)")

    print("[7/8] quantisation, linear and NB...")
    full_evaluation(apply_quantised_weights(base, 8), Xte, ds.y_test)
    nb = build_search("naive_bayes", seed=config.SEED); nb.fit(Xtr, ds.y_train)
    assert count_parameters(nb.best_estimator_) > 0
    full_evaluation(apply_quantised_weights(nb.best_estimator_, 8), Xte, ds.y_test)

    print("[8/8] multi-seed modules import...")
    import importlib.util
    for name in ("run_multiseed", "summarise_multiseed"):
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parent / f"{name}.py")
        assert spec is not None

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
