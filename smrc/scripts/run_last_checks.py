"""Last checks, answering the final round of supervision questions.

  lorenz     Cumulative weight distribution (Lorenz curves) for Logistic
             Regression and the linear SVM at C=0.1 and C=1.0, plus the F1 drop
             each suffers at 95 percent weight pruning. Draws one figure and logs
             the numbers, so the weight-concentration hypothesis can be confirmed
             or rejected visually.

  edgecases  Production edge cases on real data:
               - documents that become empty after vectorising (all words out of
                 vocabulary), before and after L1 feature pruning
               - accuracy by document length, to see whether pruning hurts
                 short documents more than long ones
               - quantiser and sparse-format behaviour on awkward arrays
                 (constant, all zero, one extreme outlier, -inf, 100% sparsity)

  overhead   Splits single-request latency into fixed overhead and real work.
             Times the full scikit-learn predict call, the same call on an empty
             document (the overhead floor), and the bare matrix arithmetic with
             no scikit-learn code around it, each over thousands of repeats.

Results go to results/last_checks.csv and figures/weight_lorenz.png. No earlier
results file is modified. Running a stage replaces any earlier rows for it.

    python scripts/run_last_checks.py              all stages
    python scripts/run_last_checks.py lorenz       one stage only
    python scripts/run_last_checks.py --quick      reduced run for checking
"""
import _common  # noqa: F401  (must come first)

import csv
import os
import statistics
import tempfile
import time
import timeit
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from src import config, data
from src.baselines import classifier, fit_tuned, get_weights
from src.pruning import prune_weights, with_weights
from src.utils import log_result, set_seed

OUT = config.RESULTS_DIR / "last_checks.csv"
FIGURE = config.FIGURES_DIR / "weight_lorenz.png"
TIMING_REPEATS = 200 if config.QUICK else 3000


def _log(row):
    log_result(row, OUT)
    print("  " + " | ".join(f"{k}={v}" for k, v in row.items()))


def drop_stage(stage: str) -> None:
    if not OUT.exists():
        return
    with OUT.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        kept = [r for r in reader if r.get("stage") != stage]
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)


def f1(y, pred) -> float:
    return round(float(f1_score(y, pred, average="macro")), 4)


# ---------------------------------------------------------------------------
# Stage 1: Lorenz curves of weight magnitude
# ---------------------------------------------------------------------------
def lorenz_points(w: np.ndarray):
    """x = fraction of weights (largest first), y = share of total |weight| held."""
    v = np.sort(np.abs(w).ravel())[::-1]
    cum = np.cumsum(v) / v.sum()
    x = np.arange(1, v.size + 1) / v.size
    return x, cum


def lorenz(ds, X_train, X_test, vec):
    print("\nWeight concentration (Lorenz curves)")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for key, factory in (("logreg", lambda c: LogisticRegression(C=c, max_iter=2000)),
                         ("linear_svm", lambda c: LinearSVC(C=c))):
        for c in (0.1, 1.0):
            model = factory(c).fit(X_train, ds.y_train)
            w = model.coef_
            full = model.predict(X_test)
            pruned = with_weights(model, prune_weights(w, 0.95)).predict(X_test)
            x, cum = lorenz_points(w)
            share_at_5 = float(np.interp(0.05, x, cum))
            drop = round((f1(ds.y_test, full) - f1(ds.y_test, pruned)) * 100, 2)
            label = f"{key} C={c}: top 5% hold {share_at_5:.0%}, F1 drop {drop} pts"
            style = "-" if key == "linear_svm" else "--"
            ax.plot(x, cum, style, label=label)
            _log({"stage": "lorenz", "model": key, "C": c,
                  "share_top1": round(float(np.interp(0.01, x, cum)), 4),
                  "share_top5": round(share_at_5, 4),
                  "share_top10": round(float(np.interp(0.10, x, cum)), 4),
                  "f1_drop_at_95": drop})
    ax.axvline(0.05, color="grey", linewidth=0.8, linestyle=":")
    ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1),
           xlabel="Fraction of weights kept, largest first (log scale)",
           ylabel="Share of total absolute weight",
           title="Weight concentration against robustness to pruning")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.savefig(FIGURE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGURE}")


# ---------------------------------------------------------------------------
# Stage 2: production edge cases
# ---------------------------------------------------------------------------
def q_asymmetric(v, bits):
    lo, hi = v.min(), v.max()
    if hi == lo:
        return v.copy()
    scale = (hi - lo) / (2 ** bits - 1)
    return lo + np.round((v - lo) / scale) * scale


def q_symmetric(v, bits):
    top = np.max(np.abs(v))
    if top == 0:
        return v.copy()
    q = 2 ** (bits - 1) - 1
    scale = top / q
    return np.clip(np.round(v / scale), -q - 1, q) * scale


def sparse_roundtrip_ok(w: np.ndarray) -> tuple:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "w.npz"
        m = sparse.csr_matrix(w)
        np.savez_compressed(path, data=m.data, indices=m.indices, indptr=m.indptr,
                            shape=np.array(m.shape))
        a = np.load(path)
        back = sparse.csr_matrix((a["data"], a["indices"], a["indptr"]),
                                 shape=tuple(a["shape"])).toarray()
        return bool(np.array_equal(back, w)), round(os.path.getsize(path) / 1024, 3)


def edgecases(ds, X_train, X_test, vec):
    print("\nEdge cases")
    svm, _ = fit_tuned("linear_svm", X_train, ds.y_train)
    nb, _ = fit_tuned("naive_bayes", X_train, ds.y_train)

    # Empty documents: rows with no non-zero features
    sel = SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                             random_state=config.SEED),
                          max_features=max(1, round(0.1 * X_train.shape[1])),
                          threshold=-np.inf).fit(X_train, ds.y_train)
    X_small = sel.transform(X_test)
    empty_full = int((X_test.getnnz(axis=1) == 0).sum())
    empty_small = int((X_small.getnnz(axis=1) == 0).sum())
    _log({"stage": "edgecases", "check": "empty_docs_full_vocab", "value": empty_full,
          "of": X_test.shape[0]})
    _log({"stage": "edgecases", "check": "empty_docs_after_l1_10pct", "value": empty_small,
          "of": X_test.shape[0]})

    # What do the models predict for a document made only of unknown words?
    oov = vec.transform(["zzqx qwvb xxyyzz"])
    _log({"stage": "edgecases", "check": "oov_doc_prediction_svm",
          "value": int(svm.predict(oov)[0]),
          "detail": "all-zero input, decided by the intercept alone"})
    _log({"stage": "edgecases", "check": "oov_doc_prediction_nb",
          "value": int(nb.predict(oov)[0]),
          "detail": "all-zero input, decided by the class priors alone"})

    # Accuracy by document length, full vocabulary against L1 10 percent
    small_model, _ = fit_tuned("linear_svm", sel.transform(X_train), ds.y_train)
    pruned = with_weights(svm, prune_weights(get_weights(svm), 0.95))
    lengths = np.array([len(t.split()) for t in ds.X_test_text])
    edges = np.quantile(lengths, [0, 0.25, 0.5, 0.75, 1.0])
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        mask = (lengths >= lo) & (lengths <= hi) if i == 3 else (lengths >= lo) & (lengths < hi)
        if mask.sum() == 0:
            continue
        y = ds.y_test[mask]
        _log({"stage": "edgecases", "check": f"length_quartile_{i + 1}",
              "words": f"{int(lo)}-{int(hi)}", "n_docs": int(mask.sum()),
              "f1_baseline": f1(y, svm.predict(X_test[mask])),
              "f1_weight_prune_95": f1(y, pruned.predict(X_test[mask])),
              "f1_l1_10pct": f1(y, small_model.predict(X_small[mask])),
              "empty_after_l1": int((X_small[mask].getnnz(axis=1) == 0).sum())})

    # Quantiser and serialisation on awkward arrays
    rng = np.random.default_rng(config.SEED)
    outlier = rng.normal(0, 0.1, 1000)
    outlier[0] = 50.0
    cases = {
        "constant": np.full(10, -1.386),
        "all_zero": np.zeros(10),
        "single_outlier": outlier,
        "contains_neg_inf": np.array([-3.0, -5.0, -np.inf, -4.0]),
    }
    for name, arr in cases.items():
        with np.errstate(all="ignore"):
            for qname, q in (("symmetric", q_symmetric), ("asymmetric", q_asymmetric)):
                out = q(arr, 4)
                finite = bool(np.all(np.isfinite(out)))
                zeroed = round(float(np.mean((out == 0) & (arr != 0))), 4)
                _log({"stage": "edgecases", "check": f"quant_{qname}_{name}",
                      "finite": finite, "share_zeroed": zeroed})

    for s in (0.0, 0.95, 1.0):
        w = prune_weights(get_weights(svm), s)
        ok, kb = sparse_roundtrip_ok(w)
        _log({"stage": "edgecases", "check": f"sparse_roundtrip_sparsity_{s}",
              "identical": ok, "file_kb": kb})


# ---------------------------------------------------------------------------
# Stage 3: separating overhead from real work in single-request latency
# ---------------------------------------------------------------------------
def median_ms(fn, repeats=TIMING_REPEATS) -> float:
    fn()
    times = timeit.repeat(fn, number=1, repeat=repeats)
    return round(statistics.median(times) * 1000, 5)


def overhead(ds, X_train, X_test, vec):
    print("\nSingle-request latency: overhead against real work")
    sel = SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                             random_state=config.SEED),
                          max_features=max(1, round(0.1 * X_train.shape[1])),
                          threshold=-np.inf).fit(X_train, ds.y_train)
    configs = []
    for key in ("logreg", "linear_svm"):
        base, _ = fit_tuned(key, X_train, ds.y_train)
        configs.append((key, "baseline", base, X_test))
        configs.append((key, "weight_prune_95",
                        with_weights(base, prune_weights(get_weights(base), 0.95)), X_test))
        small, _ = fit_tuned(key, sel.transform(X_train), ds.y_train)
        configs.append((key, "l1_keep_0.1", small, sel.transform(X_test)))

    for key, name, model, X in configs:
        clf = classifier(model)
        W = np.ascontiguousarray(clf.coef_.T)
        b = clf.intercept_
        W_sparse = sparse.csr_matrix(clf.coef_.T)
        doc = X[0:1]
        empty = sparse.csr_matrix((1, X.shape[1]))
        sklearn_ms = median_ms(lambda: model.predict(doc))
        floor_ms = median_ms(lambda: model.predict(empty))
        raw_ms = median_ms(lambda: np.argmax(doc @ W + b, axis=1))
        raw_sparse_ms = median_ms(lambda: np.argmax((doc @ W_sparse).toarray() + b, axis=1))
        _log({"stage": "overhead", "model": key, "config": name,
              "sklearn_single_ms": sklearn_ms,
              "empty_doc_floor_ms": floor_ms,
              "real_work_ms": round(max(0.0, sklearn_ms - floor_ms), 5),
              "raw_dense_matmul_ms": raw_ms,
              "raw_sparse_matmul_ms": raw_sparse_ms,
              "overhead_share": round(floor_ms / sklearn_ms, 3) if sklearn_ms else ""})

    # The vectoriser is part of every real request too
    text = ds.X_test_text[0]
    _log({"stage": "overhead", "model": "vectoriser", "config": "full_vocabulary",
          "sklearn_single_ms": median_ms(lambda: vec.transform([text]))})


STAGES = {"lorenz": lorenz, "edgecases": edgecases, "overhead": overhead}


def main():
    args = _common.parse_args(__doc__, stages=list(STAGES))
    set_seed(config.SEED)
    ds = data.load(args.data)
    X_train, X_test, vec = ds.vectorise()
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        drop_stage(name)
        start = time.perf_counter()
        STAGES[name](ds, X_train, X_test, vec)
        print(f"  ({name} took {(time.perf_counter() - start) / 60:.1f} min)")
    print(f"\nLogged to {OUT}")


if __name__ == "__main__":
    main()
