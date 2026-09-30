"""Follow-up experiments for three questions raised in supervision.

  disk      Does weight pruning shrink the model file actually written to disk?
            Pruned models are saved three ways and the real file sizes are
            measured with os.path.getsize. Each file is reloaded and checked
            to predict exactly what the in-memory model predicts.

  nbquant   Does uniform quantisation damage the Naive Bayes log-probability
            tables, and do non-linear or finer-grained schemes protect them?
            Four schemes are compared at 8, 6, 4 and 3 bits.

  latency   Batch throughput against single-document cost, warm and cold.
            Cold calls load a fresh copy of the model and evict the CPU caches
            first, so the first prediction pays the full cache-miss cost.

Results are written to results/followups.csv and never touch results.csv.
Running a stage replaces any earlier rows for that stage, so re-runs do not
leave duplicates behind.

    python scripts/run_followups.py              all three stages
    python scripts/run_followups.py nbquant      one stage only
    python scripts/run_followups.py --quick      reduced run for checking
"""
import _common  # noqa: F401  (must come first)

import copy
import csv
import gc
import os
import pickle
import statistics
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from src import config, data
from src.baselines import classifier, fit_tuned, get_weights
from src.pruning import prune_weights, with_weights
from src.utils import log_result, set_seed

OUT = config.RESULTS_DIR / "followups.csv"
SPARSITIES = [0.0, 0.9, 0.95] if config.QUICK else [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
NB_BITS = [8, 4] if config.QUICK else [8, 6, 4, 3]
COLD_TRIALS = 5 if config.QUICK else 30
WARM_TRIALS = 50 if config.QUICK else 500


def _log(row):
    log_result(row, OUT)
    print("  " + " | ".join(f"{k}={v}" for k, v in row.items()))


def drop_stage(stage: str) -> None:
    """Remove earlier rows for a stage so a re-run replaces them."""
    if not OUT.exists():
        return
    with OUT.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        kept = [row for row in reader if row.get("stage") != stage]
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)


# ---------------------------------------------------------------------------
# Stage 1: real file sizes on disk
# ---------------------------------------------------------------------------
def save_sparse(model, path: Path) -> None:
    """Save a linear model with its weights in compressed sparse format.

    Only the non-zero weights and their positions are written, plus the small
    arrays needed to rebuild the model.
    """
    clf = classifier(model)
    weights = sparse.csr_matrix(clf.coef_)
    np.savez_compressed(path, intercept=clf.intercept_, classes=clf.classes_,
                        data=weights.data, indices=weights.indices,
                        indptr=weights.indptr, shape=np.array(weights.shape))


def load_sparse(path: Path, template):
    """Rebuild a model from save_sparse output, using template for its class."""
    arrays = np.load(path)
    weights = sparse.csr_matrix((arrays["data"], arrays["indices"], arrays["indptr"]),
                                shape=tuple(arrays["shape"])).toarray()
    model = copy.deepcopy(template)
    clf = classifier(model)
    clf.coef_, clf.intercept_, clf.classes_ = weights, arrays["intercept"], arrays["classes"]
    return model


def disk(ds, X_train, X_test):
    print("\nReal file sizes after weight pruning")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for key in ("logreg", "linear_svm"):
            base, _ = fit_tuned(key, X_train, ds.y_train)
            weights = get_weights(base)
            for s in SPARSITIES:
                model = with_weights(base, prune_weights(weights, s))
                expected = model.predict(X_test)
                formats = {
                    "joblib": (lambda p: joblib.dump(model, p), joblib.load, ".joblib"),
                    "joblib_compressed": (lambda p: joblib.dump(model, p, compress=3),
                                          joblib.load, ".joblib"),
                    "sparse_npz": (lambda p: save_sparse(model, p),
                                   lambda p: load_sparse(p, base), ".npz"),
                }
                for name, (save, load, ext) in formats.items():
                    path = tmp / f"{key}_{s}_{name}{ext}"
                    save(path)
                    _log({"stage": "disk", "model": key, "sparsity": s, "format": name,
                          "file_kb": round(os.path.getsize(path) / 1024, 2),
                          "reload_identical": bool(np.array_equal(load(path).predict(X_test),
                                                                  expected))})


# ---------------------------------------------------------------------------
# Stage 2: quantising the Naive Bayes log-probability tables
# ---------------------------------------------------------------------------
# Every scheme returns its input unchanged when all the values are identical.
# AG News is perfectly balanced, so the four class log-priors are all
# log(0.25). Without that guard the asymmetric grid has zero width, divides
# zero by zero and turns the priors into NaN.

def q_symmetric(v, bits):
    """One scale for the whole table, grid centred on zero."""
    top = np.max(np.abs(v))
    if top == 0:
        return v.copy()
    q = 2 ** (bits - 1) - 1
    scale = top / q
    return np.clip(np.round(v / scale), -q - 1, q) * scale


def q_asymmetric(v, bits):
    """One scale, grid stretched across the actual minimum and maximum.

    Log-probabilities are all negative, so a symmetric grid wastes its whole
    positive half. This uses every level on the range the values occupy.
    """
    lo, hi = v.min(), v.max()
    if hi == lo:
        return v.copy()
    scale = (hi - lo) / (2 ** bits - 1)
    return lo + np.round((v - lo) / scale) * scale


def q_per_class(v, bits):
    """A separate asymmetric grid for each class row."""
    return np.vstack([q_asymmetric(row, bits) for row in v])


def q_quantile(v, bits):
    """Non-linear binning. Levels are placed at quantiles of the values, so
    densely populated regions of the log-probability range get finer steps."""
    flat = v.ravel()
    if flat.min() == flat.max():
        return v.copy()
    n = 2 ** bits
    edges = np.quantile(flat, np.linspace(0, 1, n + 1))
    idx = np.clip(np.searchsorted(edges, flat, side="right") - 1, 0, n - 1)
    centres = np.array([flat[idx == i].mean() if np.any(idx == i) else edges[i]
                        for i in range(n)])
    return centres[idx].reshape(v.shape)


SCHEMES = {"symmetric": q_symmetric, "asymmetric": q_asymmetric,
           "per_class": q_per_class, "quantile": q_quantile}


def nbquant(ds, X_train, X_test):
    print("\nNaive Bayes log-probability quantisation")
    from sklearn.metrics import f1_score
    base, _ = fit_tuned("naive_bayes", X_train, ds.y_train)
    reference = base.predict(X_test)
    table = classifier(base).feature_log_prob_
    prior = classifier(base).class_log_prior_
    _log({"stage": "nbquant", "model": "naive_bayes", "bits": 32, "scheme": "none",
          "macro_f1": round(f1_score(ds.y_test, reference, average="macro"), 4),
          "agreement": 1.0, "max_abs_error": 0.0, "mean_abs_error": 0.0,
          "log_range": f"{table.min():.2f} to {table.max():.2f}"})

    for bits in NB_BITS:
        for name, fn in SCHEMES.items():
            model = copy.deepcopy(base)
            clf = classifier(model)
            clf.feature_log_prob_ = fn(table, bits)
            clf.class_log_prior_ = fn(prior.reshape(1, -1), bits).ravel()
            if not (np.all(np.isfinite(clf.feature_log_prob_))
                    and np.all(np.isfinite(clf.class_log_prior_))):
                raise ValueError(f"{name} at {bits} bits produced non-finite values")
            pred = model.predict(X_test)
            error = np.abs(clf.feature_log_prob_ - table)
            _log({"stage": "nbquant", "model": "naive_bayes", "bits": bits, "scheme": name,
                  "macro_f1": round(f1_score(ds.y_test, pred, average="macro"), 4),
                  "agreement": round(float(np.mean(pred == reference)), 4),
                  "max_abs_error": round(float(error.max()), 4),
                  "mean_abs_error": round(float(error.mean()), 4)})


# ---------------------------------------------------------------------------
# Stage 3: cold start against warm throughput
# ---------------------------------------------------------------------------
_EVICT = np.ones(64 * 1024 * 1024 // 8)  # 64 MB, larger than any CPU cache


def evict_caches():
    """Touch a buffer larger than the CPU caches so later reads start cold."""
    _EVICT.sum()


def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def latency(ds, X_train, X_test):
    print("\nLatency: batch throughput, warm single document, cold single document")
    for key in ("naive_bayes", "logreg", "linear_svm"):
        base, _ = fit_tuned(key, X_train, ds.y_train)
        blob = pickle.dumps(base)
        one = X_test[:1]

        # Batch throughput, the figure used in the main results.
        block = X_test[:config.LATENCY_SAMPLE_SIZE]
        base.predict(block[:64])
        batch = min(_time(lambda: base.predict(block)) for _ in range(config.LATENCY_REPEATS))
        batch_per_doc = batch / block.shape[0] * 1000

        # Warm single document: the same model called repeatedly.
        base.predict(one)
        warm = [_time(lambda: base.predict(one)) * 1000 for _ in range(WARM_TRIALS)]

        # Cold single document: fresh copy of the model, caches evicted first.
        cold = []
        for i in range(COLD_TRIALS):
            fresh = pickle.loads(blob)
            doc = X_test[i + 1:i + 2]
            gc.collect()
            evict_caches()
            cold.append(_time(lambda: fresh.predict(doc)) * 1000)

        _log({"stage": "latency", "model": key,
              "batch_ms_per_doc": round(batch_per_doc, 5),
              "warm_single_ms_median": round(statistics.median(warm), 4),
              "warm_single_ms_p95": round(float(np.percentile(warm, 95)), 4),
              "cold_single_ms_median": round(statistics.median(cold), 4),
              "cold_single_ms_max": round(max(cold), 4),
              "cold_over_warm": round(statistics.median(cold) / statistics.median(warm), 2),
              "single_over_batch": round(statistics.median(warm) / batch_per_doc, 1)})


STAGES = {"disk": disk, "nbquant": nbquant, "latency": latency}


def main():
    args = _common.parse_args(__doc__, stages=list(STAGES))
    set_seed(config.SEED)
    ds = data.load(args.data)
    X_train, X_test, _ = ds.vectorise()
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        drop_stage(name)
        STAGES[name](ds, X_train, X_test)
    print(f"\nLogged to {OUT}")


if __name__ == "__main__":
    main()
