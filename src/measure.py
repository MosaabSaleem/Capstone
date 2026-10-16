"""Measuring predictive quality, size, memory and latency.

Measurement uses the Python standard library plus psutil:
  time.perf_counter   latency
  tracemalloc         peak memory allocated by Python during prediction
  psutil              change in the whole process's resident memory
  pickle / joblib     model size, in memory and as real files on disk
"""
from __future__ import annotations

import copy
import gc
import os
import pickle
import statistics
import tempfile
import time
import timeit
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
import psutil
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score

from . import config
from .models import WEIGHT_ATTRS, classifier

_PROCESS = psutil.Process()


# ---------------------------------------------------------------------------
# Predictive quality
# ---------------------------------------------------------------------------
def macro_f1(y_true, y_pred) -> float:
    return round(float(f1_score(y_true, y_pred, average="macro")), 4)


def predictive(y_true, y_pred) -> dict:
    """Accuracy and macro-F1. Macro averaging weights all four classes equally."""
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": macro_f1(y_true, y_pred)}


def agreement(pred_a, pred_b) -> float:
    """Fraction of documents on which two models make the same prediction."""
    return round(float(np.mean(np.asarray(pred_a) == np.asarray(pred_b))), 4)


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------
def dense_size_kb(model) -> float:
    """Size of the model pickled exactly as it is held in memory."""
    return round(len(pickle.dumps(model)) / 1024, 2)


def deployable_size_kb(model) -> float:
    """In-memory size if mostly-zero weight arrays are stored sparsely.

    Zeroing weights in a dense array does not make its pickle smaller, so a
    pruned model would show no saving. Any weight array below the sparsity
    threshold is converted to CSR before measuring, and the smaller of the
    dense and sparse sizes is returned.
    """
    candidate = copy.deepcopy(model)
    clf = classifier(candidate)
    converted = False
    for attr in WEIGHT_ATTRS:
        w = getattr(clf, attr, None)
        if isinstance(w, np.ndarray) and w.size and np.mean(w != 0) < config.SPARSE_STORAGE_THRESHOLD:
            setattr(clf, attr, sparse.csr_matrix(w))
            converted = True
    dense = len(pickle.dumps(model))
    if not converted:
        return round(dense / 1024, 2)
    return round(min(dense, len(pickle.dumps(candidate))) / 1024, 2)


def weight_density(model) -> float:
    """Fraction of the main weight array that is non-zero."""
    clf = classifier(model)
    for attr in WEIGHT_ATTRS:
        w = getattr(clf, attr, None)
        if isinstance(w, np.ndarray) and w.size:
            return round(float(np.mean(w != 0)), 4)
    return 1.0


def save_sparse_npz(model, path: Path) -> None:
    """Save a linear model as a compressed archive of only its non-zero weights."""
    clf = classifier(model)
    w = sparse.csr_matrix(clf.coef_)
    np.savez_compressed(path, data=w.data, indices=w.indices, indptr=w.indptr,
                        shape=np.array(w.shape), intercept=clf.intercept_, classes=clf.classes_)


def load_sparse_npz(path: Path, template):
    """Rebuild a model saved by save_sparse_npz, using template for its class."""
    a = np.load(path)
    w = sparse.csr_matrix((a["data"], a["indices"], a["indptr"]),
                          shape=tuple(a["shape"])).toarray()
    model = copy.deepcopy(template)
    clf = classifier(model)
    clf.coef_ = np.ascontiguousarray(w)
    clf.intercept_, clf.classes_ = a["intercept"], a["classes"]
    return model


def disk_sizes(model, X_check) -> dict:
    """Write the model to real files in up to three formats.

    Returns {format: (size_kb, reload_identical)}. Each file is reloaded and its
    predictions on X_check compared with the original model's.
    """
    expected = model.predict(X_check)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        writers = {
            "joblib": (lambda p: joblib.dump(model, p), joblib.load, ".joblib"),
            "joblib_compressed": (lambda p: joblib.dump(model, p, compress=3), joblib.load, ".joblib"),
        }
        if hasattr(classifier(model), "coef_"):
            writers["sparse_npz"] = (lambda p: save_sparse_npz(model, p),
                                     lambda p: load_sparse_npz(p, model), ".npz")
        for name, (save, load, ext) in writers.items():
            path = tmp / f"{name}{ext}"
            save(path)
            same = bool(np.array_equal(load(path).predict(X_check), expected))
            out[name] = (round(os.path.getsize(path) / 1024, 2), same)
    return out


def smallest_disk_kb(obj) -> float:
    """Smallest real file size for any fitted object, in KB.

    Linear models are tried as compressed joblib and as a sparse archive, since
    neither format is smallest at every sparsity. Anything else is saved as
    compressed joblib.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = tmp / "obj.joblib"
        joblib.dump(obj, path, compress=3)
        sizes = [os.path.getsize(path)]
        if hasattr(obj, "predict") and hasattr(classifier(obj), "coef_"):
            npz = tmp / "obj.npz"
            save_sparse_npz(obj, npz)
            sizes.append(os.path.getsize(npz))
    return round(min(sizes) / 1024, 2)


# ---------------------------------------------------------------------------
# Memory and latency
# ---------------------------------------------------------------------------
def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def batch_latency_ms(predict, n_docs: int) -> float:
    """Milliseconds per document, fastest of several timed runs on one block."""
    predict()  # warm-up
    best = min(_time(predict) for _ in range(config.LATENCY_REPEATS))
    return round(best / n_docs * 1000, 6)


def peak_memory_mb(model, X, n: int = 512) -> float:
    X_block = X[: min(n, X.shape[0])]
    tracemalloc.start()
    model.predict(X_block)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return round(peak / 1024 ** 2, 3)


def rss_delta_mb(model, X, n: int = 2000) -> float:
    X_block = X[: min(n, X.shape[0])]
    gc.collect()
    before = _PROCESS.memory_info().rss
    model.predict(X_block)
    return round(max(0, _PROCESS.memory_info().rss - before) / 1024 ** 2, 3)


_EVICT = np.ones(64 * 1024 * 1024 // 8)  # 64 MB, larger than any CPU cache


def evict_caches() -> None:
    """Read a buffer larger than the CPU caches so the next prediction starts cold."""
    _EVICT.sum()


def latency_profile(predict_one, predict_batch, n_batch: int, load_fresh) -> dict:
    """Batch per document, warm single document and cold single document, in ms.

    predict_one(i) predicts document i on the loaded model. load_fresh() loads a
    new copy of the model and returns a predict_one function for it. For cold
    calls the copy is loaded first, the caches are evicted, and only the
    prediction is timed.
    """
    batch = batch_latency_ms(predict_batch, n_batch)
    predict_one(0)
    warm = [_time(lambda: predict_one(0)) * 1000 for _ in range(config.WARM_TRIALS)]
    cold = []
    for i in range(config.COLD_TRIALS):
        fresh = load_fresh()
        gc.collect()
        evict_caches()
        cold.append(_time(lambda: fresh(i + 1)) * 1000)
    return {"batch_ms_per_doc": batch,
            "warm_single_ms": round(statistics.median(warm), 5),
            "warm_single_p95_ms": round(float(np.percentile(warm, 95)), 5),
            "cold_single_ms": round(statistics.median(cold), 5)}


def median_ms(fn, repeats: int = None) -> float:
    """Median time of a very short call over many repeats, in ms."""
    repeats = repeats or config.OVERHEAD_REPEATS
    fn()
    return round(statistics.median(timeit.repeat(fn, number=1, repeat=repeats)) * 1000, 5)


# ---------------------------------------------------------------------------
# Efficiency index and the full evaluation
# ---------------------------------------------------------------------------
def efficiency_index(accuracy: float, size_kb: float, latency: float) -> float:
    """The original index: accuracy / (memory in MB x latency in ms)."""
    return round(accuracy / ((size_kb / 1024 + 1e-9) * (latency + 1e-9)), 4)


def evaluate(model, X_test, y_test) -> dict:
    """Every metric logged for a fitted model in the main sweeps."""
    pred = model.predict(X_test)
    out = predictive(y_test, pred)
    n = min(config.LATENCY_SAMPLE_SIZE, X_test.shape[0])
    block = X_test[:n]
    out["latency_ms"] = batch_latency_ms(lambda: model.predict(block), n)
    out["peak_mem_mb"] = peak_memory_mb(model, X_test)
    out["rss_delta_mb"] = rss_delta_mb(model, X_test)
    out["model_size_dense_kb"] = dense_size_kb(model)
    out["model_size_kb"] = deployable_size_kb(model)
    out["weight_density"] = weight_density(model)
    out["efficiency_index"] = efficiency_index(out["accuracy"], out["model_size_kb"],
                                               out["latency_ms"])
    return out


def sparsity_report(X, bytes_per_value: int = 8) -> dict:
    """Compare the CSR footprint of X with a dense float64 array of the same shape."""
    X = sparse.csr_matrix(X)
    total = X.shape[0] * X.shape[1]
    density = X.nnz / total if total else 0.0
    sparse_bytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    dense_bytes = total * bytes_per_value
    return {"n_rows": X.shape[0], "n_cols": X.shape[1], "nnz": int(X.nnz),
            "sparsity": round(1 - density, 6),
            "csr_mb": round(sparse_bytes / 1024 ** 2, 3),
            "dense_mb": round(dense_bytes / 1024 ** 2, 3),
            "saving": round(dense_bytes / sparse_bytes, 1) if sparse_bytes else 0.0}
