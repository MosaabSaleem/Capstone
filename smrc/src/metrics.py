"""Points 3 & 5 — native CPU benchmarking and the efficiency index."""
from __future__ import annotations
import copy, gc, pickle, time, tracemalloc
from typing import Any
import numpy as np
import psutil
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, f1_score
from . import config

_PROC = psutil.Process()
_WEIGHT_ATTRS = ("coef_", "feature_log_prob_")


def predictive_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def model_size_kb(model: Any) -> float:
    """Serialised size of the fitted model exactly as it stands (dense)."""
    return round(len(pickle.dumps(model)) / 1024, 2)


def deployable_size_kb(model: Any,
                       threshold: float = config.SPARSE_STORAGE_THRESHOLD) -> float:
    """Serialised size using the SENSIBLE storage format for the weights.

    FIX: zeroing entries of a dense numpy array does not shrink its pickle, so
    magnitude pruning previously showed no size saving at all. A pruned weight
    matrix would obviously be deployed in a sparse format, so when a weight
    array is mostly zeros we measure it as CSR instead. We return the smaller of
    the dense and sparse serialisations, which is what a real deployment would
    ship.
    """
    est = copy.deepcopy(model)
    clf = est.named_steps["clf"] if hasattr(est, "named_steps") else est
    converted = False
    for attr in _WEIGHT_ATTRS:
        arr = getattr(clf, attr, None)
        if arr is None or not isinstance(arr, np.ndarray):
            continue
        density = float(np.mean(arr != 0.0)) if arr.size else 1.0
        if density < threshold:
            setattr(clf, attr, csr_matrix(arr))
            converted = True
    if not converted:
        return model_size_kb(model)
    return round(min(len(pickle.dumps(est)), len(pickle.dumps(model))) / 1024, 2)


def weight_density(model: Any) -> float:
    """Fraction of non-zero entries in the model's primary weight array."""
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    for attr in _WEIGHT_ATTRS:
        arr = getattr(clf, attr, None)
        if isinstance(arr, np.ndarray) and arr.size:
            return round(float(np.mean(arr != 0.0)), 4)
    return 1.0


def measure_latency(model: Any, X, repeats: int = config.LATENCY_REPEATS,
                    sample_size: int = config.LATENCY_SAMPLE_SIZE) -> float:
    n = min(sample_size, X.shape[0])
    Xs = X[:n]
    model.predict(Xs[: min(64, n)])
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(Xs)
        best = min(best, time.perf_counter() - t0)
    return round((best / n) * 1000, 4)


def measure_peak_memory(model: Any, X, sample_size: int = 512) -> float:
    n = min(sample_size, X.shape[0])
    Xs = X[:n]
    tracemalloc.start()
    model.predict(Xs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return round(peak / (1024 * 1024), 3)


def measure_rss_delta(model: Any, X, sample_size: int = 2000) -> float:
    n = min(sample_size, X.shape[0])
    Xs = X[:n]
    gc.collect()
    before = _PROC.memory_info().rss
    model.predict(Xs)
    after = _PROC.memory_info().rss
    return round(max(0, after - before) / (1024 * 1024), 3)


def efficiency_index(accuracy: float, mem_mb: float, latency_ms: float,
                     alpha: float = config.EFFICIENCY_ALPHA,
                     beta: float = config.EFFICIENCY_BETA) -> float:
    eps = 1e-6
    denom = ((mem_mb + eps) ** alpha) * ((latency_ms + eps) ** beta)
    return round(accuracy / denom, 4)


def full_evaluation(model: Any, X_test, y_test) -> dict[str, float]:
    y_pred = model.predict(X_test)
    out = predictive_metrics(y_test, y_pred)
    out["latency_ms"] = measure_latency(model, X_test)
    out["peak_mem_mb"] = measure_peak_memory(model, X_test)
    out["rss_delta_mb"] = measure_rss_delta(model, X_test)
    # Dense footprint kept for transparency; deployable size drives the index.
    out["model_size_dense_kb"] = model_size_kb(model)
    out["model_size_kb"] = deployable_size_kb(model)
    out["weight_density"] = weight_density(model)
    out["efficiency_index"] = efficiency_index(
        out["accuracy"], out["model_size_kb"] / 1024, out["latency_ms"])
    return out
