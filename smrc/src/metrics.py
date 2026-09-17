"""Points 3 & 5 — native CPU benchmarking and the efficiency index."""
from __future__ import annotations
import gc, pickle, time, tracemalloc
from typing import Any
import numpy as np
import psutil
from sklearn.metrics import accuracy_score, f1_score
from . import config

_PROC = psutil.Process()


def predictive_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def model_size_kb(model: Any) -> float:
    return round(len(pickle.dumps(model)) / 1024, 2)


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
    out["model_size_kb"] = model_size_kb(model)
    out["efficiency_index"] = efficiency_index(
        out["accuracy"], out["model_size_kb"] / 1024, out["latency_ms"])
    return out
