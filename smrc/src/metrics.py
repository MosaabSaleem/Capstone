"""Measuring predictive quality and computational cost.

All measurement uses the Python standard library plus psutil:

  time.perf_counter   inference latency
  tracemalloc         peak memory allocated by Python during prediction
  psutil              change in the whole process's resident memory
  pickle              serialised model size

The two memory figures answer different questions. tracemalloc is precise and
repeatable but cannot see memory allocated underneath Python, for example by
numpy. psutil sees everything the operating system holds for the process but is
noisier. Both are logged.
"""
from __future__ import annotations

import copy
import gc
import pickle
import time
import tracemalloc

import numpy as np
import psutil
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, f1_score

from . import config
from .baselines import classifier

_PROCESS = psutil.Process()
_WEIGHT_ATTRS = ("coef_", "feature_log_prob_")


def predictive_metrics(y_true, y_pred) -> dict:
    """Accuracy and macro-F1. Macro averaging weights all four classes equally."""
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4)}


def dense_size_kb(model) -> float:
    """Size of the model pickled exactly as it is held in memory."""
    return round(len(pickle.dumps(model)) / 1024, 2)


def deployable_size_kb(model) -> float:
    """Size of the model as it would sensibly be shipped.

    Setting weights to zero inside a dense array does not make its pickle any
    smaller, so measuring a pruned model naively shows no saving at all. A
    heavily pruned weight matrix would be stored sparsely in practice, so any
    weight array that is mostly zeros is converted to CSR before measuring.
    The smaller of the dense and sparse sizes is returned.
    """
    candidate = copy.deepcopy(model)
    clf = classifier(candidate)
    converted = False
    for attr in _WEIGHT_ATTRS:
        weights = getattr(clf, attr, None)
        if isinstance(weights, np.ndarray) and weights.size:
            if np.mean(weights != 0) < config.SPARSE_STORAGE_THRESHOLD:
                setattr(clf, attr, csr_matrix(weights))
                converted = True
    dense = len(pickle.dumps(model))
    if not converted:
        return round(dense / 1024, 2)
    return round(min(dense, len(pickle.dumps(candidate))) / 1024, 2)


def weight_density(model) -> float:
    """Fraction of the main weight array that is non-zero."""
    clf = classifier(model)
    for attr in _WEIGHT_ATTRS:
        weights = getattr(clf, attr, None)
        if isinstance(weights, np.ndarray) and weights.size:
            return round(float(np.mean(weights != 0)), 4)
    return 1.0


def latency_ms(model, X) -> float:
    """Milliseconds per sample, taking the fastest of several timed runs.

    Background activity on the machine can only ever slow a run down, so the
    fastest run is the one closest to the model's real speed.
    """
    n = min(config.LATENCY_SAMPLE_SIZE, X.shape[0])
    X_block = X[:n]
    model.predict(X_block[: min(64, n)])  # warm-up
    best = float("inf")
    for _ in range(config.LATENCY_REPEATS):
        start = time.perf_counter()
        model.predict(X_block)
        best = min(best, time.perf_counter() - start)
    return round(best / n * 1000, 4)


def peak_memory_mb(model, X, n: int = 512) -> float:
    """Peak Python-level allocation during one prediction call."""
    X_block = X[: min(n, X.shape[0])]
    tracemalloc.start()
    model.predict(X_block)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return round(peak / 1024 ** 2, 3)


def rss_delta_mb(model, X, n: int = 2000) -> float:
    """Growth in resident process memory across one prediction call."""
    X_block = X[: min(n, X.shape[0])]
    gc.collect()
    before = _PROCESS.memory_info().rss
    model.predict(X_block)
    after = _PROCESS.memory_info().rss
    return round(max(0, after - before) / 1024 ** 2, 3)


def efficiency_index(accuracy: float, memory_mb: float, latency: float) -> float:
    """Accuracy earned per unit of memory and latency. Higher is better.

    Because it is a ratio it should be used to rank configurations of similar
    accuracy, not read as an absolute quantity.
    """
    eps = 1e-6
    cost = (memory_mb + eps) ** config.EFFICIENCY_ALPHA * (latency + eps) ** config.EFFICIENCY_BETA
    return round(accuracy / cost, 4)


def evaluate(model, X_test, y_test) -> dict:
    """Every metric the project logs, for one fitted model."""
    out = predictive_metrics(y_test, model.predict(X_test))
    out["latency_ms"] = latency_ms(model, X_test)
    out["peak_mem_mb"] = peak_memory_mb(model, X_test)
    out["rss_delta_mb"] = rss_delta_mb(model, X_test)
    out["model_size_dense_kb"] = dense_size_kb(model)
    out["model_size_kb"] = deployable_size_kb(model)
    out["weight_density"] = weight_density(model)
    out["efficiency_index"] = efficiency_index(
        out["accuracy"], out["model_size_kb"] / 1024, out["latency_ms"])
    return out
