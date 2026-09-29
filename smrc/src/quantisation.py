"""Track B — simulated quantisation for classical models."""
from __future__ import annotations
import copy
import numpy as np

QUANTISABLE_ATTRS = ("coef_", "intercept_", "feature_log_prob_", "class_log_prior_")


def quantise_array(w, bits: int, scheme: str = "symmetric"):
    w = np.asarray(w, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1
    if scheme == "symmetric":
        max_abs = float(np.max(np.abs(w))) or 1.0
        scale = max_abs / qmax
        q = np.clip(np.round(w / scale), -qmax - 1, qmax)
        return q * scale, scale, 0
    w_min, w_max = float(w.min()), float(w.max())
    if w_max == w_min:
        return w.copy(), 1.0, 0
    scale = (w_max - w_min) / (2 * qmax + 1)
    zp = int(round(-w_min / scale))
    q = np.clip(np.round(w / scale) + zp, 0, 2 * qmax + 1)
    return (q - zp) * scale, scale, zp


def _quantise_attr(clf, attr, bits, scheme) -> int:
    arr = getattr(clf, attr, None)
    if arr is None:
        return 0
    original = np.asarray(arr)
    deq, _, _ = quantise_array(original, bits, scheme)
    setattr(clf, attr, deq.astype(original.dtype))
    return int(original.size)


def apply_quantised_weights(estimator, bits: int, scheme: str = "symmetric"):
    est = copy.deepcopy(estimator)
    clf = est.named_steps["clf"] if hasattr(est, "named_steps") else est
    n = sum(_quantise_attr(clf, a, bits, scheme) for a in QUANTISABLE_ATTRS)
    if n == 0:
        raise AttributeError("estimator exposes no quantisable parameter arrays")
    return est


def count_parameters(estimator) -> int:
    clf = estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator
    return sum(int(np.asarray(getattr(clf, a)).size)
               for a in QUANTISABLE_ATTRS if getattr(clf, a, None) is not None)


def theoretical_size_kb(n_params: int, bits: int) -> float:
    return round((n_params * bits) / 8 / 1024, 3)
