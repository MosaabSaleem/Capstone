"""Track B: simulated low-precision quantisation for classical models.

Quantisation is usually described for neural network layers, but any model
that stores its knowledge in numeric arrays can be quantised the same way.

  Logistic Regression and linear SVM   coef_ (weights) and intercept_ (bias)
  Multinomial Naive Bayes              feature_log_prob_ and class_log_prior_

This is simulated ("fake") quantisation. Each array is rounded onto a low-bit
integer grid and then converted back to floating point. Prediction still runs
in ordinary float arithmetic, but it carries exactly the rounding error a real
low-precision implementation would introduce.
"""
from __future__ import annotations

import copy

import numpy as np

from .baselines import classifier

PARAMETER_ARRAYS = ("coef_", "intercept_", "feature_log_prob_", "class_log_prior_")


def quantise_array(values, bits: int, scheme: str = "symmetric") -> np.ndarray:
    """Round an array onto a grid with 2 ** bits levels, then map it back."""
    values = np.asarray(values, dtype=np.float64)
    q_max = 2 ** (bits - 1) - 1  # 127 for 8 bits, 7 for 4 bits

    if scheme == "symmetric":
        scale = (np.max(np.abs(values)) or 1.0) / q_max
        return np.clip(np.round(values / scale), -q_max - 1, q_max) * scale

    lo, hi = values.min(), values.max()
    if hi == lo:
        return values.copy()
    scale = (hi - lo) / (2 * q_max + 1)
    zero_point = round(-lo / scale)
    q = np.clip(np.round(values / scale) + zero_point, 0, 2 * q_max + 1)
    return (q - zero_point) * scale


def quantise_model(estimator, bits: int, scheme: str = "symmetric"):
    """A copy of a fitted model with every learned parameter array quantised."""
    est = copy.deepcopy(estimator)
    clf = classifier(est)
    changed = 0
    for attr in PARAMETER_ARRAYS:
        original = getattr(clf, attr, None)
        if original is not None:
            original = np.asarray(original)
            setattr(clf, attr, quantise_array(original, bits, scheme).astype(original.dtype))
            changed += 1
    if not changed:
        raise AttributeError("estimator has no parameter arrays to quantise")
    return est


def count_parameters(estimator) -> int:
    clf = classifier(estimator)
    return sum(np.asarray(getattr(clf, a)).size
               for a in PARAMETER_ARRAYS if getattr(clf, a, None) is not None)


def packed_size_kb(n_params: int, bits: int) -> float:
    """Size of the parameters if they were stored at the given bit-width."""
    return round(n_params * bits / 8 / 1024, 3)
