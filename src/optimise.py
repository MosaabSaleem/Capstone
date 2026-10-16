"""Feature selection, weight pruning and quantisation."""
from __future__ import annotations

import copy

import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from . import config
from .models import classifier

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
# Each selector produces one score per term. Scores do not depend on how many
# terms are kept, so they are computed once and every vocabulary budget simply
# keeps the top-scoring terms. This gives the same selection as refitting the
# selector at every budget, at a fraction of the cost.


def selector_scores(method: str, X, y, seed: int = config.SEED) -> np.ndarray:
    """One importance score per term. Higher means more useful."""
    if method == "chi2":
        scores, _ = chi2(X, y)
    elif method == "mutual_info":
        # Scored on term presence rather than the TF-IDF weight. This is the
        # classical formulation for text, keeps the matrix sparse, and makes
        # discrete_features=True an accurate description of the input.
        presence = (X > 0).astype(np.int8)
        n = config.MI_SUBSAMPLE
        y_mi = y
        if n is not None and presence.shape[0] > n:
            presence, _, y_mi, _ = train_test_split(presence, y, train_size=n,
                                                    stratify=y, random_state=seed)
        scores = mutual_info_classif(presence, y_mi, discrete_features=True,
                                     random_state=seed)
    elif method == "l1":
        # L1-regularised logistic regression drives most weights to zero.
        # A term's importance is the sum of its absolute weights across classes.
        model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=seed)
        model.fit(X, y)
        scores = np.abs(model.coef_).sum(axis=0)
    else:
        raise ValueError(f"unknown selector: {method}")
    return np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)


def top_terms(scores: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Column indices of the top keep_fraction of terms, in their original order."""
    k = max(1, round(keep_fraction * scores.size))
    if k >= scores.size:
        return np.arange(scores.size)
    order = np.argsort(-scores, kind="mergesort")  # stable, so ties are deterministic
    return np.sort(order[:k])


# ---------------------------------------------------------------------------
# Weight pruning
# ---------------------------------------------------------------------------
def prune_weights(weights: np.ndarray, sparsity: float) -> np.ndarray:
    """Zero the smallest-magnitude fraction of weights, using one global threshold."""
    pruned = weights.copy()
    n_zero = round(sparsity * weights.size)
    if n_zero <= 0:
        return pruned
    if n_zero >= weights.size:
        return np.zeros_like(weights)
    threshold = np.partition(np.abs(weights).ravel(), n_zero)[n_zero]
    pruned[np.abs(pruned) < threshold] = 0.0
    return pruned


def achieved_sparsity(weights: np.ndarray) -> float:
    return round(float(np.mean(weights == 0)), 4)


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------
# Simulated ("fake") quantisation: values are rounded onto a low-bit grid and
# converted back to floating point. Prediction still runs in float arithmetic
# but carries exactly the rounding error a real low-precision model would have.
# Every grid returns its input unchanged when all values are identical. AG News
# is perfectly balanced, so the four class log-priors are all log(0.25).

PARAMETER_ARRAYS = ("coef_", "intercept_", "feature_log_prob_", "class_log_prior_")


def q_symmetric(v: np.ndarray, bits: int) -> np.ndarray:
    """One scale for the whole array, grid centred on zero."""
    top = np.max(np.abs(v))
    if top == 0:
        return v.copy()
    q = 2 ** (bits - 1) - 1
    scale = top / q
    return np.clip(np.round(v / scale), -q - 1, q) * scale


def q_asymmetric(v: np.ndarray, bits: int) -> np.ndarray:
    """One scale, grid stretched over the actual minimum and maximum."""
    lo, hi = v.min(), v.max()
    if hi == lo:
        return v.copy()
    scale = (hi - lo) / (2 ** bits - 1)
    return lo + np.round((v - lo) / scale) * scale


def q_per_class(v: np.ndarray, bits: int) -> np.ndarray:
    """A separate asymmetric grid for each row (each class)."""
    v2 = np.atleast_2d(v)
    out = np.vstack([q_asymmetric(row, bits) for row in v2])
    return out.reshape(v.shape)


def q_quantile(v: np.ndarray, bits: int) -> np.ndarray:
    """Non-linear grid with levels at quantiles of the values."""
    flat = v.ravel()
    if flat.min() == flat.max():
        return v.copy()
    n = 2 ** bits
    edges = np.quantile(flat, np.linspace(0, 1, n + 1))
    idx = np.clip(np.searchsorted(edges, flat, side="right") - 1, 0, n - 1)
    centres = np.array([flat[idx == i].mean() if np.any(idx == i) else edges[i]
                        for i in range(n)])
    return centres[idx].reshape(v.shape)


GRIDS = {"symmetric": q_symmetric, "asymmetric": q_asymmetric,
         "per_class": q_per_class, "quantile": q_quantile}


def quantise_model(estimator, bits: int, scheme: str = config.QUANT_SCHEME):
    """A copy of a fitted model with every learned parameter array quantised."""
    est = copy.deepcopy(estimator)
    clf = classifier(est)
    grid = GRIDS[scheme]
    changed = 0
    for attr in PARAMETER_ARRAYS:
        original = getattr(clf, attr, None)
        if original is None:
            continue
        original = np.asarray(original)
        new = grid(original.astype(np.float64), bits).astype(original.dtype)
        if not np.all(np.isfinite(new)):
            raise ValueError(f"{scheme} grid at {bits} bits produced non-finite values in {attr}")
        setattr(clf, attr, new)
        changed += 1
    if not changed:
        raise AttributeError("estimator has no parameter arrays to quantise")
    return est


def count_parameters(estimator) -> int:
    clf = classifier(estimator)
    return int(sum(np.asarray(getattr(clf, a)).size
                   for a in PARAMETER_ARRAYS if getattr(clf, a, None) is not None))


def packed_size_kb(n_params: int, bits: int) -> float:
    """Size of the parameters if stored at the given bit-width."""
    return round(n_params * bits / 8 / 1024, 3)
