"""Track A — feature selection (chi2/MI/L1), doc-freq filtering, weight pruning."""
from __future__ import annotations
import copy
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_selection import (SelectFromModel, SelectKBest, chi2,
                                       mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from . import config


def _mutual_info_scorer(X, y, seed: int = None):
    """MI over BINARISED term presence.

    TF-IDF values are continuous, so discrete_features=True on the raw matrix is
    a false assumption. discrete_features=False is unavailable because sklearn
    rejects continuous features on a sparse matrix, and densifying 120k x 20k is
    not feasible. Scoring term PRESENCE keeps the matrix sparse and makes the
    discreteness assumption true, matching Yang and Pedersen (1997).
    """
    seed = config.SEED if seed is None else seed
    Xb = (X > 0).astype(np.int8)
    n = config.MI_SUBSAMPLE
    if n is not None and Xb.shape[0] > n:
        Xb, _, y, _ = train_test_split(Xb, y, train_size=n, stratify=y, random_state=seed)
    return mutual_info_classif(Xb, y, discrete_features=True, random_state=seed)


def make_feature_selector(method: str, keep_fraction: float, n_features: int,
                          seed: int = None):
    seed = config.SEED if seed is None else seed
    k = max(1, int(round(keep_fraction * n_features)))
    if method == "chi2":
        return SelectKBest(chi2, k=k)
    if method == "mutual_info":
        return SelectKBest(lambda X, y: _mutual_info_scorer(X, y, seed), k=k)
    if method == "l1":
        base = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                  random_state=seed)
        return SelectFromModel(base, max_features=k, threshold=-np.inf)
    raise ValueError(f"unknown selector: {method}")


def prune_features(estimator_factory, X_train, y_train, X_test, method,
                   keep_fraction, seed: int = None):
    selector = make_feature_selector(method, keep_fraction, X_train.shape[1], seed)
    Xtr = selector.fit_transform(X_train, y_train)
    Xte = selector.transform(X_test)
    model = estimator_factory()
    model.fit(Xtr, y_train)
    return model, csr_matrix(Xte), Xtr.shape[1]


def prune_weights(coef: np.ndarray, sparsity: float) -> np.ndarray:
    if sparsity <= 0:
        return coef.copy()
    flat = np.abs(coef).ravel()
    n_zero = int(round(sparsity * flat.size))
    if n_zero <= 0:
        return coef.copy()
    if n_zero >= flat.size:
        return np.zeros_like(coef)
    threshold = np.partition(flat, n_zero)[n_zero]
    pruned = coef.copy()
    pruned[np.abs(pruned) < threshold] = 0.0
    return pruned


def apply_pruned_weights(estimator, pruned_coef: np.ndarray):
    est = copy.deepcopy(estimator)
    clf = est.named_steps["clf"] if hasattr(est, "named_steps") else est
    if hasattr(clf, "coef_"):
        clf.coef_ = pruned_coef
    elif hasattr(clf, "feature_log_prob_"):
        clf.feature_log_prob_ = pruned_coef
    else:
        raise AttributeError("estimator has no prunable weight attribute")
    return est


def achieved_sparsity(coef: np.ndarray) -> float:
    return round(float(np.mean(coef == 0.0)), 4)
