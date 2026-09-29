"""Track A: feature selection and magnitude weight pruning."""
from __future__ import annotations

import copy

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from . import config
from .baselines import classifier


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def mutual_info_scores(X, y, seed: int = config.SEED):
    """Mutual information between each term and the class label.

    Scores are computed on term presence (does the term appear or not) rather
    than on the TF-IDF weight. This is the classical formulation used in text
    feature selection. It keeps the matrix sparse, and it means the
    discrete_features=True setting is an accurate description of the input.
    Scoring is done on a stratified subsample to keep the runtime reasonable.
    """
    presence = (X > 0).astype(np.int8)
    n = config.MI_SUBSAMPLE
    if n is not None and presence.shape[0] > n:
        presence, _, y, _ = train_test_split(presence, y, train_size=n,
                                             stratify=y, random_state=seed)
    return mutual_info_classif(presence, y, discrete_features=True, random_state=seed)


def make_selector(method: str, keep_fraction: float, n_features: int, seed: int = config.SEED):
    """Build a selector that keeps the top keep_fraction of the vocabulary."""
    k = max(1, round(keep_fraction * n_features))
    if method == "chi2":
        return SelectKBest(chi2, k=k)
    if method == "mutual_info":
        return SelectKBest(lambda X, y: mutual_info_scores(X, y, seed), k=k)
    if method == "l1":
        lasso_like = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=seed)
        return SelectFromModel(lasso_like, max_features=k, threshold=-np.inf)
    raise ValueError(f"unknown selector: {method}")


def prune_features(model_factory, X_train, y_train, X_test, method: str,
                   keep_fraction: float, seed: int = config.SEED):
    """Select features on the training split, then fit a fresh model on them.

    The fresh model uses its default settings. Returns
    (fitted_model, reduced_X_test, number_of_features_kept).
    """
    selector = make_selector(method, keep_fraction, X_train.shape[1], seed)
    X_train_small = selector.fit_transform(X_train, y_train)
    X_test_small = selector.transform(X_test)
    model = model_factory()
    model.fit(X_train_small, y_train)
    return model, csr_matrix(X_test_small), X_train_small.shape[1]


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


def with_weights(estimator, new_weights: np.ndarray):
    """A copy of a fitted estimator with its main weight array replaced."""
    est = copy.deepcopy(estimator)
    clf = classifier(est)
    for attr in ("coef_", "feature_log_prob_"):
        if hasattr(clf, attr):
            setattr(clf, attr, new_weights)
            return est
    raise AttributeError("estimator has no weight array to prune")


def achieved_sparsity(weights: np.ndarray) -> float:
    return round(float(np.mean(weights == 0)), 4)
