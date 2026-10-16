"""The three classifiers, their grid search, and helpers for their weights."""
from __future__ import annotations

import copy

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import config

FACTORIES = {
    "naive_bayes": lambda: MultinomialNB(),
    "logreg": lambda: LogisticRegression(max_iter=2000),
    "linear_svm": lambda: LinearSVC(),
}
WEIGHT_ATTRS = ("coef_", "feature_log_prob_")


def tune(key: str, X, y, seed: int = config.SEED):
    """Grid search with stratified k-fold CV on the training data only.

    The best configuration is refitted on all of the training data.
    Returns (fitted_pipeline, best_params).
    """
    search = GridSearchCV(
        Pipeline([("clf", FACTORIES[key]())]),
        param_grid=config.PARAM_GRIDS[key],
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, params


def classifier(estimator):
    """The classifier inside a pipeline, or the estimator itself."""
    return estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator


def get_weights(estimator) -> np.ndarray:
    """coef_ for the linear models, feature_log_prob_ for Naive Bayes."""
    clf = classifier(estimator)
    for attr in WEIGHT_ATTRS:
        if hasattr(clf, attr):
            return getattr(clf, attr)
    raise AttributeError("estimator has no weight array")


def with_weights(estimator, new_weights: np.ndarray):
    """A copy of a fitted estimator with its main weight array replaced."""
    est = copy.deepcopy(estimator)
    clf = classifier(est)
    for attr in WEIGHT_ATTRS:
        if hasattr(clf, attr):
            setattr(clf, attr, np.ascontiguousarray(new_weights))
            return est
    raise AttributeError("estimator has no weight array")
