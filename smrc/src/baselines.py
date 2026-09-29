"""The three baseline classifiers and their hyper-parameter search."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import config

MODEL_FACTORIES = {
    "naive_bayes": lambda: MultinomialNB(),
    "logreg": lambda: LogisticRegression(max_iter=2000),
    "linear_svm": lambda: LinearSVC(),
}


def build_search(model_key: str, seed: int = config.SEED) -> GridSearchCV:
    """Grid search with stratified k-fold CV on the training split only.

    The test split is never seen during tuning. The best configuration is
    refitted on the full training split.
    """
    pipeline = Pipeline([("clf", MODEL_FACTORIES[model_key]())])
    folds = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    return GridSearchCV(pipeline, param_grid=config.PARAM_GRIDS[model_key],
                        scoring="f1_macro", cv=folds, n_jobs=-1, refit=True)


def fit_tuned(model_key: str, X_train, y_train, seed: int = config.SEED):
    """Tune and fit one model. Returns (best_estimator, best_params)."""
    search = build_search(model_key, seed)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def classifier(estimator):
    """Return the classifier inside a pipeline, or the estimator itself."""
    return estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator


def get_weights(estimator):
    """The main learned parameter array.

    coef_ for the linear models, feature_log_prob_ for Naive Bayes.
    """
    clf = classifier(estimator)
    for attr in ("coef_", "feature_log_prob_"):
        if hasattr(clf, attr):
            return getattr(clf, attr)
    return None
