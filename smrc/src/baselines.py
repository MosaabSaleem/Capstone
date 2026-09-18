"""Baseline models trained with nested/stratified cross-validation."""
from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from . import config

MODEL_FACTORIES = {
    "naive_bayes": lambda: MultinomialNB(),
    "logreg": lambda: LogisticRegression(max_iter=2000, n_jobs=-1),
    "linear_svm": lambda: LinearSVC(),
}
LINEAR_WEIGHT_MODELS = ("logreg", "linear_svm")


def build_search(model_key: str, seed: int = config.SEED) -> GridSearchCV:
    pipe = Pipeline([("clf", MODEL_FACTORIES[model_key]())])
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, param_grid=config.PARAM_GRIDS[model_key],
                        scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)


def get_coef(estimator):
    clf = estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator
    if hasattr(clf, "coef_"):
        return clf.coef_
    if hasattr(clf, "feature_log_prob_"):
        return clf.feature_log_prob_
    return None
