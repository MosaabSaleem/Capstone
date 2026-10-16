"""Shared state for one pipeline run.

Loading data, vectorising it, tuning baselines and scoring features are the
expensive steps, and several stages need the same results. The Context does
each of them once and hands the result to every stage that asks.
"""
from __future__ import annotations

import time
from collections import defaultdict

import pandas as pd

from . import config, data
from .models import tune
from .optimise import selector_scores, top_terms


class Context:
    def __init__(self, source: str = "auto"):
        self.source = source
        self._loaded = False
        self._baselines = {}       # (model, seed) -> (fitted, params)
        self._scores = {}          # (selector, seed) -> scores
        self._feature_models = {}  # (model, selector, keep, seed) -> (fitted, params, columns)
        self.rows = defaultdict(list)
        self.timings = {}

    # ---------------------------------------------------------------- data
    def _load(self):
        if self._loaded:
            return
        start = time.perf_counter()
        self.ds = data.load(self.source)
        self.X_train, self.X_test, self.vec = self.ds.vectorise()
        self.y_train, self.y_test = self.ds.y_train, self.ds.y_test
        self._loaded = True
        print(f"Data loaded: train {self.X_train.shape}, test {self.X_test.shape} "
              f"({time.perf_counter() - start:.0f}s)")

    def __getattr__(self, name):
        # Load lazily the first time any data attribute is used.
        if name in ("ds", "X_train", "X_test", "vec", "y_train", "y_test"):
            self._load()
            return self.__dict__[name]
        raise AttributeError(name)

    # -------------------------------------------------------------- models
    def baseline(self, key: str, seed: int = config.SEED):
        if (key, seed) not in self._baselines:
            self._baselines[(key, seed)] = tune(key, self.X_train, self.y_train, seed)
        return self._baselines[(key, seed)]

    def scores(self, method: str, seed: int = config.SEED):
        if (method, seed) not in self._scores:
            start = time.perf_counter()
            self._scores[(method, seed)] = selector_scores(method, self.X_train, self.y_train, seed)
            print(f"  scored features with {method} (seed {seed}, "
                  f"{time.perf_counter() - start:.0f}s)")
        return self._scores[(method, seed)]

    def kept_columns(self, method: str, keep: float, seed: int = config.SEED):
        return top_terms(self.scores(method, seed), keep)

    def feature_model(self, key: str, method: str, keep: float, seed: int = config.SEED):
        """A model tuned on the top keep fraction of terms. Returns (fitted, params, columns)."""
        cache_key = (key, method, keep, seed)
        if cache_key not in self._feature_models:
            cols = self.kept_columns(method, keep, seed)
            if len(cols) == self.X_train.shape[1]:
                fitted, params = self.baseline(key, seed)
            else:
                fitted, params = tune(key, self.X_train[:, cols], self.y_train, seed)
            self._feature_models[cache_key] = (fitted, params, cols)
        return self._feature_models[cache_key]

    # ------------------------------------------------------------- results
    def add(self, table: str, row: dict, show: bool = True):
        self.rows[table].append(row)
        if show:
            keep = {k: v for k, v in row.items()
                    if k not in ("track", "best_params") and not str(k).endswith("dense_kb")}
            print("  " + " | ".join(f"{k}={v}" for k, v in list(keep.items())[:9]))

    def save(self, table: str):
        path = config.RESULTS_DIR / f"{table}.csv"
        pd.DataFrame(self.rows[table]).to_csv(path, index=False)
        print(f"  -> {path.name} ({len(self.rows[table])} rows)")


def read_table(table: str):
    """Read a saved results table, or None if it does not exist."""
    path = config.RESULTS_DIR / f"{table}.csv"
    return pd.read_csv(path) if path.exists() else None
