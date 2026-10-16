"""Loading AG News and building TF-IDF features.

With source "auto" the loader tries, in order:
  1. train.csv and test.csv in the data/ folder (original AG News layout)
  2. the Hugging Face datasets library
  3. a direct download of the original CSV files
The "synthetic" source is a small four-class dataset used by the smoke test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from . import config

_NON_WORD = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")
_CSV_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
_CSV_COLUMNS = ["label", "title", "description"]


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", text.lower())).strip()


def make_vectoriser(**overrides) -> TfidfVectorizer:
    params = dict(max_features=config.TFIDF_MAX_FEATURES,
                  ngram_range=config.TFIDF_NGRAM_RANGE,
                  min_df=config.TFIDF_MIN_DF,
                  max_df=config.TFIDF_MAX_DF,
                  sublinear_tf=True)
    params.update(overrides)
    return TfidfVectorizer(**params)


@dataclass
class Dataset:
    X_train_text: list
    X_test_text: list
    y_train: np.ndarray
    y_test: np.ndarray

    def vectorise(self, **overrides):
        """Fit TF-IDF on the training text. Returns (X_train, X_test, vectoriser)."""
        vec = make_vectoriser(**overrides)
        X_train = csr_matrix(vec.fit_transform(self.X_train_text))
        X_test = csr_matrix(vec.transform(self.X_test_text))
        return X_train, X_test, vec


def _from_frames(train: pd.DataFrame, test: pd.DataFrame) -> Dataset:
    def split(df):
        text = df["title"].fillna("") + " " + df["description"].fillna("")
        return [clean_text(t) for t in text], df["label"].to_numpy() - 1

    X_train, y_train = split(train)
    X_test, y_test = split(test)
    return Dataset(X_train, X_test, y_train, y_test)


def _load_local() -> Dataset:
    tr, te = config.DATA_DIR / "train.csv", config.DATA_DIR / "test.csv"
    if not (tr.exists() and te.exists()):
        raise FileNotFoundError("no train.csv and test.csv in data/")
    return _from_frames(pd.read_csv(tr, header=None, names=_CSV_COLUMNS),
                        pd.read_csv(te, header=None, names=_CSV_COLUMNS))


def _load_hf() -> Dataset:
    from datasets import load_dataset

    ds = load_dataset(config.DATASET_NAME, cache_dir=str(config.DATA_DIR))
    tr, te = ds["train"], ds["test"]
    return Dataset([clean_text(t) for t in tr["text"]], [clean_text(t) for t in te["text"]],
                   np.array(tr["label"]), np.array(te["label"]))


def _load_csv_download() -> Dataset:
    return _from_frames(pd.read_csv(_CSV_URL + "train.csv", header=None, names=_CSV_COLUMNS),
                        pd.read_csv(_CSV_URL + "test.csv", header=None, names=_CSV_COLUMNS))


def make_synthetic(n_per_class: int = 250, seed: int = config.SEED) -> Dataset:
    """A small, deterministic, balanced four-class dataset with overlapping vocabulary."""
    rng = np.random.default_rng(seed)
    topics = {
        0: "nation government border election treaty minister vote embassy",
        1: "match team score player league coach season goal",
        2: "market stocks profit trade revenue shares bank economy",
        3: "software chip research device space network data launch",
    }
    shared = "the a of in on report new said year week today group".split()
    texts, labels = [], []
    for label, words in topics.items():
        own = words.split()
        other = topics[(label + 1) % 4].split()
        for _ in range(n_per_class):
            n = int(rng.integers(4, 30))
            pool = own * 3 + shared * 2 + other
            texts.append(" ".join(rng.choice(pool, size=n)))
            labels.append(label)
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    labels = np.array(labels)[order]
    cut = int(0.8 * len(texts))
    return Dataset(texts[:cut], texts[cut:], labels[:cut], labels[cut:])


def _subsample_train(ds: Dataset) -> Dataset:
    """Quick mode only: shrink the training set. The test set never changes."""
    n = config.TRAIN_SUBSAMPLE
    if n is None or len(ds.X_train_text) <= n:
        return ds
    X, _, y, _ = train_test_split(ds.X_train_text, ds.y_train, train_size=n,
                                  stratify=ds.y_train, random_state=config.SEED)
    return Dataset(list(X), ds.X_test_text, np.asarray(y), ds.y_test)


def load(source: str = "auto") -> Dataset:
    if source == "synthetic":
        return make_synthetic()
    loaders = {"local": [_load_local], "hf": [_load_hf], "csv": [_load_csv_download],
               "auto": [_load_local, _load_hf, _load_csv_download]}
    if source not in loaders:
        raise ValueError(f"unknown data source: {source}")
    errors = []
    for loader in loaders[source]:
        try:
            return _subsample_train(loader())
        except Exception as exc:  # try the next source
            errors.append(f"{loader.__name__}: {exc!r}")
    raise RuntimeError("could not load AG News:\n  " + "\n  ".join(errors))
