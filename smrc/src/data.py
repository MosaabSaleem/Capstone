"""Loading AG News and building TF-IDF features.

The loader tries several sources in turn so it works on a laptop, on Kaggle
and on Colab:

  1. train.csv and test.csv placed in the data/ folder
  2. the Hugging Face datasets library
  3. a direct download of the original CSV files
  4. a small synthetic dataset, used by the smoke test
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
    text = _NON_WORD.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", text).strip()


@dataclass
class Dataset:
    X_train_text: list
    X_test_text: list
    y_train: np.ndarray
    y_test: np.ndarray

    def vectorise(self, **overrides):
        """Fit TF-IDF on the training text and transform both splits.

        Any TfidfVectorizer argument can be overridden, which is how the
        document-frequency sweep changes min_df and max_df.
        Returns (X_train, X_test, vectoriser) with both matrices in CSR format.
        """
        params = dict(max_features=config.TFIDF_MAX_FEATURES,
                      ngram_range=config.TFIDF_NGRAM_RANGE,
                      min_df=config.TFIDF_MIN_DF,
                      max_df=config.TFIDF_MAX_DF,
                      sublinear_tf=True)
        params.update(overrides)
        vec = TfidfVectorizer(**params)
        X_train = vec.fit_transform(self.X_train_text)
        X_test = vec.transform(self.X_test_text)
        return csr_matrix(X_train), csr_matrix(X_test), vec


def _from_frames(train: pd.DataFrame, test: pd.DataFrame) -> Dataset:
    """Build a Dataset from the original CSV layout (labels start at 1)."""
    def split(df):
        text = df["title"].fillna("") + " " + df["description"].fillna("")
        return [clean_text(t) for t in text], df["label"].to_numpy() - 1

    X_train, y_train = split(train)
    X_test, y_test = split(test)
    return Dataset(X_train, X_test, y_train, y_test)


def _load_local() -> Dataset:
    train_path = config.DATA_DIR / "train.csv"
    test_path = config.DATA_DIR / "test.csv"
    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError("no train.csv and test.csv in data/")
    return _from_frames(pd.read_csv(train_path, header=None, names=_CSV_COLUMNS),
                        pd.read_csv(test_path, header=None, names=_CSV_COLUMNS))


def _load_hf() -> Dataset:
    from datasets import load_dataset

    ds = load_dataset(config.DATASET_NAME, cache_dir=str(config.DATA_DIR))
    train, test = ds["train"], ds["test"]
    return Dataset([clean_text(t) for t in train["text"]],
                   [clean_text(t) for t in test["text"]],
                   np.array(train["label"]),
                   np.array(test["label"]))


def _load_csv_download() -> Dataset:
    return _from_frames(pd.read_csv(_CSV_URL + "train.csv", header=None, names=_CSV_COLUMNS),
                        pd.read_csv(_CSV_URL + "test.csv", header=None, names=_CSV_COLUMNS))


def make_synthetic(n_per_class: int = 200, seed: int = config.SEED) -> Dataset:
    """A tiny, deterministic four-class dataset for testing the pipeline."""
    rng = np.random.default_rng(seed)
    vocab = {
        0: ["nation", "government", "border", "election", "treaty", "conflict"],
        1: ["match", "team", "score", "player", "league", "tournament"],
        2: ["market", "stocks", "profit", "revenue", "trade", "economy"],
        3: ["software", "chip", "research", "space", "device", "algorithm"],
    }
    fillers = ["the", "a", "of", "in", "on", "and"]
    texts, labels = [], []
    for label, words in vocab.items():
        for _ in range(n_per_class):
            k = rng.integers(6, 12)
            tokens = list(rng.choice(fillers, size=k)) + list(rng.choice(words, size=k))
            texts.append(clean_text(" ".join(tokens)))
            labels.append(label)

    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    labels = np.array(labels)[order]
    cut = int(0.8 * len(texts))
    return Dataset(texts[:cut], texts[cut:], labels[:cut], labels[cut:])


def _subsample_train(ds: Dataset) -> Dataset:
    """Shrink the training set for quick mode. The test set is never changed."""
    n = config.TRAIN_SUBSAMPLE
    if n is None or len(ds.X_train_text) <= n:
        return ds
    X, _, y, _ = train_test_split(ds.X_train_text, ds.y_train, train_size=n,
                                  stratify=ds.y_train, random_state=config.SEED)
    return Dataset(list(X), ds.X_test_text, np.asarray(y), ds.y_test)


def load(source: str = "auto") -> Dataset:
    """Load AG News.

    source is one of "auto", "local", "hf", "csv" or "synthetic".
    "auto" tries local files, then Hugging Face, then the CSV download.
    """
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
