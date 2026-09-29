"""Load + preprocess AG News, and build TF-IDF features."""
from __future__ import annotations
import re
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from . import config

_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9\s]")


def clean_text(text: str) -> str:
    return _WS.sub(" ", _NONWORD.sub(" ", text.lower())).strip()


@dataclass
class Dataset:
    X_train_text: list
    X_test_text: list
    y_train: np.ndarray
    y_test: np.ndarray

    def vectorise(self, **kw):
        params = dict(max_features=config.TFIDF_MAX_FEATURES,
                      ngram_range=config.TFIDF_NGRAM_RANGE,
                      min_df=config.TFIDF_MIN_DF, max_df=config.TFIDF_MAX_DF,
                      sublinear_tf=True)
        params.update(kw)
        vec = TfidfVectorizer(**params)
        Xtr = vec.fit_transform(self.X_train_text)
        Xte = vec.transform(self.X_test_text)
        return csr_matrix(Xtr), csr_matrix(Xte), vec


def _prep(tr, te):
    def f(df):
        t = (df["title"].fillna("") + " " + df["description"].fillna(""))
        return [clean_text(x) for x in t], (df["label"].to_numpy() - 1)
    a, b = f(tr); c, d = f(te)
    return Dataset(a, c, b, d)


def _load_local_csv() -> Dataset:
    cols = ["label", "title", "description"]
    tr_p, te_p = config.DATA_DIR / "train.csv", config.DATA_DIR / "test.csv"
    if not (tr_p.exists() and te_p.exists()):
        raise FileNotFoundError("no local train.csv/test.csv in data/")
    return _prep(pd.read_csv(tr_p, header=None, names=cols),
                 pd.read_csv(te_p, header=None, names=cols))


def _load_from_hf() -> Dataset:
    from datasets import load_dataset
    ds = load_dataset(config.DATASET_NAME, cache_dir=str(config.DATA_DIR))
    tr, te = ds["train"], ds["test"]
    return Dataset([clean_text(t) for t in tr["text"]],
                   [clean_text(t) for t in te["text"]],
                   np.array(tr["label"]), np.array(te["label"]))


def _load_from_csv() -> Dataset:
    base = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
    cols = ["label", "title", "description"]
    return _prep(pd.read_csv(base + "train.csv", header=None, names=cols),
                 pd.read_csv(base + "test.csv", header=None, names=cols))


def make_synthetic(n_per_class: int = 200, seed: int = config.SEED) -> Dataset:
    rng = np.random.default_rng(seed)
    vocab = {0: ["nation", "government", "border", "election", "treaty", "conflict"],
             1: ["match", "team", "score", "player", "league", "tournament"],
             2: ["market", "stocks", "profit", "revenue", "trade", "economy"],
             3: ["software", "chip", "research", "space", "device", "algorithm"]}
    texts, labels = [], []
    for cls, words in vocab.items():
        for _ in range(n_per_class):
            k = rng.integers(6, 12)
            filler = rng.choice(["the", "a", "of", "in", "on", "and"], size=k)
            body = rng.choice(words, size=k)
            texts.append(clean_text(" ".join(list(filler) + list(body))))
            labels.append(cls)
    idx = rng.permutation(len(texts))
    texts = [texts[i] for i in idx]
    labels = np.array(labels)[idx]
    cut = int(0.8 * len(texts))
    return Dataset(texts[:cut], texts[cut:], labels[:cut], labels[cut:])


def load(prefer: str = "auto") -> Dataset:
    if prefer == "synthetic":
        return make_synthetic()
    if prefer == "local":
        return _load_local_csv()
    if prefer == "auto":
        try:
            return _load_local_csv()
        except Exception:
            pass
    if prefer in ("auto", "hf"):
        try:
            return _load_from_hf()
        except Exception as e:
            print(f"[data] HF load failed ({e!r}); trying CSV download.")
            if prefer == "hf":
                raise
    return _load_from_csv()
