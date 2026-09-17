"""Load + preprocess AG News, and build TF-IDF features.

Loader strategy (robust across Kaggle/Colab):
  1. local CSV in data/  (if you upload ag_news as a Kaggle dataset)
  2. HuggingFace `datasets`
  3. direct CSV download
  4. tiny synthetic dataset (smoke test / fully offline)
"""
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
    text = text.lower()
    text = _NONWORD.sub(" ", text)
    return _WS.sub(" ", text).strip()


@dataclass
class Dataset:
    X_train_text: list
    X_test_text: list
    y_train: np.ndarray
    y_test: np.ndarray

    def vectorise(self, **tfidf_kwargs):
        params = dict(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            sublinear_tf=True,
        )
        params.update(tfidf_kwargs)
        vec = TfidfVectorizer(**params)
        Xtr = vec.fit_transform(self.X_train_text)
        Xte = vec.transform(self.X_test_text)
        return csr_matrix(Xtr), csr_matrix(Xte), vec


def _prep_frames(tr, te):
    def prep(df):
        text = (df["title"].fillna("") + " " + df["description"].fillna(""))
        return [clean_text(t) for t in text], (df["label"].to_numpy() - 1)
    Xtr, ytr = prep(tr)
    Xte, yte = prep(te)
    return Dataset(Xtr, Xte, ytr, yte)


def _load_local_csv() -> Dataset:
    """Kaggle-friendly: read train.csv/test.csv from data/ if present."""
    cols = ["label", "title", "description"]
    tr_path = config.DATA_DIR / "train.csv"
    te_path = config.DATA_DIR / "test.csv"
    if not (tr_path.exists() and te_path.exists()):
        raise FileNotFoundError("no local train.csv/test.csv in data/")
    tr = pd.read_csv(tr_path, header=None, names=cols)
    te = pd.read_csv(te_path, header=None, names=cols)
    return _prep_frames(tr, te)


def _load_from_hf() -> Dataset:
    from datasets import load_dataset
    ds = load_dataset(config.DATASET_NAME, cache_dir=str(config.DATA_DIR))
    tr, te = ds["train"], ds["test"]
    return Dataset(
        X_train_text=[clean_text(t) for t in tr["text"]],
        X_test_text=[clean_text(t) for t in te["text"]],
        y_train=np.array(tr["label"]),
        y_test=np.array(te["label"]),
    )


def _load_from_csv() -> Dataset:
    base = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
    cols = ["label", "title", "description"]
    tr = pd.read_csv(base + "train.csv", header=None, names=cols)
    te = pd.read_csv(base + "test.csv", header=None, names=cols)
    return _prep_frames(tr, te)


def make_synthetic(n_per_class: int = 200, seed: int = config.SEED) -> Dataset:
    rng = np.random.default_rng(seed)
    vocab = {
        0: ["nation", "government", "border", "election", "treaty", "conflict"],
        1: ["match", "team", "score", "player", "league", "tournament"],
        2: ["market", "stocks", "profit", "revenue", "trade", "economy"],
        3: ["software", "chip", "research", "space", "device", "algorithm"],
    }
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
    """prefer: auto (local->hf->csv), local, hf, csv, synthetic."""
    if prefer == "synthetic":
        return make_synthetic()
    if prefer == "local":
        return _load_local_csv()
    if prefer in ("auto", "hf", "csv"):
        if prefer in ("auto",):
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
    raise ValueError(f"unknown prefer={prefer}")


if __name__ == "__main__":
    from .sparsity import analyse, pretty
    ds = load()
    print(f"train={len(ds.X_train_text)}  test={len(ds.X_test_text)}")
    Xtr, Xte, vec = ds.vectorise()
    print(f"TF-IDF shape: train={Xtr.shape}  test={Xte.shape}")
    print(pretty(analyse(Xtr)))
