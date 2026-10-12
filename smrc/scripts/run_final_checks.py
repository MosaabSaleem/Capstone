"""Final checks promised in Status Check 3.

  latency    Batch, warm single-document and cold single-document latency on
             the pruned models, not just the baselines. Tests whether feature
             pruning keeps its speed advantage outside batch processing.

  endtoend   Raw text in, prediction out. Includes the TF-IDF vectoriser, which
             a real deployment also has to load and run. Compares the full
             20,000-term vocabulary with a vocabulary cut down by L1 selection.

  weights    Why does the linear SVM survive weight pruning better than
             Logistic Regression? Measures how concentrated each model's weight
             is, at two regularisation strengths, so that model type and
             regularisation strength can be separated.

  combined   Feature pruning followed by weight pruning, to see whether the two
             combine the speed benefit of one with the storage benefit of the
             other.

  index      Final efficiency index. Recomputes the original index, an
             accuracy-gated version and a scale-normalised version from
             results.csv, and checks how sensitive the ranking is to the
             accuracy penalty.

Results go to results/final_checks.csv (the index stage also writes
results/final_index.csv). results.csv is only read, never modified. Running a
stage replaces any earlier rows for that stage.

    python scripts/run_final_checks.py              all stages
    python scripts/run_final_checks.py weights      one stage only
    python scripts/run_final_checks.py --quick      reduced run for checking
"""
import _common  # noqa: F401  (must come first)

import copy
import csv
import gc
import os
import pickle
import statistics
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from src import config, data
from src.baselines import classifier, fit_tuned, get_weights
from src.pruning import prune_weights, with_weights
from src.utils import log_result, set_seed

OUT = config.RESULTS_DIR / "final_checks.csv"
INDEX_OUT = config.RESULTS_DIR / "final_index.csv"

WARM_TRIALS = 50 if config.QUICK else 500
COLD_TRIALS = 5 if config.QUICK else 30
FEATURE_KEEPS = [0.25, 0.1] if config.QUICK else [0.25, 0.1, 0.05]
COMBINED_SPARSITIES = [0.0, 0.8, 0.9] if config.QUICK else [0.0, 0.5, 0.8, 0.9, 0.95]
LINEAR_MODELS = ("logreg", "linear_svm")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _log(row):
    log_result(row, OUT)
    print("  " + " | ".join(f"{k}={v}" for k, v in row.items()))


def drop_stage(stage: str) -> None:
    """Remove earlier rows for a stage so a re-run replaces them."""
    if not OUT.exists():
        return
    with OUT.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        kept = [row for row in reader if row.get("stage") != stage]
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)


def macro_f1(model, X, y) -> float:
    return round(float(f1_score(y, model.predict(X), average="macro")), 4)


def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


_EVICT = np.ones(64 * 1024 * 1024 // 8)  # 64 MB, larger than any CPU cache


def evict_caches() -> None:
    _EVICT.sum()


def latency_profile(predict_one, predict_batch, n_batch, load_fresh) -> dict:
    """Batch per document, warm single document and cold single document, in ms.

    predict_one(i) predicts document i on an already loaded model.
    load_fresh() loads a fresh copy of the model and returns a function that
    predicts document i with it. The copy is loaded first, then the caches are
    evicted, and only the prediction is timed, matching Section 4.6.3.
    """
    predict_batch()  # warm-up
    batch = min(_time(predict_batch) for _ in range(config.LATENCY_REPEATS))
    predict_one(0)
    warm = [_time(lambda: predict_one(0)) * 1000 for _ in range(WARM_TRIALS)]
    cold = []
    for i in range(COLD_TRIALS):
        fresh = load_fresh()
        gc.collect()
        evict_caches()
        cold.append(_time(lambda: fresh(i + 1)) * 1000)
    return {"batch_ms_per_doc": round(batch / n_batch * 1000, 5),
            "warm_single_ms": round(statistics.median(warm), 4),
            "cold_single_ms": round(statistics.median(cold), 4)}


def disk_kb(obj) -> float:
    """Smallest on-disk size of a fitted object, in KB.

    Linear models are tried both as compressed joblib and as a sparse archive
    of their weights, since Section 4.6.1 showed neither format always wins.
    Anything else is saved as compressed joblib.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "obj.joblib"
        joblib.dump(obj, path, compress=3)
        sizes = [os.path.getsize(path)]
        clf = classifier(obj) if hasattr(obj, "predict") else obj
        if hasattr(clf, "coef_"):
            w = sparse.csr_matrix(clf.coef_)
            npz = Path(tmp) / "obj.npz"
            np.savez_compressed(npz, data=w.data, indices=w.indices, indptr=w.indptr,
                                shape=np.array(w.shape), intercept=clf.intercept_,
                                classes=clf.classes_)
            sizes.append(os.path.getsize(npz))
    return round(min(sizes) / 1024, 2)


def l1_selector(keep: float, n_features: int, seed: int = config.SEED):
    """The same L1-embedded selector used in the main sweep."""
    k = max(1, round(keep * n_features))
    base = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=seed)
    return SelectFromModel(base, max_features=k, threshold=-np.inf)


_selector_cache = {}


def fitted_selector(keep, X_train, y_train):
    """Fit each L1 selector once and reuse it across stages."""
    if keep not in _selector_cache:
        _selector_cache[keep] = l1_selector(keep, X_train.shape[1]).fit(X_train, y_train)
    return _selector_cache[keep]


# ---------------------------------------------------------------------------
# Stage 1: latency on the pruned models
# ---------------------------------------------------------------------------
def latency(ds, X_train, X_test, vec):
    print("\nLatency on pruned models (model only, input already vectorised)")
    block = config.LATENCY_SAMPLE_SIZE

    for key in LINEAR_MODELS:
        base, params = fit_tuned(key, X_train, ds.y_train)
        configs = [("baseline", base, X_test)]
        configs.append(("weight_prune_95", with_weights(base, prune_weights(get_weights(base), 0.95)), X_test))
        for keep in FEATURE_KEEPS:
            sel = fitted_selector(keep, X_train, ds.y_train)
            model, _ = fit_tuned(key, sel.transform(X_train), ds.y_train)
            configs.append((f"l1_keep_{keep}", model, sel.transform(X_test)))

        for name, model, X in configs:
            blob = pickle.dumps(model)
            Xb = X[:block]
            profile = latency_profile(
                predict_one=lambda i: model.predict(X[i:i + 1]),
                predict_batch=lambda: model.predict(Xb),
                n_batch=Xb.shape[0],
                load_fresh=lambda: (lambda m: (lambda i: m.predict(X[i:i + 1])))(pickle.loads(blob)),
            )
            _log({"stage": "latency", "model": key, "config": name, "n_features": X.shape[1],
                  "macro_f1": macro_f1(model, X, ds.y_test), "disk_kb": disk_kb(model), **profile})


# ---------------------------------------------------------------------------
# Stage 2: end to end, raw text to prediction
# ---------------------------------------------------------------------------
def deployable_vectoriser(vec):
    """A copy of a fitted vectoriser with its stop_words_ attribute removed.

    scikit-learn keeps every term it discarded in stop_words_, which can be
    several megabytes. It is only used for introspection and the scikit-learn
    documentation notes it can be safely removed before saving.
    """
    v = copy.deepcopy(vec)
    if hasattr(v, "stop_words_"):
        delattr(v, "stop_words_")
    return v


def endtoend(ds, X_train, X_test, vec):
    print("\nEnd to end: raw text to prediction, including the vectoriser")
    texts = ds.X_test_text
    block = texts[:config.LATENCY_SAMPLE_SIZE]

    pipelines = []
    for key in LINEAR_MODELS:
        model, _ = fit_tuned(key, X_train, ds.y_train)
        pipelines.append((key, "full_vocabulary", deployable_vectoriser(vec), model))
        for keep in FEATURE_KEEPS:
            # Rebuild the vectoriser around only the kept terms, then refit
            # the classifier on its output. This is how a pruned vocabulary
            # would actually be deployed.
            sel = fitted_selector(keep, X_train, ds.y_train)
            terms = vec.get_feature_names_out()[sel.get_support()]
            small_vec = clone(vec).set_params(vocabulary=list(terms))
            Xs_train = small_vec.fit_transform(ds.X_train_text)
            small_model, _ = fit_tuned(key, Xs_train, ds.y_train)
            pipelines.append((key, f"l1_keep_{keep}", deployable_vectoriser(small_vec), small_model))

    for key, name, v, model in pipelines:
        blob = pickle.dumps((v, model))

        def load_fresh():
            loaded_vec, loaded_model = pickle.loads(blob)
            return lambda i: loaded_model.predict(loaded_vec.transform([texts[i]]))

        profile = latency_profile(
            predict_one=lambda i: model.predict(v.transform([texts[i]])),
            predict_batch=lambda: model.predict(v.transform(block)),
            n_batch=len(block),
            load_fresh=load_fresh,
        )
        f1 = round(float(f1_score(ds.y_test, model.predict(v.transform(texts)), average="macro")), 4)
        vec_kb, model_kb = disk_kb(v), disk_kb(model)
        _log({"stage": "endtoend", "model": key, "config": name,
              "vocabulary": len(v.vocabulary_), "macro_f1": f1,
              "vectoriser_kb": vec_kb, "model_kb": model_kb,
              "total_kb": round(vec_kb + model_kb, 2), **profile})


# ---------------------------------------------------------------------------
# Stage 3: why the SVM prunes more gracefully
# ---------------------------------------------------------------------------
def gini(values: np.ndarray) -> float:
    """0 when every weight is equal, approaching 1 when a few weights hold everything."""
    v = np.sort(np.abs(values).ravel())
    n = v.size
    if v.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


def top_share(values: np.ndarray, fraction: float) -> float:
    """Share of the total absolute weight held by the largest fraction of weights."""
    v = np.sort(np.abs(values).ravel())[::-1]
    k = max(1, round(fraction * v.size))
    return float(v[:k].sum() / v.sum())


def weights(ds, X_train, X_test, vec):
    print("\nWeight concentration and pruning robustness")
    # The tuned models use different regularisation (Logistic Regression C=1.0,
    # SVM C=0.1). Fitting both at both strengths separates the effect of the
    # model type from the effect of stronger regularisation.
    models = {}
    for c in (0.1, 1.0):
        models[("logreg", c)] = LogisticRegression(C=c, max_iter=2000)
        models[("linear_svm", c)] = LinearSVC(C=c)

    for (key, c), model in models.items():
        model.fit(X_train, ds.y_train)
        w = model.coef_
        full_pred = model.predict(X_test)
        full_f1 = round(float(f1_score(ds.y_test, full_pred, average="macro")), 4)
        pruned = with_weights(model, prune_weights(w, 0.95))
        pruned_pred = pruned.predict(X_test)
        kept = pruned.coef_
        _log({"stage": "weights", "model": key, "C": c,
              "macro_f1": full_f1,
              "macro_f1_at_95": round(float(f1_score(ds.y_test, pruned_pred, average="macro")), 4),
              "f1_drop_points": round((full_f1 - f1_score(ds.y_test, pruned_pred, average="macro")) * 100, 2),
              "agreement_at_95": round(float(np.mean(pruned_pred == full_pred)), 4),
              "top1_share": round(top_share(w, 0.01), 4),
              "top5_share": round(top_share(w, 0.05), 4),
              "top10_share": round(top_share(w, 0.10), 4),
              "gini": round(gini(w), 4),
              "l2_kept_at_95": round(float(np.linalg.norm(kept) / np.linalg.norm(w)), 4)})


# ---------------------------------------------------------------------------
# Stage 4: feature pruning followed by weight pruning
# ---------------------------------------------------------------------------
def combined(ds, X_train, X_test, vec):
    print("\nCombined feature and weight pruning")
    block = config.LATENCY_SAMPLE_SIZE
    for key in LINEAR_MODELS:
        for keep in [1.0, *FEATURE_KEEPS]:
            if keep == 1.0:
                Xtr, Xte = X_train, X_test
            else:
                sel = fitted_selector(keep, X_train, ds.y_train)
                Xtr, Xte = sel.transform(X_train), sel.transform(X_test)
            model, _ = fit_tuned(key, Xtr, ds.y_train)
            w = get_weights(model)
            Xb = Xte[:block]
            for s in COMBINED_SPARSITIES:
                m = with_weights(model, prune_weights(w, s))
                m.predict(Xb)
                batch = min(_time(lambda: m.predict(Xb)) for _ in range(config.LATENCY_REPEATS))
                _log({"stage": "combined", "model": key, "vocab_keep": keep, "weight_sparsity": s,
                      "n_features": Xte.shape[1], "macro_f1": macro_f1(m, Xte, ds.y_test),
                      "disk_kb": disk_kb(m),
                      "batch_ms_per_doc": round(batch / Xb.shape[0] * 1000, 5)})


# ---------------------------------------------------------------------------
# Stage 5: the final efficiency index
# ---------------------------------------------------------------------------
def index(ds=None, X_train=None, X_test=None, vec=None):
    print("\nFinal efficiency index")
    if not config.RESULTS_CSV.exists():
        print(f"  {config.RESULTS_CSV} not found, skipping. Restore results.csv first.")
        return
    df = pd.read_csv(config.RESULTS_CSV)
    for col in ("macro_f1", "latency_ms", "model_size_kb"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["macro_f1", "latency_ms", "model_size_kb"]).copy()

    base = df[(df["model"] == "logreg") & (df["variant"] == "tuned")].iloc[0]
    df["setting"] = (df.get("selector", pd.Series("", index=df.index)).fillna("").astype(str) + " "
                     + df.get("keep_fraction", pd.Series("", index=df.index)).fillna("").astype(str) + " "
                     + df.get("weight_sparsity", pd.Series("", index=df.index)).fillna("").astype(str) + " "
                     + df.get("bits", pd.Series("", index=df.index)).fillna("").astype(str)).str.split().str.join(" ")

    df["index_original"] = df["macro_f1"] / ((df["model_size_kb"] / 1024) * df["latency_ms"])
    df["size_halvings"] = np.log2(base["model_size_kb"] / df["model_size_kb"])
    df["latency_halvings"] = np.log2(base["latency_ms"] / df["latency_ms"])
    df["points_lost"] = (base["macro_f1"] - df["macro_f1"]) * 100
    for penalty in (0.5, 1.0, 2.0):
        df[f"score_penalty_{penalty}"] = df["size_halvings"] + df["latency_halvings"] - penalty * df["points_lost"]
    best = df["macro_f1"].max()
    df["index_gated"] = np.where(df["macro_f1"] >= best - 0.01, df["index_original"], np.nan)

    show = ["model", "variant", "setting", "macro_f1", "model_size_kb", "latency_ms"]
    for col in ("index_original", "index_gated", "score_penalty_1.0"):
        print(f"\n  Top 5 by {col}")
        print(df.dropna(subset=[col]).nlargest(5, col)[show + [col]].round(4).to_string(index=False))

    print("\n  Sensitivity of the normalised score to the accuracy penalty (top 3 each)")
    for penalty in (0.5, 1.0, 2.0):
        top = df.nlargest(3, f"score_penalty_{penalty}")
        names = "; ".join(f"{r.model} {r.variant} {r.setting}".strip() for r in top.itertuples())
        print(f"    penalty {penalty}: {names}")

    df.to_csv(INDEX_OUT, index=False)
    print(f"\n  Written to {INDEX_OUT}")


STAGES = {"latency": latency, "endtoend": endtoend, "weights": weights,
          "combined": combined, "index": index}


def main():
    args = _common.parse_args(__doc__, stages=list(STAGES))
    set_seed(config.SEED)
    stages = list(STAGES) if args.stage == "all" else [args.stage]

    needs_data = any(s != "index" for s in stages)
    ds = X_train = X_test = vec = None
    if needs_data:
        ds = data.load(args.data)
        X_train, X_test, vec = ds.vectorise()

    for name in stages:
        if name != "index":
            drop_stage(name)
        start = time.perf_counter()
        STAGES[name](ds, X_train, X_test, vec)
        print(f"  ({name} took {(time.perf_counter() - start) / 60:.1f} min)")
    print(f"\nLogged to {OUT}")


if __name__ == "__main__":
    main()
