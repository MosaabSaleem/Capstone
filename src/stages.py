"""Every experiment in the project, one function per stage.

Each stage writes its own CSV to the results folder when it finishes, so a run
that stops part-way keeps everything completed before that point.
"""
from __future__ import annotations

import copy
import pickle

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from . import config
from .measure import (agreement, disk_sizes, evaluate, latency_profile, macro_f1,
                      median_ms, predictive, smallest_disk_kb, sparsity_report)
from .models import classifier, get_weights, tune, with_weights
from .optimise import (GRIDS, achieved_sparsity, count_parameters, packed_size_kb,
                       prune_weights, q_asymmetric, q_symmetric, quantise_model)

S = config.SEED


def _base_row(model, track, variant, setting, **extra):
    return {"seed": S, "model": model, "track": track, "variant": variant,
            "setting": setting, **extra}


# ---------------------------------------------------------------------------
# 1. Sparsity of the TF-IDF matrices
# ---------------------------------------------------------------------------
def sparsity(ctx):
    for split, X in (("train", ctx.X_train), ("test", ctx.X_test)):
        ctx.add("sparsity", {"split": split, **sparsity_report(X)})
    ctx.save("sparsity")


# ---------------------------------------------------------------------------
# 2. Tuned baselines
# ---------------------------------------------------------------------------
def baselines(ctx):
    for key in config.MODELS:
        model, params = ctx.baseline(key)
        ctx.add("baselines", _base_row(key, "baseline", "tuned", "tuned",
                                       best_params=str(params), n_features=ctx.X_train.shape[1],
                                       **evaluate(model, ctx.X_test, ctx.y_test)))
    ctx.save("baselines")


# ---------------------------------------------------------------------------
# 3. Track A: feature selection
# ---------------------------------------------------------------------------
def features(ctx):
    for method in config.SELECTORS:
        for key in config.MODELS:
            for keep in config.KEEP_FRACTIONS:
                model, params, cols = ctx.feature_model(key, method, keep)
                ctx.add("feature_selection", _base_row(
                    key, "A", "feature_prune", f"{method} {keep:g}",
                    selector=method, keep_fraction=keep, n_features=len(cols),
                    best_params=str(params),
                    **evaluate(model, ctx.X_test[:, cols], ctx.y_test)))
    ctx.save("feature_selection")


# ---------------------------------------------------------------------------
# 4. Track A: document-frequency filtering
# ---------------------------------------------------------------------------
def docfreq(ctx):
    for min_df, max_df in config.DOCFREQ_FILTERS:
        X_train, X_test, _ = ctx.ds.vectorise(min_df=min_df, max_df=max_df)
        for key in config.MODELS:
            model, params = tune(key, X_train, ctx.y_train, S)
            ctx.add("docfreq", _base_row(
                key, "A", "docfreq_filter", f"min_df {min_df} max_df {max_df}",
                min_df=min_df, max_df=max_df, n_features=X_train.shape[1],
                best_params=str(params), **evaluate(model, X_test, ctx.y_test)))
    ctx.save("docfreq")


# ---------------------------------------------------------------------------
# 5. Track A: magnitude weight pruning
# ---------------------------------------------------------------------------
def weights(ctx):
    for key in config.LINEAR_MODELS:
        base, _ = ctx.baseline(key)
        w = get_weights(base)
        for s in config.SPARSITIES:
            pruned_w = prune_weights(w, s)
            ctx.add("weight_pruning", _base_row(
                key, "A", "weight_prune", f"sparsity {s:g}",
                weight_sparsity=s, achieved_sparsity=achieved_sparsity(pruned_w),
                n_features=ctx.X_train.shape[1],
                **evaluate(with_weights(base, pruned_w), ctx.X_test, ctx.y_test)))
    ctx.save("weight_pruning")


# ---------------------------------------------------------------------------
# 6. Track B: quantisation with the symmetric grid
# ---------------------------------------------------------------------------
def quant(ctx):
    for key in config.MODELS:
        base, _ = ctx.baseline(key)
        reference = base.predict(ctx.X_test)
        n_params = count_parameters(base)
        for bits in [32, *config.QUANT_BITS]:
            model = base if bits == 32 else quantise_model(base, bits, config.QUANT_SCHEME)
            ctx.add("quantisation", _base_row(
                key, "B", "quantise", f"{bits}-bit",
                bits=bits, scheme="none" if bits == 32 else config.QUANT_SCHEME,
                n_params=n_params, packed_size_kb=packed_size_kb(n_params, bits),
                agreement=agreement(model.predict(ctx.X_test), reference),
                n_features=ctx.X_train.shape[1],
                **evaluate(model, ctx.X_test, ctx.y_test)))
    ctx.save("quantisation")


# ---------------------------------------------------------------------------
# 7. Track B: four quantisation grids for the Naive Bayes tables
# ---------------------------------------------------------------------------
def nbgrids(ctx):
    base, _ = ctx.baseline("naive_bayes")
    reference = base.predict(ctx.X_test)
    table = classifier(base).feature_log_prob_
    ctx.add("nb_grids", {"bits": 32, "scheme": "none", **predictive(ctx.y_test, reference),
                         "agreement": 1.0, "max_abs_error": 0.0, "mean_abs_error": 0.0,
                         "log_min": round(float(table.min()), 4),
                         "log_max": round(float(table.max()), 4)})
    for bits in config.NB_GRID_BITS:
        for scheme in GRIDS:
            model = quantise_model(base, bits, scheme)
            pred = model.predict(ctx.X_test)
            err = np.abs(classifier(model).feature_log_prob_ - table)
            ctx.add("nb_grids", {"bits": bits, "scheme": scheme, **predictive(ctx.y_test, pred),
                                 "agreement": agreement(pred, reference),
                                 "max_abs_error": round(float(err.max()), 4),
                                 "mean_abs_error": round(float(err.mean()), 4)})
    ctx.save("nb_grids")


# ---------------------------------------------------------------------------
# 8. Feature pruning followed by weight pruning
# ---------------------------------------------------------------------------
def combined(ctx):
    for key in config.LINEAR_MODELS:
        for keep in config.COMBINED_KEEPS:
            model, _, cols = ctx.feature_model(key, "l1", keep)
            X_test = ctx.X_test[:, cols]
            w = get_weights(model)
            for s in config.COMBINED_SPARSITIES:
                m = with_weights(model, prune_weights(w, s))
                ev = evaluate(m, X_test, ctx.y_test)
                ctx.add("combined", {"model": key, "vocab_keep": keep, "weight_sparsity": s,
                                     "n_features": len(cols), "macro_f1": ev["macro_f1"],
                                     "accuracy": ev["accuracy"],
                                     "disk_kb": smallest_disk_kb(m),
                                     "batch_ms_per_doc": ev["latency_ms"]})
    ctx.save("combined")


# ---------------------------------------------------------------------------
# 9. Real file sizes on disk after weight pruning
# ---------------------------------------------------------------------------
def disk(ctx):
    for key in config.LINEAR_MODELS:
        base, _ = ctx.baseline(key)
        w = get_weights(base)
        for s in config.SPARSITIES:
            m = with_weights(base, prune_weights(w, s))
            for fmt, (kb, same) in disk_sizes(m, ctx.X_test).items():
                ctx.add("disk_size", {"model": key, "weight_sparsity": s, "format": fmt,
                                      "file_kb": kb, "reload_identical": same})
    ctx.save("disk_size")


# ---------------------------------------------------------------------------
# 10. End to end: raw text in, prediction out, vectoriser included
# ---------------------------------------------------------------------------
def deployable_vectoriser(vec):
    """Copy of a fitted vectoriser without stop_words_.

    scikit-learn stores every discarded term in stop_words_, which can be
    megabytes. It is only used for inspection and is safe to remove before
    saving.
    """
    v = copy.deepcopy(vec)
    if hasattr(v, "stop_words_"):
        delattr(v, "stop_words_")
    return v


def endtoend(ctx):
    texts = ctx.ds.X_test_text
    block = texts[:config.LATENCY_SAMPLE_SIZE]
    full_vec = deployable_vectoriser(ctx.vec)
    names = ctx.vec.get_feature_names_out()

    pipelines = []
    for key in config.LINEAR_MODELS:
        base, _ = ctx.baseline(key)
        pipelines.append((key, "full_vocabulary", full_vec, base))
        pruned = with_weights(base, prune_weights(get_weights(base), 0.95))
        pipelines.append((key, "weight_prune_0.95", full_vec, pruned))
        for keep in config.DEPLOY_KEEPS:
            # Rebuild the vectoriser around only the kept terms and tune the
            # model on its output. TF-IDF rows are renormalised over the smaller
            # vocabulary, so scores differ slightly from the feature sweep,
            # which selects columns from the full-vocabulary matrix.
            cols = ctx.kept_columns("l1", keep)
            small_vec = clone(ctx.vec).set_params(vocabulary=list(names[cols]))
            Xs = small_vec.fit_transform(ctx.ds.X_train_text)
            model, _ = tune(key, Xs, ctx.y_train, S)
            pipelines.append((key, f"l1_keep_{keep:g}", deployable_vectoriser(small_vec), model))

    for key, name, v, model in pipelines:
        blob = pickle.dumps((v, model))

        def load_fresh(blob=blob):
            fv, fm = pickle.loads(blob)
            return lambda i: fm.predict(fv.transform([texts[i]]))

        profile = latency_profile(
            predict_one=lambda i, v=v, model=model: model.predict(v.transform([texts[i]])),
            predict_batch=lambda v=v, model=model: model.predict(v.transform(block)),
            n_batch=len(block), load_fresh=load_fresh)
        vec_kb, model_kb = smallest_disk_kb(v), smallest_disk_kb(model)
        ctx.add("end_to_end", {"model": key, "config": name, "vocabulary": len(v.vocabulary_),
                               "macro_f1": macro_f1(ctx.y_test, model.predict(v.transform(texts))),
                               "vectoriser_kb": vec_kb, "model_kb": model_kb,
                               "total_kb": round(vec_kb + model_kb, 2), **profile})
    ctx.save("end_to_end")


# ---------------------------------------------------------------------------
# 11. Single-request latency on the pruned models, split into overhead and work
# ---------------------------------------------------------------------------
def latency(ctx):
    for key in config.LINEAR_MODELS:
        base, _ = ctx.baseline(key)
        configs = [("baseline", base, ctx.X_test),
                   ("weight_prune_0.95",
                    with_weights(base, prune_weights(get_weights(base), 0.95)), ctx.X_test)]
        for keep in config.DEPLOY_KEEPS:
            model, _, cols = ctx.feature_model(key, "l1", keep)
            configs.append((f"l1_keep_{keep:g}", model, ctx.X_test[:, cols]))

        for name, model, X in configs:
            blob = pickle.dumps(model)
            Xb = X[:config.LATENCY_SAMPLE_SIZE]

            def load_fresh(blob=blob, X=X):
                fm = pickle.loads(blob)
                return lambda i: fm.predict(X[i:i + 1])

            profile = latency_profile(
                predict_one=lambda i, model=model, X=X: model.predict(X[i:i + 1]),
                predict_batch=lambda model=model, Xb=Xb: model.predict(Xb),
                n_batch=Xb.shape[0], load_fresh=load_fresh)

            # Overhead split: a full predict call, the same call on an empty
            # document (the fixed cost), and the bare arithmetic with no
            # scikit-learn code around it.
            clf = classifier(model)
            W, b = np.ascontiguousarray(clf.coef_.T), clf.intercept_
            doc, empty = X[0:1], sparse.csr_matrix((1, X.shape[1]))
            full_ms = median_ms(lambda: model.predict(doc))
            floor_ms = median_ms(lambda: model.predict(empty))
            ctx.add("latency", {"model": key, "config": name, "n_features": X.shape[1],
                                "macro_f1": macro_f1(ctx.y_test, model.predict(X)),
                                **profile,
                                "predict_call_ms": full_ms, "empty_doc_floor_ms": floor_ms,
                                "real_work_ms": round(max(0.0, full_ms - floor_ms), 5),
                                "raw_matmul_ms": median_ms(lambda: np.argmax(doc @ W + b, axis=1)),
                                "overhead_share": round(floor_ms / full_ms, 3) if full_ms else None})

    text = ctx.ds.X_test_text[0]
    ctx.add("latency", {"model": "vectoriser", "config": "full_vocabulary",
                        "predict_call_ms": median_ms(lambda: ctx.vec.transform([text]))})
    ctx.save("latency")


# ---------------------------------------------------------------------------
# 12. Repeat the headline configurations across seeds
# ---------------------------------------------------------------------------
def multiseed(ctx):
    for seed in config.SEEDS:
        print(f"  seed {seed}")
        for key in config.MODELS:
            base, params = ctx.baseline(key, seed)
            ctx.add("multiseed", {"seed": seed, "model": key, "variant": "tuned",
                                  "best_params": str(params),
                                  **predictive(ctx.y_test, base.predict(ctx.X_test))}, show=False)
            for method in config.SELECTORS:
                for keep in config.MULTISEED_KEEPS:
                    m, _, cols = ctx.feature_model(key, method, keep, seed)
                    ctx.add("multiseed", {"seed": seed, "model": key, "variant": "feature_prune",
                                          "selector": method, "keep_fraction": keep,
                                          **predictive(ctx.y_test, m.predict(ctx.X_test[:, cols]))},
                            show=False)
            for bits in config.MULTISEED_BITS:
                m = quantise_model(base, bits)
                ctx.add("multiseed", {"seed": seed, "model": key, "variant": "quantise",
                                      "bits": bits,
                                      **predictive(ctx.y_test, m.predict(ctx.X_test))}, show=False)
            if key in config.LINEAR_MODELS:
                for s in config.MULTISEED_SPARSITIES:
                    m = with_weights(base, prune_weights(get_weights(base), s))
                    ctx.add("multiseed", {"seed": seed, "model": key, "variant": "weight_prune",
                                          "weight_sparsity": s,
                                          **predictive(ctx.y_test, m.predict(ctx.X_test))},
                            show=False)
    ctx.save("multiseed")


# ---------------------------------------------------------------------------
# 13. Robustness: weight concentration and production edge cases
# ---------------------------------------------------------------------------
def _lorenz(w):
    """Cumulative share of total |weight| held by the largest weights."""
    v = np.sort(np.abs(w).ravel())[::-1]
    return np.arange(1, v.size + 1) / v.size, np.cumsum(v) / v.sum()


def robustness(ctx):
    # Weight concentration at two regularisation strengths, so the effect of
    # model type can be separated from the effect of regularisation.
    grid = np.logspace(-4, 0, 200)
    for key, make in (("logreg", lambda c: LogisticRegression(C=c, max_iter=2000)),
                      ("linear_svm", lambda c: LinearSVC(C=c))):
        for c in (0.1, 1.0):
            model = make(c).fit(ctx.X_train, ctx.y_train)
            w = model.coef_
            full = model.predict(ctx.X_test)
            pruned = with_weights(model, prune_weights(w, 0.95)).predict(ctx.X_test)
            x, cum = _lorenz(w)
            curve = np.interp(grid, x, cum)
            ctx.add("lorenz", {"model": key, "C": c,
                               "macro_f1": macro_f1(ctx.y_test, full),
                               "macro_f1_at_0.95": macro_f1(ctx.y_test, pruned),
                               "f1_drop_points": round((macro_f1(ctx.y_test, full)
                                                        - macro_f1(ctx.y_test, pruned)) * 100, 2),
                               "agreement_at_0.95": agreement(pruned, full),
                               "share_top1": round(float(np.interp(0.01, x, cum)), 4),
                               "share_top5": round(float(np.interp(0.05, x, cum)), 4),
                               "share_top10": round(float(np.interp(0.10, x, cum)), 4)})
            for gx, gy in zip(grid, curve):
                ctx.add("lorenz_curves", {"model": key, "C": c, "fraction": round(float(gx), 6),
                                          "share": round(float(gy), 5)}, show=False)
    ctx.save("lorenz")
    ctx.save("lorenz_curves")

    # Edge cases
    svm, _ = ctx.baseline("linear_svm")
    nb, _ = ctx.baseline("naive_bayes")
    small_svm, _, cols = ctx.feature_model("linear_svm", "l1", 0.1)
    X_small = ctx.X_test[:, cols]
    pruned_svm = with_weights(svm, prune_weights(get_weights(svm), 0.95))

    def edge(check, **values):
        ctx.add("edge_cases", {"check": check, **values})

    edge("empty_docs_full_vocab", value=int((ctx.X_test.getnnz(axis=1) == 0).sum()),
         of=ctx.X_test.shape[0])
    edge("empty_docs_after_l1_0.1", value=int((X_small.getnnz(axis=1) == 0).sum()),
         of=ctx.X_test.shape[0])
    unknown = ctx.vec.transform(["zzqx qwvb xxyyzz"])
    edge("unknown_words_prediction_svm", value=int(svm.predict(unknown)[0]),
         detail="all-zero input, decided by the intercept alone")
    edge("unknown_words_prediction_nb", value=int(nb.predict(unknown)[0]),
         detail="all-zero input, decided by the class priors alone")

    lengths = np.array([len(t.split()) for t in ctx.ds.X_test_text])
    edges_q = np.quantile(lengths, [0, 0.25, 0.5, 0.75, 1.0])
    for i in range(4):
        lo, hi = edges_q[i], edges_q[i + 1]
        mask = (lengths >= lo) & ((lengths <= hi) if i == 3 else (lengths < hi))
        if not mask.any():
            continue
        y = ctx.y_test[mask]
        edge(f"length_quartile_{i + 1}", words=f"{int(lo)}-{int(hi)}", n_docs=int(mask.sum()),
             f1_baseline=macro_f1(y, svm.predict(ctx.X_test[mask])),
             f1_weight_prune_0_95=macro_f1(y, pruned_svm.predict(ctx.X_test[mask])),
             f1_l1_0_1=macro_f1(y, small_svm.predict(X_small[mask])))

    rng = np.random.default_rng(S)
    outlier = rng.normal(0, 0.1, 1000)
    outlier[0] = 50.0
    arrays = {"constant": np.full(10, -1.386), "all_zero": np.zeros(10),
              "single_outlier": outlier, "contains_neg_inf": np.array([-3.0, -5.0, -np.inf, -4.0])}
    for name, arr in arrays.items():
        for qname, q in (("symmetric", q_symmetric), ("asymmetric", q_asymmetric)):
            with np.errstate(all="ignore"):
                out = q(arr, 4)
            edge(f"quantiser_{qname}_{name}", finite=bool(np.all(np.isfinite(out))),
                 share_zeroed=round(float(np.mean((out == 0) & (arr != 0))), 4))
    ctx.save("edge_cases")


STAGES = {
    "sparsity": sparsity,
    "baselines": baselines,
    "features": features,
    "docfreq": docfreq,
    "weights": weights,
    "quant": quant,
    "nbgrids": nbgrids,
    "combined": combined,
    "disk": disk,
    "endtoend": endtoend,
    "latency": latency,
    "multiseed": multiseed,
    "robustness": robustness,
}
