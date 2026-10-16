"""Final analysis: index, Pareto frontier, multi-seed summary, figures, manifest.

Reads only the CSVs written by the experiment stages, so it can be re-run on its
own at any time with: python scripts/run_all.py --only analysis
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .context import read_table

MAIN_TABLES = ["baselines", "feature_selection", "docfreq", "weight_pruning", "quantisation"]
PACKAGES = ["scikit-learn", "numpy", "scipy", "pandas", "matplotlib", "psutil", "joblib", "datasets"]


def _save_fig(fig, name):
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def all_configs() -> pd.DataFrame:
    frames = [t for t in (read_table(n) for n in MAIN_TABLES) if t is not None]
    if not frames:
        raise SystemExit("No results found. Run the experiment stages first.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df.to_csv(config.RESULTS_DIR / "all_configs.csv", index=False)
    return df


def efficiency_tables(df: pd.DataFrame) -> pd.DataFrame:
    base = df[(df["model"] == "logreg") & (df["variant"] == "tuned")].iloc[0]
    out = df[["model", "track", "variant", "setting", "macro_f1", "model_size_kb",
              "latency_ms", "efficiency_index"]].copy()
    out["size_halvings"] = np.log2(base["model_size_kb"] / out["model_size_kb"])
    out["latency_halvings"] = np.log2(base["latency_ms"] / out["latency_ms"])
    out["points_lost"] = (base["macro_f1"] - out["macro_f1"]) * 100
    for p in config.INDEX_PENALTIES:
        out[f"score_penalty_{p:g}"] = (out["size_halvings"] + out["latency_halvings"]
                                       - p * out["points_lost"])
    out["final_index"] = out[f"score_penalty_{config.INDEX_PENALTY:g}"]
    gate = out["macro_f1"] >= out["macro_f1"].max() - config.INDEX_GATE_POINTS / 100
    out["gated_index"] = np.where(gate, out["efficiency_index"], np.nan)
    out = out.round(4).sort_values("final_index", ascending=False)
    out.to_csv(config.RESULTS_DIR / "index.csv", index=False)

    show = ["model", "variant", "setting", "macro_f1", "model_size_kb", "latency_ms"]
    for col in ("efficiency_index", "gated_index", "final_index"):
        print(f"\n  Top 5 by {col}")
        print(out.dropna(subset=[col]).nlargest(5, col)[show + [col]].to_string(index=False))
    print("\n  Top 3 under each accuracy penalty")
    for p in config.INDEX_PENALTIES:
        top = out.nlargest(3, f"score_penalty_{p:g}")
        print(f"    penalty {p:g}: " + "; ".join(f"{r.model} {r.setting}" for r in top.itertuples()))
    return out


def pareto(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.dropna(subset=["macro_f1", "model_size_kb"]).sort_values(
        ["macro_f1", "model_size_kb"], ascending=[False, True])
    front, cheapest = [], np.inf
    for _, row in ordered.iterrows():
        if row["model_size_kb"] < cheapest:
            front.append(row)
            cheapest = row["model_size_kb"]
    front = pd.DataFrame(front)
    front.to_csv(config.RESULTS_DIR / "pareto.csv", index=False)
    print("\n  Pareto frontier (macro-F1 against deployable size)")
    print(front[["model", "variant", "setting", "macro_f1", "model_size_kb"]].to_string(index=False))
    return front


def multiseed_summary():
    ms = read_table("multiseed")
    if ms is None:
        return
    keys = [k for k in ("model", "variant", "selector", "keep_fraction", "weight_sparsity", "bits")
            if k in ms.columns]
    summary = (ms.groupby(keys, dropna=False)["macro_f1"]
                 .agg(["mean", "std", "min", "max", "count"]).reset_index().round(4))
    summary.to_csv(config.RESULTS_DIR / "multiseed_summary.csv", index=False)
    same = int((summary["std"].fillna(0) == 0).sum())
    print(f"\n  Multi-seed: {same} of {len(summary)} configurations identical across seeds")

    fp = ms[ms["variant"] == "feature_prune"]
    wide = fp.pivot_table(index=["model", "keep_fraction", "seed"], columns="selector",
                          values="macro_f1")
    rows = []
    names = [s for s in ("l1", "chi2", "mutual_info") if s in wide.columns]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for model in ["all", *sorted(fp["model"].unique())]:
                sub = wide if model == "all" else wide.xs(model, level="model", drop_level=False)
                d = (sub[a] - sub[b]).dropna()
                rows.append({"comparison": f"{a} vs {b}", "model": model, "n": len(d),
                             "mean_gap_points": round(d.mean() * 100, 3),
                             "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                             "losses": int((d < 0).sum())})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(config.RESULTS_DIR / "selector_pairs.csv", index=False)
    print("\n  Paired selector comparison (positive gap favours the first selector)")
    print(pairs[pairs["model"] == "all"].to_string(index=False))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figures(df: pd.DataFrame, idx: pd.DataFrame):
    fs = read_table("feature_selection")
    if fs is not None:
        models = list(dict.fromkeys(fs["model"]))
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
        for ax, model in zip(np.atleast_1d(axes), models):
            for sel, g in fs[fs["model"] == model].groupby("selector"):
                g = g.sort_values("keep_fraction")
                ax.plot(g["keep_fraction"], g["macro_f1"], marker="o", label=sel)
            ax.set(xscale="log", title=model, xlabel="Fraction of vocabulary kept")
            ax.grid(True, alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("Macro-F1")
        np.atleast_1d(axes)[0].legend(fontsize=8)
        fig.tight_layout()
        fig.suptitle("Track A: feature selection", y=1.04)
        _save_fig(fig, "trackA_feature_pruning.png")

    wp = read_table("weight_pruning")
    if wp is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, g in wp.groupby("model"):
            ax.plot(g["weight_sparsity"], g["macro_f1"], marker="o", label=model)
        ax.set(xlabel="Weight sparsity (fraction zeroed)", ylabel="Macro-F1",
               title="Track A: macro-F1 against weight sparsity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save_fig(fig, "trackA_weight_sparsity.png")

    dk = read_table("disk_size")
    if dk is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (model, fmt), g in dk.groupby(["model", "format"]):
            ax.plot(g["weight_sparsity"], g["file_kb"], marker="o", label=f"{model} {fmt}",
                    linestyle="-" if model == "linear_svm" else "--")
        ax.set(yscale="log", xlabel="Weight sparsity (fraction zeroed)",
               ylabel="File size on disk (KB)", title="On-disk size against weight sparsity")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        _save_fig(fig, "trackA_weight_sparsity_size.png")

    qt = read_table("quantisation")
    if qt is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, g in qt.groupby("model"):
            g = g.sort_values("bits")
            ax.plot(g["bits"], g["macro_f1"], marker="o", label=model)
        ax.set(xlabel="Weight precision (bits)", ylabel="Macro-F1",
               title="Track B: macro-F1 against quantisation precision")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save_fig(fig, "trackB_quantisation.png")

    nb = read_table("nb_grids")
    if nb is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for scheme, g in nb[nb["scheme"] != "none"].groupby("scheme"):
            g = g.sort_values("bits")
            ax.plot(g["bits"], g["agreement"] * 100, marker="o", label=scheme)
        ax.set(xlabel="Bits", ylabel="Agreement with full precision (%)",
               title="Naive Bayes: quantisation grids")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save_fig(fig, "trackB_nb_grids.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for track, g in df.groupby(df["track"].fillna("baseline")):
        ax.scatter(g["model_size_kb"], g["macro_f1"], label=str(track), alpha=0.7)
    ax.set(xscale="log", xlabel="Deployable model size (KB)", ylabel="Macro-F1",
           title="Macro-F1 against model size")
    ax.grid(True, alpha=0.3)
    ax.legend(title="track")
    _save_fig(fig, "pareto_accuracy_vs_size.png")

    for col, name, title in (("efficiency_index", "efficiency_index_top.png",
                              "Original index: accuracy / (MB x ms)"),
                             ("final_index", "efficiency_final_top.png",
                              f"Final index: halvings saved minus {config.INDEX_PENALTY:g} per F1 point lost")):
        top = idx.dropna(subset=[col]).nlargest(12, col)
        labels = top["model"] + " " + top["setting"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(labels[::-1], top[col][::-1])
        for i, f1 in enumerate(top["macro_f1"][::-1]):
            ax.text(ax.get_xlim()[1] * 0.01, i, f"F1 {f1:.3f}", va="center", fontsize=7, color="white")
        ax.set(title=title)
        ax.grid(True, axis="x", alpha=0.3)
        _save_fig(fig, name)

    e2e = read_table("end_to_end")
    if e2e is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = e2e["model"] + " " + e2e["config"]
        ax.barh(labels, e2e["vectoriser_kb"], label="vectoriser")
        ax.barh(labels, e2e["model_kb"], left=e2e["vectoriser_kb"], label="model")
        for i, (tot, f1) in enumerate(zip(e2e["total_kb"], e2e["macro_f1"])):
            ax.text(tot, i, f" {tot:.0f} KB, F1 {f1:.3f}", va="center", fontsize=7)
        ax.set(xlabel="Size on disk (KB)", title="End to end: what actually has to be shipped")
        ax.invert_yaxis()
        ax.set_xlim(0, e2e["total_kb"].max() * 1.35)
        ax.legend(loc="lower right")
        _save_fig(fig, "end_to_end_size.png")

    lc, lz = read_table("lorenz_curves"), read_table("lorenz")
    if lc is not None and lz is not None:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for (model, c), g in lc.groupby(["model", "C"]):
            info = lz[(lz["model"] == model) & (lz["C"] == c)].iloc[0]
            ax.plot(g["fraction"], g["share"], "-" if model == "linear_svm" else "--",
                    label=f"{model} C={c:g}: top 5% hold {info['share_top5']:.0%}, "
                          f"drop {info['f1_drop_points']} pts")
        ax.axvline(0.05, color="grey", linewidth=0.8, linestyle=":")
        ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1),
               xlabel="Fraction of weights, largest first (log scale)",
               ylabel="Share of total absolute weight",
               title="Weight concentration against robustness to 95% pruning")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")
        _save_fig(fig, "weight_lorenz.png")


# ---------------------------------------------------------------------------
# Checks and manifest
# ---------------------------------------------------------------------------
def sanity_check(source: str) -> dict:
    b = read_table("baselines")
    if b is None:
        return {"status": "no baselines"}
    got = dict(zip(b["model"], b["macro_f1"]))
    if config.QUICK or source == "synthetic":
        return {"status": "skipped (quick or synthetic run)", "baselines": got}
    diff = {k: round(got.get(k, np.nan) - v, 4) for k, v in config.EXPECTED_BASELINES.items()}
    ok = all(abs(d) < 1e-9 for d in diff.values())
    status = "ok" if ok else "MISMATCH"
    print(f"\n  Baseline check against the reported run: {status} {got}")
    if not ok:
        print("  WARNING: baselines differ from the reported run. Check the data and library versions.")
    return {"status": status, "baselines": got, "difference": diff}


def manifest(ctx):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    path = config.RESULTS_DIR / "manifest.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    timings = {**previous.get("stage_minutes", {}),
               **{k: round(v / 60, 2) for k, v in ctx.timings.items()}}
    tables = {p.stem: len(pd.read_csv(p)) for p in sorted(config.RESULTS_DIR.glob("*.csv"))}
    info = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_mode": config.QUICK,
        "data_source": ctx.source,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "stage_minutes": timings,
        "total_minutes": round(sum(timings.values()), 1),
        "baseline_check": sanity_check(ctx.source),
        "rows_per_table": tables,
    }
    path.write_text(json.dumps(info, indent=2))
    (config.RESULTS_DIR / "environment.txt").write_text(
        f"python {info['python']}\n" + "".join(f"{k}=={v}\n" for k, v in versions.items() if v))
    print(f"\n  wrote {path.name} and environment.txt")


def run(ctx):
    df = all_configs()
    idx = efficiency_tables(df)
    pareto(df)
    multiseed_summary()
    figures(df, idx)
    manifest(ctx)
