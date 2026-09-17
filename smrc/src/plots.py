"""Report-ready figures from results/results.csv."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from . import config


def _load() -> pd.DataFrame:
    if not config.RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"No results at {config.RESULTS_CSV}. Run the experiment scripts first.")
    df = pd.read_csv(config.RESULTS_CSV)
    for col in ("accuracy", "macro_f1", "latency_ms", "model_size_kb",
                "weight_sparsity", "keep_fraction", "bits", "efficiency_index"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _save(fig, name: str):
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_weight_sparsity(df):
    sub = df[df.get("variant") == "weight_prune"] if "variant" in df else df.iloc[0:0]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in sub.groupby("model"):
        g = g.sort_values("weight_sparsity")
        ax.plot(g["weight_sparsity"], g["accuracy"], marker="o", label=model)
    ax.set_xlabel("Weight sparsity (fraction zeroed)"); ax.set_ylabel("Test accuracy")
    ax.set_title("Track A - accuracy vs weight sparsity")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, "trackA_weight_sparsity.png")


def plot_feature_pruning(df):
    sub = df[df.get("variant") == "feature_prune"] if "variant" in df else df.iloc[0:0]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (model, sel), g in sub.groupby(["model", "selector"]):
        g = g.sort_values("keep_fraction")
        ax.plot(g["keep_fraction"], g["accuracy"], marker="o", label=f"{model}/{sel}")
    ax.set_xlabel("Fraction of vocabulary kept"); ax.set_ylabel("Test accuracy")
    ax.set_xscale("log"); ax.set_title("Track A - accuracy vs feature pruning")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, "trackA_feature_pruning.png")


def plot_quantisation(df):
    sub = df[df.get("track") == "B"] if "track" in df else df.iloc[0:0]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in sub.groupby("model"):
        g = g.sort_values("bits")
        ax.plot(g["bits"], g["accuracy"], marker="o", label=model)
    ax.set_xlabel("Weight precision (bits)"); ax.set_ylabel("Test accuracy")
    ax.set_title("Track B - accuracy vs quantisation precision")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, "trackB_quantisation.png")


def plot_pareto(df):
    if not {"accuracy", "model_size_kb"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for track, g in df.groupby(df.get("track", "baseline").fillna("baseline")):
        ax.scatter(g["model_size_kb"], g["accuracy"], label=str(track), alpha=0.7)
    ax.set_xlabel("Model size (KB)"); ax.set_ylabel("Test accuracy")
    ax.set_xscale("log"); ax.set_title("Accuracy vs model size")
    ax.grid(True, alpha=0.3); ax.legend(title="track")
    _save(fig, "pareto_accuracy_vs_size.png")


def plot_efficiency(df):
    if "efficiency_index" not in df.columns:
        return
    sub = df.dropna(subset=["efficiency_index"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values("efficiency_index", ascending=False).head(12)
    labels = sub.get("variant", sub.get("model")).astype(str) + " / " + sub["model"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels[::-1], sub["efficiency_index"][::-1])
    ax.set_xlabel("Efficiency index (accuracy / mem x latency)")
    ax.set_title("Top configurations by efficiency index")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, "efficiency_index_top.png")


def make_all():
    df = _load()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_weight_sparsity(df)
    plot_feature_pruning(df)
    plot_quantisation(df)
    plot_pareto(df)
    plot_efficiency(df)
    print("Done. Figures in", config.FIGURES_DIR)


if __name__ == "__main__":
    make_all()
