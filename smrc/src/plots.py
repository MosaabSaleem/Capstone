"""All figures used in the report, drawn from results.csv."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config
from .analysis import load_results


def _save(fig, name: str) -> None:
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _subset(df, variant):
    return df[df["variant"] == variant] if "variant" in df.columns else df.iloc[0:0]


def feature_pruning(df):
    sub = _subset(df, "feature_prune")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (model, selector), g in sub.groupby(["model", "selector"]):
        g = g.sort_values("keep_fraction")
        ax.plot(g["keep_fraction"], g["accuracy"], marker="o", label=f"{model}/{selector}")
    ax.set(xscale="log", xlabel="Fraction of vocabulary kept", ylabel="Test accuracy",
           title="Track A: accuracy against feature pruning")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "trackA_feature_pruning.png")


def weight_pruning(df):
    sub = _subset(df, "weight_prune")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in sub.groupby("model"):
        g = g.sort_values("weight_sparsity")
        ax.plot(g["weight_sparsity"], g["accuracy"], marker="o", label=model)
    ax.set(xlabel="Weight sparsity (fraction zeroed)", ylabel="Test accuracy",
           title="Track A: accuracy against weight sparsity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, "trackA_weight_sparsity.png")


def weight_pruning_size(df):
    sub = _subset(df, "weight_prune")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in sub.groupby("model"):
        g = g.sort_values("weight_sparsity")
        ax.plot(g["weight_sparsity"], g["model_size_kb"], marker="o", label=f"{model} (sparse)")
        ax.plot(g["weight_sparsity"], g["model_size_dense_kb"], marker="x",
                linestyle="--", alpha=0.6, label=f"{model} (dense)")
    ax.set(xlabel="Weight sparsity (fraction zeroed)", ylabel="Model size (KB)",
           title="Track A: deployable size against weight sparsity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "trackA_weight_sparsity_size.png")


def quantisation(df):
    sub = df[df["track"] == "B"] if "track" in df.columns else df.iloc[0:0]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in sub.groupby("model"):
        g = g.sort_values("bits")
        ax.plot(g["bits"], g["accuracy"], marker="o", label=model)
    ax.set(xlabel="Weight precision (bits)", ylabel="Test accuracy",
           title="Track B: accuracy against quantisation precision")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, "trackB_quantisation.png")


def pareto(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for track, g in df.groupby(df["track"].fillna("baseline")):
        ax.scatter(g["model_size_kb"], g["accuracy"], label=str(track), alpha=0.7)
    ax.set(xscale="log", xlabel="Model size (KB)", ylabel="Test accuracy",
           title="Accuracy against model size")
    ax.grid(True, alpha=0.3)
    ax.legend(title="track")
    _save(fig, "pareto_accuracy_vs_size.png")


def efficiency(df):
    best = df.dropna(subset=["efficiency_index"]).nlargest(12, "efficiency_index")
    if best.empty:
        return
    labels = best["variant"].astype(str) + " / " + best["model"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels[::-1], best["efficiency_index"][::-1])
    ax.set(xlabel="Efficiency index (accuracy / memory x latency)",
           title="Top configurations by efficiency index")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, "efficiency_index_top.png")


def main():
    df = load_results()
    for draw in (feature_pruning, weight_pruning, weight_pruning_size,
                 quantisation, pareto, efficiency):
        draw(df)
    print(f"Figures in {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
