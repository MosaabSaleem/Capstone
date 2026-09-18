"""Point 5 — F1 vs efficiency-index rankings + Pareto frontier."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config


def load_results() -> pd.DataFrame:
    if not config.RESULTS_CSV.exists():
        raise FileNotFoundError(f"No results at {config.RESULTS_CSV}.")
    df = pd.read_csv(config.RESULTS_CSV)
    for col in ("accuracy", "macro_f1", "latency_ms", "model_size_kb",
                "model_size_dense_kb", "weight_density", "peak_mem_mb",
                "efficiency_index"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pareto_front(df, benefit="macro_f1", cost="model_size_kb") -> pd.DataFrame:
    sub = df.dropna(subset=[benefit, cost]).copy()
    sub = sub.sort_values([benefit, cost], ascending=[False, True])
    front, best = [], np.inf
    for _, row in sub.iterrows():
        if row[cost] < best:
            front.append(row); best = row[cost]
    return pd.DataFrame(front)


def top_by(df, metric: str, n: int = 10) -> pd.DataFrame:
    cols = [c for c in ["model", "track", "variant", "selector", "keep_fraction",
                        "weight_sparsity", "bits", "macro_f1", "accuracy",
                        "latency_ms", "model_size_kb", "weight_density",
                        "efficiency_index"] if c in df.columns]
    return df.sort_values(metric, ascending=False)[cols].head(n)


def summarise():
    df = load_results()
    print(f"Loaded {len(df)} experiment rows.\n")
    print("=== Top 10 by macro-F1 ===")
    print(top_by(df, "macro_f1").to_string(index=False))
    if "efficiency_index" in df.columns:
        print("\n=== Top 10 by efficiency index ===")
        print(top_by(df, "efficiency_index").to_string(index=False))
    print("\n=== Pareto frontier: macro-F1 vs model size ===")
    pf = pareto_front(df, "macro_f1", "model_size_kb")
    cols = [c for c in ["model", "track", "variant", "macro_f1", "model_size_kb",
                        "weight_density", "latency_ms", "efficiency_index"]
            if c in pf.columns]
    print(pf[cols].to_string(index=False))
    out = config.RESULTS_DIR / "pareto_front.csv"
    pf.to_csv(out, index=False)
    print(f"\nPareto frontier written -> {out}")


if __name__ == "__main__":
    summarise()
