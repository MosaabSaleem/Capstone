"""Rankings and the Pareto frontier."""
from __future__ import annotations

import pandas as pd

from . import config

NUMERIC = ["accuracy", "macro_f1", "latency_ms", "model_size_kb", "model_size_dense_kb",
           "weight_density", "efficiency_index", "keep_fraction", "weight_sparsity", "bits"]
SHOW = ["model", "track", "variant", "selector", "keep_fraction", "weight_sparsity",
        "bits", "macro_f1", "latency_ms", "model_size_kb", "efficiency_index"]


def load_results() -> pd.DataFrame:
    if not config.RESULTS_CSV.exists():
        raise SystemExit(f"No results at {config.RESULTS_CSV}. Run the experiments first.")
    df = pd.read_csv(config.RESULTS_CSV)
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["macro_f1"])


def pareto_front(df: pd.DataFrame, benefit: str = "macro_f1",
                 cost: str = "model_size_kb") -> pd.DataFrame:
    """Configurations that no other configuration beats on both axes at once."""
    ordered = df.dropna(subset=[benefit, cost]).sort_values([benefit, cost],
                                                            ascending=[False, True])
    front, cheapest = [], float("inf")
    for _, row in ordered.iterrows():
        if row[cost] < cheapest:
            front.append(row)
            cheapest = row[cost]
    return pd.DataFrame(front)


def top(df: pd.DataFrame, metric: str, n: int = 10) -> pd.DataFrame:
    cols = [c for c in SHOW if c in df.columns]
    return df.sort_values(metric, ascending=False)[cols].head(n)


def main():
    df = load_results()
    print(f"Loaded {len(df)} model evaluations.\n")
    print("Top 10 by macro-F1")
    print(top(df, "macro_f1").to_string(index=False))
    print("\nTop 10 by efficiency index")
    print(top(df, "efficiency_index").to_string(index=False))

    front = pareto_front(df)
    front.to_csv(config.PARETO_CSV, index=False)
    print("\nPareto frontier (macro-F1 against deployable size)")
    print(front[[c for c in SHOW if c in front.columns]].to_string(index=False))
    print(f"\nWritten to {config.PARETO_CSV}")


if __name__ == "__main__":
    main()
