"""Aggregate the multi-seed runs into mean +/- standard deviation.

Produces:
  1. A summary table (mean, std, min, max, n) for every configuration, written
     to results/multiseed_summary.csv for direct use in the report.
  2. A report-ready "mean +/- std" text column.
  3. A paired comparison of the feature selectors, which is the specific
     question the multi-seed run exists to answer: are chi-square and mutual
     information actually different, or is the gap inside run-to-run noise?

The selector comparison is done as a PAIRED test across seeds, because both
selectors see exactly the same seed and data split, so pairing removes the
between-seed variance and is the appropriate way to compare them.

Usage from the repo root:
    python scripts/summarise_multiseed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np
import pandas as pd

from src import config

# Columns that together identify one configuration (ignoring the seed).
CONFIG_KEYS = ["model", "track", "variant", "selector", "keep_fraction",
               "weight_sparsity", "bits"]
METRICS = ["macro_f1", "accuracy", "latency_ms", "model_size_kb",
           "weight_density", "efficiency_index"]


def load() -> pd.DataFrame:
    if not config.MULTISEED_CSV.exists():
        raise SystemExit(
            f"No multi-seed results at {config.MULTISEED_CSV}.\n"
            "Run: python scripts/run_multiseed.py")
    df = pd.read_csv(config.MULTISEED_CSV)
    for c in METRICS + ["seed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in CONFIG_KEYS if k in df.columns]
    metrics = [m for m in METRICS if m in df.columns]
    g = df.groupby(keys, dropna=False)[metrics]
    out = g.agg(["mean", "std", "min", "max", "count"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out = out.reset_index()

    # Report-ready text, e.g. "0.9059 +/- 0.0012"
    for m in ("macro_f1", "accuracy"):
        if f"{m}_mean" in out.columns:
            out[f"{m}_report"] = out.apply(
                lambda r, m=m: f"{r[f'{m}_mean']:.4f} +/- {r[f'{m}_std']:.4f}"
                if pd.notna(r[f"{m}_std"]) else f"{r[f'{m}_mean']:.4f}", axis=1)
    return out


def paired_selector_test(df: pd.DataFrame):
    """Compare selectors pairwise on macro-F1, paired by (model, budget, seed)."""
    fp = df[df.get("variant") == "feature_prune"].copy()
    if fp.empty or "selector" not in fp.columns:
        print("No feature-pruning rows to compare.")
        return

    print("\n" + "=" * 74)
    print("SELECTOR COMPARISON (paired across seeds)")
    print("=" * 74)

    wide = fp.pivot_table(index=["model", "keep_fraction", "seed"],
                          columns="selector", values="macro_f1")
    selectors = [c for c in ("chi2", "l1", "mutual_info") if c in wide.columns]

    # Per-budget means, which is what the report table needs.
    means = fp.pivot_table(index=["model", "keep_fraction"],
                           columns="selector", values="macro_f1",
                           aggfunc=["mean", "std"])
    print("\nMean macro-F1 by selector and vocabulary budget:")
    print(means.round(4).to_string())

    print("\nPairwise differences over all paired observations:")
    print(f"{'comparison':<26} {'mean diff':>10} {'std':>9} {'n':>4} {'wins':>7}  verdict")
    for i, a in enumerate(selectors):
        for b in selectors[i + 1:]:
            sub = wide[[a, b]].dropna()
            if sub.empty:
                continue
            d = sub[a] - sub[b]
            n = len(d)
            mean_d, sd = d.mean(), d.std(ddof=1)
            wins = int((d > 0).sum())

            # Paired t-statistic. Reported alongside the effect size rather than
            # as a bare significance claim, given the small number of seeds.
            if sd and sd > 0 and n > 1:
                t = mean_d / (sd / np.sqrt(n))
                # Two-sided critical value at ~95% for the relevant df.
                crit = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
                        6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}.get(n - 1, 2.0)
                verdict = ("distinguishable" if abs(t) > crit
                           else "within noise")
                tinfo = f"t={t:+.2f}"
            else:
                verdict, tinfo = "identical", "t=n/a"

            print(f"{a + ' - ' + b:<26} {mean_d:>+10.4f} {sd:>9.4f} {n:>4} "
                  f"{wins:>3}/{n:<3}  {verdict} ({tinfo})")

    print("\nReading: a positive mean difference favours the first selector.")
    print("'within noise' means the gap is smaller than seed-to-seed variation,")
    print("so the two selectors should be reported as comparable rather than ranked.")


def stability_report(df: pd.DataFrame):
    """Which results are stable across seeds, and which are not."""
    keys = [k for k in CONFIG_KEYS if k in df.columns]
    g = df.groupby(keys, dropna=False)["macro_f1"].agg(["mean", "std", "count"])
    g = g[g["count"] > 1].dropna(subset=["std"]).sort_values("std", ascending=False)
    if g.empty:
        return
    print("\n" + "=" * 74)
    print("STABILITY: configurations with the largest seed-to-seed variation")
    print("=" * 74)
    print(g.head(8).round(4).to_string())
    print(f"\nMedian standard deviation across all configurations: "
          f"{g['std'].median():.4f}")
    print("Differences smaller than roughly twice this value should not be")
    print("treated as meaningful.")


def main():
    df = load()
    seeds = sorted(df["seed"].dropna().unique().tolist()) if "seed" in df else []
    print(f"Loaded {len(df)} runs across seeds {seeds}")

    out = summarise(df)
    out.to_csv(config.MULTISEED_SUMMARY, index=False)

    cols = [c for c in ["model", "track", "variant", "selector", "keep_fraction",
                        "weight_sparsity", "bits", "macro_f1_report",
                        "model_size_kb_mean"] if c in out.columns]
    print("\n" + "=" * 74)
    print("SUMMARY (mean +/- std across seeds)")
    print("=" * 74)
    print(out[cols].to_string(index=False))

    paired_selector_test(df)
    stability_report(df)

    print(f"\nFull summary written -> {config.MULTISEED_SUMMARY}")


if __name__ == "__main__":
    main()
