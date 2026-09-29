"""Summarise the multi-seed runs and compare the feature selectors.

Writes the mean, standard deviation, minimum and maximum macro-F1 of every
repeated configuration, then compares the selectors as paired observations.
Two selectors are paired when they share a model, a vocabulary budget and a
seed, so each comparison is made under identical conditions.
"""
import _common  # noqa: F401  (must come first)

import numpy as np
import pandas as pd

from src import config

KEYS = ["model", "track", "variant", "selector", "keep_fraction", "weight_sparsity", "bits"]


def load() -> pd.DataFrame:
    if not config.MULTISEED_CSV.exists():
        raise SystemExit(f"No multi-seed results at {config.MULTISEED_CSV}. "
                         "Run scripts/run_multiseed.py first.")
    df = pd.read_csv(config.MULTISEED_CSV)
    for col in ["seed", "macro_f1", "keep_fraction", "weight_sparsity", "bits"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in KEYS if k in df.columns]
    out = (df.groupby(keys, dropna=False)["macro_f1"]
             .agg(["mean", "std", "min", "max", "count"])
             .reset_index()
             .rename(columns=lambda c: f"macro_f1_{c}" if c in
                     ("mean", "std", "min", "max", "count") else c))
    out["macro_f1_report"] = [f"{m:.4f} +/- {s:.4f}"
                              for m, s in zip(out["macro_f1_mean"], out["macro_f1_std"].fillna(0))]
    return out


def compare_selectors(df: pd.DataFrame) -> None:
    fp = df[df["variant"] == "feature_prune"]
    if fp.empty:
        return
    wide = fp.pivot_table(index=["model", "keep_fraction", "seed"],
                          columns="selector", values="macro_f1")
    names = [s for s in ("l1", "chi2", "mutual_info") if s in wide.columns]

    print("\nPaired selector comparison (positive gap favours the first selector)")
    print(f"{'comparison':<22}{'mean gap':>10}{'wins':>9}{'ties':>6}{'n':>5}{'t':>8}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = (wide[a] - wide[b]).dropna()
            n = len(d)
            t = d.mean() / (d.std(ddof=1) / np.sqrt(n)) if n > 1 and d.std(ddof=1) > 0 else float("nan")
            print(f"{a + ' vs ' + b:<22}{d.mean() * 100:>+9.2f}p"
                  f"{int((d > 0).sum()):>6}/{n:<2}{int((d == 0).sum()):>6}{n:>5}{t:>8.2f}")
    print("Gaps are in macro-F1 points. Selectors with zero variance across seeds (such as")
    print("chi-square) put all the spread on the other selector, so read the win counts as")
    print("the main evidence rather than the t value.")


def main():
    _common.parse_args(__doc__)
    df = load()
    print(f"Loaded {len(df)} runs over seeds {sorted(df['seed'].dropna().astype(int).unique())}")

    summary = summarise(df)
    summary.to_csv(config.MULTISEED_SUMMARY, index=False)
    cols = [c for c in KEYS + ["macro_f1_report"] if c in summary.columns]
    print(summary[cols].to_string(index=False))

    identical = int((summary["macro_f1_std"].fillna(0) == 0).sum())
    print(f"\n{identical} of {len(summary)} configurations gave the same macro-F1 under every seed.")

    compare_selectors(df)
    print(f"\nSummary written to {config.MULTISEED_SUMMARY}")


if __name__ == "__main__":
    main()
