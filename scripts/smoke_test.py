"""Run the whole pipeline on a tiny synthetic dataset and check the output.

No download is needed and nothing is written to the real results folders.
Takes about a minute. If this passes, a full run should work too.

    python scripts/smoke_test.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TABLES = ["sparsity", "baselines", "feature_selection", "docfreq", "weight_pruning",
                   "quantisation", "nb_grids", "combined", "disk_size", "end_to_end", "latency",
                   "multiseed", "lorenz", "lorenz_curves", "edge_cases", "all_configs", "index",
                   "pareto", "multiseed_summary", "selector_pairs"]
EXPECTED_FIGURES = ["trackA_feature_pruning", "trackA_weight_sparsity",
                    "trackA_weight_sparsity_size", "trackB_quantisation", "trackB_nb_grids",
                    "pareto_accuracy_vs_size", "efficiency_index_top", "efficiency_final_top",
                    "end_to_end_size", "weight_lorenz"]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        results, figures = Path(tmp) / "results", Path(tmp) / "figures"
        env = {**os.environ, "SMRC_RESULTS_DIR": str(results), "SMRC_FIGURES_DIR": str(figures)}
        cmd = [sys.executable, str(ROOT / "scripts" / "run_all.py"), "--quick", "--data", "synthetic"]
        print("Running the full pipeline on synthetic data...")
        run = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if run.returncode != 0:
            print(run.stdout[-3000:], run.stderr[-3000:])
            print("\nSMOKE TEST FAILED: the pipeline did not finish")
            return 1

        problems = []
        for t in EXPECTED_TABLES:
            if not (results / f"{t}.csv").exists():
                problems.append(f"missing table {t}.csv")
        for f in EXPECTED_FIGURES:
            if not (figures / f"{f}.png").exists():
                problems.append(f"missing figure {f}.png")
        if not (results / "manifest.json").exists():
            problems.append("missing manifest.json")

        if not problems:
            configs = pd.read_csv(results / "all_configs.csv")
            if configs["macro_f1"].isna().any():
                problems.append("missing macro-F1 values")
            disk = pd.read_csv(results / "disk_size.csv")
            if not disk["reload_identical"].all():
                problems.append("a saved model did not reload identically")
            nb = pd.read_csv(results / "nb_grids.csv")
            if (nb["macro_f1"] < 0.5).any():
                problems.append("a Naive Bayes quantisation grid collapsed")
            weights = pd.read_csv(results / "weight_pruning.csv")
            pruned = weights[weights["weight_sparsity"] > 0.5]
            dense = weights[weights["weight_sparsity"] == 0]
            if not (pruned["model_size_kb"].max() < dense["model_size_kb"].min()):
                problems.append("weight pruning did not reduce the deployable size")

        if problems:
            print("\n".join(f"  - {p}" for p in problems))
            print("\nSMOKE TEST FAILED")
            return 1

    print(f"  {len(EXPECTED_TABLES)} tables and {len(EXPECTED_FIGURES)} figures produced")
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
