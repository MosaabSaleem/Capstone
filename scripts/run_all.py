"""Run every experiment in the project and produce every table and figure.

    python scripts/run_all.py                    full run, about two hours on a CPU
    python scripts/run_all.py --quick            reduced run, for checking the pipeline
    python scripts/run_all.py --from latency     resume from a stage after an interruption
    python scripts/run_all.py --only analysis    re-run selected stages (comma separated)

A full run (no --from or --only) clears the results and figures folders first,
so every number comes from the same run. Each stage saves its CSV as soon as it
finishes, so an interrupted run keeps all completed stages.

Stages, in order:
  sparsity, baselines, features, docfreq, weights, quant, nbgrids, combined,
  disk, endtoend, latency, multiseed, robustness, analysis
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

if "--quick" in sys.argv:
    os.environ["SMRC_QUICK"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import analysis, config  # noqa: E402
from src.context import Context  # noqa: E402
from src.stages import STAGES  # noqa: E402
from src.utils import set_seed  # noqa: E402

ORDER = [*STAGES, "analysis"]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="small sweeps on a 20k training subsample")
    p.add_argument("--data", default="auto", choices=["auto", "local", "hf", "csv", "synthetic"])
    p.add_argument("--only", help="comma-separated stages to run")
    p.add_argument("--from", dest="start", choices=ORDER, help="resume from this stage")
    return p.parse_args()


def clear_outputs():
    for folder, pattern in ((config.RESULTS_DIR, "*.csv"), (config.RESULTS_DIR, "*.json"),
                            (config.RESULTS_DIR, "*.txt"), (config.FIGURES_DIR, "*.png")):
        for f in folder.glob(pattern):
            f.unlink()


def main():
    args = parse()
    if args.only:
        stages = [s.strip() for s in args.only.split(",")]
        unknown = [s for s in stages if s not in ORDER]
        if unknown:
            sys.exit(f"Unknown stage(s): {unknown}. Choose from: {', '.join(ORDER)}")
    elif args.start:
        stages = ORDER[ORDER.index(args.start):]
    else:
        stages = ORDER
        clear_outputs()

    print(f"Mode: {'quick check' if config.QUICK else 'full run'} | data: {args.data}")
    print(f"Results: {config.RESULTS_DIR}\nStages: {', '.join(stages)}\n")
    set_seed(config.SEED)
    ctx = Context(args.data)
    total = time.perf_counter()

    for name in stages:
        print(f"{'=' * 70}\n{name}\n{'=' * 70}")
        start = time.perf_counter()
        try:
            if name == "analysis":
                analysis.run(ctx)
            else:
                STAGES[name](ctx)
        except Exception:
            traceback.print_exc()
            sys.exit(f"\nStage '{name}' failed. Completed stages are saved. "
                     f"Fix the problem and resume with: python scripts/run_all.py --from {name}"
                     + (" --quick" if config.QUICK else ""))
        ctx.timings[name] = time.perf_counter() - start
        print(f"({name} took {ctx.timings[name] / 60:.1f} min)\n")

    print(f"Finished in {(time.perf_counter() - total) / 60:.1f} minutes.")
    print(f"Tables in {config.RESULTS_DIR}, figures in {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
