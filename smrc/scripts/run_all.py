"""Run the whole pipeline in order.

    python scripts/run_all.py              full run, about two hours on a CPU
    python scripts/run_all.py --quick      reduced run, a few minutes
    python scripts/run_all.py --skip-multiseed
"""
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

STEPS = [
    "print_env.py",
    "report_sparsity.py",
    "run_baselines.py",
    "run_track_a.py",
    "run_track_b.py",
    "analyse.py",
    "make_figures.py",
    "run_multiseed.py",
    "summarise_multiseed.py",
]
MULTISEED_STEPS = {"run_multiseed.py", "summarise_multiseed.py"}


def main():
    args = sys.argv[1:]
    skip_multiseed = "--skip-multiseed" in args
    passthrough = [a for a in args if a != "--skip-multiseed"]

    start = time.perf_counter()
    for step in STEPS:
        if skip_multiseed and step in MULTISEED_STEPS:
            continue
        print(f"\n{'=' * 60}\n{step}\n{'=' * 60}", flush=True)
        result = subprocess.run([sys.executable, str(SCRIPTS / step), *passthrough])
        if result.returncode != 0:
            sys.exit(f"{step} failed with exit code {result.returncode}")

    print(f"\nPipeline finished in {(time.perf_counter() - start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
