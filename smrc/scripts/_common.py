"""Setup shared by every script.

Import this before anything from src. It puts the repository root on the path
and switches on quick mode when --quick is passed, which has to happen before
src.config is first imported.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "--quick" in sys.argv:
    os.environ["SMRC_QUICK"] = "1"

DATA_SOURCES = ["auto", "local", "hf", "csv", "synthetic"]


def parse_args(description: str, stages: list[str] | None = None):
    parser = argparse.ArgumentParser(description=description)
    if stages:
        parser.add_argument("stage", nargs="?", default="all", choices=["all", *stages],
                            help="run one stage only (default: all)")
    parser.add_argument("--data", default="auto", choices=DATA_SOURCES,
                        help="where to load AG News from (default: auto)")
    parser.add_argument("--quick", action="store_true",
                        help="small sweeps on a 20k training subsample, for checking the pipeline")
    return parser.parse_args()
