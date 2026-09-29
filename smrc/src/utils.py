"""Seeding and results logging."""
from __future__ import annotations

import csv
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed every source of randomness the project uses."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def log_result(row: dict, path: Path | None = None) -> None:
    """Append one experiment to a CSV file.

    Different experiments record different settings (a selector, a sparsity
    level, a bit-width), so the header grows whenever a new column appears.
    Older rows are kept and simply left blank in the new columns.
    """
    path = path or config.RESULTS_CSV
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}

    existing, fieldnames = [], []
    if path.exists():
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing = list(reader)

    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for old in existing:
            writer.writerow({k: old.get(k, "") for k in fieldnames})
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def print_row(row: dict) -> None:
    """Print a one-line summary of a logged result."""
    keys = ["seed", "model", "track", "variant", "selector", "keep_fraction",
            "min_df", "max_df", "weight_sparsity", "bits", "macro_f1",
            "latency_ms", "model_size_kb"]
    print("  " + " | ".join(f"{k}={row[k]}" for k in keys if k in row))
