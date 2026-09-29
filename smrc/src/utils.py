"""Seeding + schema-flexible results logging."""
from __future__ import annotations
import csv, os, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from . import config


def set_seed(seed: int = config.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_result(row: dict, path: Path = None) -> None:
    """Append one experiment row, growing columns as needed."""
    path = path or config.RESULTS_CSV
    row = {"timestamp": utc_stamp(), **row}
    path.parent.mkdir(parents=True, exist_ok=True)
    existing, fieldnames = [], []
    if path.exists():
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing = list(reader)
    for k in row:
        if k not in fieldnames:
            fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for old in existing:
            w.writerow({k: old.get(k, "") for k in fieldnames})
        w.writerow({k: row.get(k, "") for k in fieldnames})


def print_row(row: dict) -> None:
    keys = ["seed", "model", "track", "variant", "selector", "keep_fraction",
            "weight_sparsity", "bits", "macro_f1", "model_size_kb"]
    print("  " + " | ".join(f"{k}={row[k]}" for k in keys if k in row))
