"""Seeding."""
from __future__ import annotations

import os
import random

import numpy as np

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed every source of randomness the project uses."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
