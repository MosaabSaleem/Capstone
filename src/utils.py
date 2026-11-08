"""Seeding."""
from __future__ import annotations

import os
import random

import numpy as np

from . import config

# Seed every source of randomness the project uses
def set_seed(seed: int = config.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
