"""Project configuration.

Every setting that affects the experiments lives here, so a run can be
reproduced by reading one file. Quick mode (set by passing --quick to any
script) swaps in a much smaller version of every sweep so the whole pipeline
can be checked in a few minutes instead of about two hours.
"""
from __future__ import annotations

import os
from pathlib import Path

QUICK = os.environ.get("SMRC_QUICK", "0") == "1"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# Quick runs write to their own folders so they never overwrite real results.
RESULTS_DIR = ROOT / ("results_quick" if QUICK else "results")
FIGURES_DIR = ROOT / ("figures_quick" if QUICK else "figures")

RESULTS_CSV = RESULTS_DIR / "results.csv"
PARETO_CSV = RESULTS_DIR / "pareto_front.csv"
MULTISEED_CSV = RESULTS_DIR / "results_multiseed.csv"
MULTISEED_SUMMARY = RESULTS_DIR / "multiseed_summary.csv"
ENV_FILE = RESULTS_DIR / "environment.txt"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data and features
# ---------------------------------------------------------------------------
SEED = 42
DATASET_NAME = "ag_news"
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 1.0

# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "naive_bayes": {"clf__alpha": [0.1, 0.5, 1.0]},
    "logreg": {"clf__C": [0.1, 1.0, 10.0]},
    "linear_svm": {"clf__C": [0.1, 1.0, 10.0]},
}

# ---------------------------------------------------------------------------
# Track A (pruning) and Track B (quantisation)
# ---------------------------------------------------------------------------
FEATURE_SELECTORS = ["chi2", "mutual_info", "l1"]
FEATURE_MODELS = ["naive_bayes", "logreg"]
WEIGHT_MODELS = ["logreg", "linear_svm"]
QUANT_MODELS = ["naive_bayes", "logreg", "linear_svm"]

# Mutual information is estimated on a stratified subsample of the training
# set, because scoring all 120,000 documents is very slow.
MI_SUBSAMPLE = 30_000

QUANT_SCHEME = "symmetric"

# Weights are stored sparsely once fewer than this fraction are non-zero.
SPARSE_STORAGE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Measurement and the efficiency index
# ---------------------------------------------------------------------------
LATENCY_REPEATS = 5
LATENCY_SAMPLE_SIZE = 2_000

# index = accuracy / (memory_MB ** ALPHA * latency_ms ** BETA)
EFFICIENCY_ALPHA = 1.0
EFFICIENCY_BETA = 1.0

# ---------------------------------------------------------------------------
# Sweep sizes: full run versus quick check
# ---------------------------------------------------------------------------
if not QUICK:
    TRAIN_SUBSAMPLE = None
    CV_FOLDS = 5
    FEATURE_KEEP_FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    DOCFREQ_FILTERS = [(2, 1.0), (5, 1.0), (10, 0.9), (20, 0.8), (50, 0.7)]
    WEIGHT_PRUNE_SPARSITIES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    QUANT_BITS = [16, 8, 4]

    SEEDS = [42, 43, 44]
    MULTISEED_KEEP_FRACTIONS = [0.5, 0.25, 0.1, 0.05]
    MULTISEED_SPARSITIES = [0.9, 0.95, 0.99]
    MULTISEED_QUANT_BITS = [8, 4]
else:
    TRAIN_SUBSAMPLE = 20_000
    CV_FOLDS = 3
    FEATURE_KEEP_FRACTIONS = [1.0, 0.25, 0.05]
    DOCFREQ_FILTERS = [(2, 1.0), (20, 0.8)]
    WEIGHT_PRUNE_SPARSITIES = [0.0, 0.9, 0.95, 0.99]
    QUANT_BITS = [8, 4]

    SEEDS = [42, 43]
    MULTISEED_KEEP_FRACTIONS = [0.25, 0.05]
    MULTISEED_SPARSITIES = [0.95]
    MULTISEED_QUANT_BITS = [4]
    MI_SUBSAMPLE = 10_000
