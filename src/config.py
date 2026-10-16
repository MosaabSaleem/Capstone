"""Project configuration.

Every setting that affects the experiments lives here, so a run can be
reproduced by reading one file. Quick mode (pass --quick to run_all.py) swaps in
a much smaller version of every sweep so the whole pipeline can be checked in a
few minutes. Quick runs write to results_quick/ and figures_quick/, so they can
never overwrite the real results.
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
RESULTS_DIR = Path(os.environ.get("SMRC_RESULTS_DIR",
                                  ROOT / ("results_quick" if QUICK else "results")))
FIGURES_DIR = Path(os.environ.get("SMRC_FIGURES_DIR",
                                  ROOT / ("figures_quick" if QUICK else "figures")))
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
# Models
# ---------------------------------------------------------------------------
MODELS = ["naive_bayes", "logreg", "linear_svm"]
LINEAR_MODELS = ["logreg", "linear_svm"]
PARAM_GRIDS = {
    "naive_bayes": {"clf__alpha": [0.1, 0.5, 1.0]},
    "logreg": {"clf__C": [0.1, 1.0, 10.0]},
    "linear_svm": {"clf__C": [0.1, 1.0, 10.0]},
}

# Every model in the project, including those fitted after feature selection,
# is tuned with the same grid search. Results from earlier separate runs, where
# models after feature selection were refitted at default settings, are
# therefore not directly comparable with this run.

# ---------------------------------------------------------------------------
# Optimisation settings
# ---------------------------------------------------------------------------
SELECTORS = ["chi2", "mutual_info", "l1"]
MI_SUBSAMPLE = 30_000          # mutual information is scored on a subsample
SPARSE_STORAGE_THRESHOLD = 0.5  # store weights sparsely below this density
QUANT_SCHEME = "symmetric"     # grid used for the main quantisation sweep

# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
LATENCY_REPEATS = 5        # batch latency: best of this many runs
LATENCY_SAMPLE_SIZE = 2_000

# Efficiency index. The original index is accuracy / (MB x ms). The final
# index counts halvings of size and latency against the tuned Logistic
# Regression baseline and subtracts INDEX_PENALTY for every macro-F1 point lost.
INDEX_PENALTY = 1.0
INDEX_PENALTIES = [0.5, 1.0, 2.0]
INDEX_GATE_POINTS = 1.0     # gated index: only configs within this many points

# Baselines from the reported run. The pipeline warns if a full run on AG News
# does not reproduce them, which would point to a change in data or libraries.
EXPECTED_BASELINES = {"naive_bayes": 0.8939, "logreg": 0.9179, "linear_svm": 0.9177}

# ---------------------------------------------------------------------------
# Sweep sizes: full run against quick check
# ---------------------------------------------------------------------------
if not QUICK:
    TRAIN_SUBSAMPLE = None
    CV_FOLDS = 5
    KEEP_FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    DOCFREQ_FILTERS = [(2, 1.0), (5, 1.0), (10, 0.9), (20, 0.8), (50, 0.7)]
    SPARSITIES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    QUANT_BITS = [16, 8, 4]
    NB_GRID_BITS = [8, 6, 4, 3]
    COMBINED_KEEPS = [1.0, 0.25, 0.1, 0.05]
    COMBINED_SPARSITIES = [0.0, 0.5, 0.8, 0.9, 0.95]
    DEPLOY_KEEPS = [0.25, 0.1, 0.05]
    SEEDS = [42, 43, 44]
    MULTISEED_KEEPS = [0.5, 0.25, 0.1, 0.05]
    MULTISEED_SPARSITIES = [0.9, 0.95, 0.99]
    MULTISEED_BITS = [8, 4]
    WARM_TRIALS = 500
    COLD_TRIALS = 30
    OVERHEAD_REPEATS = 3000
else:
    TRAIN_SUBSAMPLE = 20_000
    CV_FOLDS = 3
    KEEP_FRACTIONS = [1.0, 0.25, 0.1, 0.05]
    DOCFREQ_FILTERS = [(2, 1.0), (20, 0.8)]
    SPARSITIES = [0.0, 0.9, 0.95, 0.99]
    QUANT_BITS = [8, 4]
    NB_GRID_BITS = [8, 4]
    COMBINED_KEEPS = [1.0, 0.1]
    COMBINED_SPARSITIES = [0.0, 0.9]
    DEPLOY_KEEPS = [0.1]
    SEEDS = [42, 43]
    MULTISEED_KEEPS = [0.25, 0.1]
    MULTISEED_SPARSITIES = [0.95]
    MULTISEED_BITS = [4]
    WARM_TRIALS = 50
    COLD_TRIALS = 5
    OVERHEAD_REPEATS = 200
    MI_SUBSAMPLE = 10_000
