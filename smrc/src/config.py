"""Central configuration — every knob lives here so experiments stay reproducible."""
from __future__ import annotations
from pathlib import Path

SEED = 42
SEEDS = [42, 43, 44]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
ARTIFACT_DIR = ROOT / "artifacts"
RESULTS_CSV = RESULTS_DIR / "results.csv"
for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "ag_news"
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 1.0

CV_FOLDS = 5
PARAM_GRIDS = {
    "naive_bayes": {"clf__alpha": [0.1, 0.5, 1.0]},
    "logreg": {"clf__C": [0.1, 1.0, 10.0]},
    "linear_svm": {"clf__C": [0.1, 1.0, 10.0]},
}

FEATURE_KEEP_FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
FEATURE_SELECTORS = ["chi2", "mutual_info", "l1"]
DOCFREQ_FILTERS = [(2, 1.0), (5, 1.0), (10, 0.9), (20, 0.8), (50, 0.7)]
WEIGHT_PRUNE_SPARSITIES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

# --- FIX 1 support -----------------------------------------------------------
# Mutual information is scored on BINARISED term presence (see pruning.py), which
# is the classical text feature-selection formulation and keeps X sparse.
# Optionally fit the selector on a stratified subsample for speed.
# Set to None to score on all rows.
MI_SUBSAMPLE = 30_000

QUANT_BITS = [16, 8, 4]
QUANT_SCHEME = "symmetric"

LATENCY_REPEATS = 5
LATENCY_SAMPLE_SIZE = 2_000

# --- FIX 2 support -----------------------------------------------------------
# Below this density, storing a weight matrix as CSR is the sensible deployment
# choice, so the reported model size should reflect the sparse footprint.
SPARSE_STORAGE_THRESHOLD = 0.5

EFFICIENCY_ALPHA = 1.0
EFFICIENCY_BETA = 1.0
