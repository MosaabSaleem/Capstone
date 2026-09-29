"""Central configuration — every knob lives here so experiments stay reproducible."""
from __future__ import annotations
from pathlib import Path

SEED = 42
# Seeds used by the multi-seed runner (scripts/run_multiseed.py).
SEEDS = [42, 43, 44]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
ARTIFACT_DIR = ROOT / "artifacts"
RESULTS_CSV = RESULTS_DIR / "results.csv"
# Multi-seed runs log here, kept separate so the single-seed sweep stays intact.
MULTISEED_CSV = RESULTS_DIR / "results_multiseed.csv"
MULTISEED_SUMMARY = RESULTS_DIR / "multiseed_summary.csv"

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

MI_SUBSAMPLE = 30_000

QUANT_BITS = [16, 8, 4]
QUANT_SCHEME = "symmetric"

LATENCY_REPEATS = 5
LATENCY_SAMPLE_SIZE = 2_000

SPARSE_STORAGE_THRESHOLD = 0.5

EFFICIENCY_ALPHA = 1.0
EFFICIENCY_BETA = 1.0

# ---------------------------------------------------------------------------
# Multi-seed experiment scope
# ---------------------------------------------------------------------------
# Only the headline configurations are repeated across seeds. Running the whole
# 76-row sweep three times would take hours and most of those rows are not
# claims the report actually leans on.
#
# The chi2-vs-mutual_info comparison is the specific question multi-seed exists
# to settle, so the full selector sweep is included at the budgets where the
# report makes a claim.
MULTISEED_KEEP_FRACTIONS = [0.5, 0.25, 0.1, 0.05]
MULTISEED_SELECTORS = ["chi2", "mutual_info", "l1"]
MULTISEED_FEATURE_MODELS = ["logreg", "naive_bayes"]
# Sparsity levels around the reported knee of the curve.
MULTISEED_SPARSITIES = [0.9, 0.95, 0.99]
MULTISEED_WEIGHT_MODELS = ["logreg", "linear_svm"]
MULTISEED_QUANT_BITS = [8, 4]
MULTISEED_QUANT_MODELS = ["naive_bayes", "logreg", "linear_svm"]
