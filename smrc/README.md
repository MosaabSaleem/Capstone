# Small Models, Real Constraints

Lightweight text classification on AG News, run entirely on a CPU. Three
classical models (Multinomial Naive Bayes, Logistic Regression and a linear
SVM) are tuned as baselines and then actively optimised in two ways:

- **Track A, pruning.** Feature selection with chi-square, mutual information
  and L1-embedded selection, document-frequency filtering, and magnitude
  pruning of the learned weights.
- **Track B, quantisation.** Simulated low-precision storage of every learned
  parameter array, including the log-probability tables of Naive Bayes.

Each configuration is measured for macro-F1, latency, memory and model size,
ranked with an efficiency index, and summarised on a Pareto frontier. The
headline configurations are then repeated across three random seeds.

## Quick start

```bash
pip install -r requirements.txt
python scripts/smoke_test.py            # about 10 seconds, no download
python scripts/run_all.py --quick       # reduced run, a few minutes
python scripts/run_all.py               # full run, about two hours on a CPU
```

`--quick` uses a 20,000-document training subsample and much smaller sweeps.
It exists to check the pipeline works. Its output goes to `results_quick/` and
`figures_quick/`, so it never overwrites the real results.

## Running on Kaggle or Colab

Open `Small_Models_Real_Constraints_Runner.ipynb`. The first code cell has a
`QUICK` switch. Leave it on to check everything in a few minutes, or turn it off
for the full run. CPU is fine on both platforms, no accelerator is needed.

## Running the steps individually

Run from the repository root, in this order:

| Step | Command | Writes |
|------|---------|--------|
| 1 | `python scripts/print_env.py` | `environment.txt` |
| 2 | `python scripts/report_sparsity.py` | sparsity row in `results.csv` |
| 3 | `python scripts/run_baselines.py` | baseline rows |
| 4 | `python scripts/run_track_a.py` | feature selection, document frequency and weight pruning rows |
| 5 | `python scripts/run_track_b.py` | quantisation rows |
| 6 | `python scripts/analyse.py` | `pareto_front.csv` and the rankings |
| 7 | `python scripts/make_figures.py` | every figure in the report |
| 8 | `python scripts/run_multiseed.py` | `results_multiseed.csv` |
| 9 | `python scripts/summarise_multiseed.py` | `multiseed_summary.csv` and the paired selector comparison |

Every script accepts `--quick` and `--data {auto,local,hf,csv,synthetic}`.
`run_multiseed.py` can also run a single stage, for example
`python scripts/run_multiseed.py features`.

## Where the data comes from

With `--data auto` (the default) the loader tries, in order:

1. `data/train.csv` and `data/test.csv`, in the original AG News CSV layout
2. the Hugging Face `datasets` library
3. a direct download of the original CSV files

If a machine has no internet access, place the two CSV files in `data/`.

## Runtime on a Kaggle CPU instance

| Stage | Time |
|-------|------|
| Main sweep (steps 1 to 7) | about 42 minutes |
| Multi-seed runs (steps 8 and 9) | about 68 minutes |
| Quick mode, everything | a few minutes |

Most of the time goes into mutual-information feature selection, which is
estimated on a 30,000-document subsample to keep it manageable.

## Repository layout

```
src/
  config.py         every setting, including quick-mode sweep sizes
  data.py           loading AG News and building TF-IDF features
  sparsity.py       CSR against dense memory measurement
  baselines.py      the three classifiers and their grid search
  metrics.py        latency, memory, model size and the efficiency index
  pruning.py        feature selection and magnitude weight pruning
  quantisation.py   simulated quantisation for classical models
  analysis.py       rankings and the Pareto frontier
  plots.py          report figures
scripts/            one script per step, plus run_all.py and smoke_test.py
results/            logged results (created on first run)
figures/            generated figures
```

## Reproducibility

Every random process is seeded from `config.SEED` (42). The multi-seed runs use
seeds 42, 43 and 44. AG News has a fixed train and test split, and most of the
pipeline is deterministic, so repeated runs on the same environment reproduce
the reported numbers exactly. The exact package versions for a run are written
to `results/environment.txt` by step 1.
