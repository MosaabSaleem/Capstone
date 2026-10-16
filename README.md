# Small Models, Real Constraints

Lightweight text classification on AG News, run entirely on a CPU. Three
classical models (Multinomial Naive Bayes, Logistic Regression and a linear
SVM) are tuned as baselines and then actively optimised:

- **Track A, pruning.** Feature selection with chi-square, mutual information
  and L1-embedded selection, document-frequency filtering, magnitude pruning
  of the learned weights, and the two pruning methods combined.
- **Track B, quantisation.** Simulated low-precision storage of every learned
  parameter array, including four quantisation grids for the Naive Bayes
  log-probability tables.

Every configuration is measured for macro-F1, latency, memory and size, both
in memory and as real files on disk. Deployment cost is also measured end to
end (raw text to prediction, vectoriser included), single-request latency is
split into fixed overhead and real work, headline results are repeated across
three seeds, and configurations are ranked with an efficiency index and a
Pareto frontier.

## Quick start

```bash
pip install -r requirements.txt
python scripts/smoke_test.py          # about a minute, no download
python scripts/run_all.py --quick     # reduced run on real data, for checking
python scripts/run_all.py             # full run, about two hours on a CPU
```

A full run clears `results/` and `figures/` first, so every number comes from
the same run. `--quick` uses a 20,000-document training subsample and smaller
sweeps, and writes to `results_quick/` and `figures_quick/` instead.

### If a run is interrupted

Each stage saves its results as soon as it finishes. Resume from the stage that
was running:

```bash
python scripts/run_all.py --from latency
```

Selected stages can also be re-run on their own, for example
`python scripts/run_all.py --only analysis` to redraw every figure.

## Running on Kaggle or Colab

Open `Small_Models_Real_Constraints_Runner.ipynb`. The first code cell has a
`QUICK` switch. CPU is fine on both platforms. On Kaggle, turn Internet on so
the code and data can be downloaded.

## Stages

| Stage | What it does | Writes |
|-------|--------------|--------|
| sparsity | TF-IDF sparsity, CSR against dense memory | `sparsity.csv` |
| baselines | Tune and evaluate the three models | `baselines.csv` |
| features | Chi-square, mutual information and L1 at six vocabulary budgets | `feature_selection.csv` |
| docfreq | min_df / max_df filtering | `docfreq.csv` |
| weights | Magnitude weight pruning, 0 to 99 percent | `weight_pruning.csv` |
| quant | Symmetric quantisation at 16, 8 and 4 bits | `quantisation.csv` |
| nbgrids | Four quantisation grids for Naive Bayes | `nb_grids.csv` |
| combined | L1 feature selection followed by weight pruning | `combined.csv` |
| disk | Real file sizes in three formats, with reload checks | `disk_size.csv` |
| endtoend | Raw text to prediction, vectoriser included | `end_to_end.csv` |
| latency | Batch, warm, cold, and overhead against real work | `latency.csv` |
| multiseed | Headline configurations under seeds 42, 43 and 44 | `multiseed.csv` |
| robustness | Weight concentration (Lorenz curves) and edge cases | `lorenz.csv`, `edge_cases.csv` |
| analysis | Efficiency index, Pareto frontier, seed summary, figures | `index.csv`, `pareto.csv`, `multiseed_summary.csv`, `selector_pairs.csv`, `manifest.json`, all figures |

`results/manifest.json` records the package versions, the time each stage
took, the number of rows in every table, and whether the baselines match the
reported run.

## Methodological notes

- Every model is tuned with the same grid search and stratified cross-validation
  on the training split only, including models fitted after feature selection.
- Feature-selection scores do not depend on how many terms are kept, so each
  selector is scored once and every vocabulary budget keeps the top terms.
- Mutual information is scored on term presence, on a 30,000-document subsample.
- Feature selection in the main sweep takes columns from the full TF-IDF matrix.
  The end-to-end stage instead rebuilds the vectoriser around the kept terms, as
  a real deployment would, so its rows are renormalised and its scores differ
  slightly.
- The final efficiency index counts halvings of size and of latency against the
  tuned Logistic Regression baseline and subtracts one for every macro-F1 point
  lost. The original index and an accuracy-gated version are reported alongside.

## Data

With `--data auto` (the default) the loader tries `data/train.csv` and
`data/test.csv`, then the Hugging Face `datasets` library, then a direct
download. Offline machines can use the first option.

## Repository layout

```
src/
  config.py      every setting, including quick-mode sweep sizes
  data.py        loading AG News and building TF-IDF features
  models.py      the three classifiers and their grid search
  optimise.py    feature selection, weight pruning and quantisation
  measure.py     accuracy, size, memory and latency measurement
  context.py     shared data and fitted models for a run
  stages.py      every experiment, one function per stage
  analysis.py    index, Pareto frontier, seed summary, figures, manifest
scripts/
  run_all.py     runs every stage in order
  smoke_test.py  end-to-end check on synthetic data
```
