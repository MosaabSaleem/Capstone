# Multi-seed runs

## Why

The main sweep runs on a single seed (42). That is fine for large effects, but it
cannot distinguish small differences from run-to-run noise. The clearest example
is the chi-square versus mutual-information comparison, where the gap is a few
thousandths of a point of macro-F1. On one seed there is no way to tell whether
that is a real ordering or luck.

This adds a second, separate experiment that repeats only the headline
configurations across several seeds and reports mean +/- standard deviation.

## How to run

```bash
python scripts/run_multiseed.py            # all stages, all seeds
python scripts/summarise_multiseed.py      # aggregate + paired comparison
```

Individual stages, if you want to split the runtime up:

```bash
python scripts/run_multiseed.py baselines
python scripts/run_multiseed.py features    # the selector comparison
python scripts/run_multiseed.py weights
python scripts/run_multiseed.py quant
```

## What it writes

| File | Contents |
|------|----------|
| `results/results_multiseed.csv` | every individual run, one row per seed per config |
| `results/multiseed_summary.csv` | mean, std, min, max, n per configuration |

The original `results/results.csv` is never touched, so the single-seed sweep
and its figures stay valid.

## Scope

Only configurations the report actually makes claims about are repeated, set by
the `MULTISEED_*` values in `src/config.py`:

- **Baselines** — all three models
- **Feature selection** — all three selectors at 50%, 25%, 10% and 5% budgets
- **Weight pruning** — 90%, 95% and 99% sparsity, around the reported knee
- **Quantisation** — 8-bit and 4-bit

Running the whole 76-row sweep three times would take hours and most of those
rows are not claims the report leans on.

## Reading the output

The summariser prints three things.

1. **Summary table** with a `macro_f1_report` column already formatted as
   `0.9059 +/- 0.0012`, ready to paste into the report.

2. **Selector comparison**, done as a paired test. Both selectors see the same
   seed and the same split, so pairing removes between-seed variance. For each
   pair it reports the mean difference, its standard deviation, how often the
   first selector wins, and a paired t-statistic. A gap smaller than seed
   variation is labelled "within noise", which means the two should be reported
   as comparable rather than ranked.

3. **Stability report**, listing the configurations that move most across seeds,
   plus the median standard deviation. Differences smaller than roughly twice
   that value should not be treated as meaningful.

## Using it in the report

If chi-square and mutual information come back "within noise", say so directly:
they are statistically indistinguishable on this dataset, and only the L1
advantage is large enough to assert. That is a stronger and more honest claim
than ranking three methods on one run, and it removes the single-seed limitation
currently listed in Section 5.5.
