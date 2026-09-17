# Running on Kaggle then submitting on Colab

Same workflow as the deep learning project: **test on Kaggle → submit on Colab**.
Everything here is CPU-only, so there is no GPU or runtime type to worry about.

The notebook `Small_Models_Real_Constraints_Runner.ipynb` drives the whole
pipeline. Below is the one-time setup for each platform.

---

## Step 0 — push this repo to GitHub (once)

```bash
cd small-models-real-constraints
git init
git add .
git commit -m "Small Models, Real Constraints - capstone code"
git branch -M main
git remote add origin https://github.com/<your-username>/small-models-real-constraints.git
git push -u origin main
```

Then open the notebook and set `GITHUB_URL` in **Cell 1** to that repo URL.

---

## A) Test on Kaggle

1. Create a new Kaggle Notebook and upload
   `Small_Models_Real_Constraints_Runner.ipynb` (or paste the cells).
2. In the right-hand panel, turn **Internet = On** (needed to clone the repo and
   fetch AG News). Accelerator can stay **None** (CPU is the point).
3. Run cells top to bottom. Cell 3 (smoke test) should print `SMOKE TEST PASSED`.

### Fully offline on Kaggle (optional)
If you cannot enable internet:
- Add the repo as a Kaggle **Dataset** (or Utility Script). Cell 1 auto-detects a
  folder under `/kaggle/input` that contains `requirements.txt` + `src/`.
- Add the **AG News** dataset, then copy its `train.csv` / `test.csv` into
  `data/` and call `data.load(prefer="local")` in Cell 4.

---

## B) Submit on Colab

1. Open Colab → **File → Open notebook → GitHub**, paste your repo URL, and pick
   `Small_Models_Real_Constraints_Runner.ipynb`. (Or File → Upload notebook.)
2. Runtime type can stay **CPU**.
3. Run all cells. Cell 1 clones the repo, Cell 2 installs deps, the rest runs the
   experiments and displays the figures inline.
4. Cell 11 zips `results/` and `figures/` and downloads them.

---

## What each cell does

| Cell | Action |
|------|--------|
| 1 | Detect Colab/Kaggle, clone or locate the repo, `cd` into it |
| 2 | `pip install -r requirements.txt` |
| 3 | Smoke test on synthetic data (no downloads) |
| 4 | Load AG News (local → HuggingFace → CSV fallback) + sparsity print |
| 5 | Point 1 sparsity / RAM-savings report |
| 6 | Baselines (NB, LogReg, SVM) with nested CV |
| 7 | Track A pruning sweeps |
| 8 | Track B quantisation (incl. Naive Bayes) |
| 9 | Analysis: F1 vs efficiency index + Pareto |
| 10 | Generate + display all figures |
| 11 | Zip and download `results/` and `figures/` |

## Runtime tips
- The **baselines** cell is the slowest (nested CV). Expect a few minutes on CPU.
- In **Track A**, the `mutual_info` selector is the long pole on 20k features. For
  a fast first pass, remove it from `FEATURE_SELECTORS` in `src/config.py`.
- Re-running is safe: results append to `results/results.csv`. Delete that file if
  you want a clean slate.
