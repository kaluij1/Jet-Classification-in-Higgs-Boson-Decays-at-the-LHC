# Jet Classification in Higgs Boson Decays at the LHC

Binary classification of CMS-style fat jets: \(H \rightarrow b\bar{b}\) (signal) versus QCD multijet (background), using 26 high-level tracking and secondary-vertex features.

The default pipeline is a leakage-controlled train / validation / test evaluation (P1). The original course notebook is archived under `reports/original/` and can still be reproduced with `--legacy`.

## Results snapshot

Numbers below are from `python -m scripts.train` with `random_state=42` on the SHA-256-checked CSV. The **test set keeps the natural 35% / 65% class mix**. Thresholds, PCA, imputers, and scalers are fit on training data only. The Punzi threshold is chosen on **validation** and applied to test once.

| Model | Val AUC | Test AUC | Test AP | Bkg. rejection @ 50% \(\varepsilon_s\) | @ 80% \(\varepsilon_s\) |
|---|---|---|---|---|---|
| Random forest | **0.907** | **0.906** | **0.827** | **23.5** | **6.79** |
| Logistic regression | 0.859 | 0.856 | 0.735 | 13.1 | 4.31 |
| Logistic regression + PCA | 0.853 | 0.851 | 0.720 | 11.9 | 4.16 |
| LDA | 0.849 | 0.846 | 0.726 | 12.3 | 3.84 |
| Naive Bayes | 0.821 | 0.821 | 0.658 | 8.63 | 3.56 |
| Physics cut (`trackSip2dSigAboveBottom_0`) | 0.762 | 0.757 | 0.628 | 7.88 | 2.22 |

Background rejection is \(1/\mathrm{FPR}\) at the stated signal efficiency. The physics baseline is the single feature with the best **validation** AUC among `nSV`, `trackSipdSig_0`, `trackSip2dSigAboveBottom_0`, and `tau_flightDistance2dSig_0` (higher values more signal-like).

Random forest is selected by validation AUC. On the test set, its Punzi operating point (threshold **0.724**, chosen on validation) gives Punzi **0.0137**, signal precision **0.876**, and signal recall **0.450**. At the default 0.5 threshold it reaches accuracy **0.836** and signal F1 **0.756**. Accuracy is a weak headline metric here because the test prior is unbalanced.

ROC and Punzi curves from this run are written to `artifacts/figures/`.

## Dataset

| | |
|---|---|
| File | `data/raw/cms_Hbb.csv` |
| Rows × columns | 225,868 × 29 |
| Signal / background | 79,310 / 146,558 |
| Size | 105,219,354 bytes |
| SHA-256 | `79455ebf15da90a9a28d5f816bee56d428b8799aa78d9439a18dc8119a93c751` |
| Source | Derived from CERN Open Data [record 12102](http://opendata.cern.ch/record/12102), DOI [10.7483/OPENDATA.CMS.JGJX.MS7Q](https://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q) |
| Parent license | CC0-1.0 (confirmed on the record 12102 page) |
| Download | [10.5281/zenodo.22310265](https://doi.org/10.5281/zenodo.22310265) — see [`data/raw/README.md`](data/raw/README.md) |

Each row is one jet. `isSignal` and `isBackground` are complementary labels. `Unnamed: 0` is an export index, **not** a unique event ID (134,604 distinct values in 225,868 rows), so splits are stratified by label only.

102 rows have `tau_vertexEnergyRatio_{0,1} = -1` on both columns. Those values are treated as missing: a binary indicator plus a median imputed from **training** non-sentinel values. The effect on logistic-regression validation AUC is small (0.8583 raw vs 0.8586 handled).

The published CERN record is ~228 GiB of ROOT files. This repo does not rebuild the CSV from those files. Download `cms_Hbb.csv` from Zenodo ([10.5281/zenodo.22310265](https://doi.org/10.5281/zenodo.22310265)) and run `python -m scripts.prepare_data`. Details: [`data/raw/README.md`](data/raw/README.md).

## Quickstart

Use **Python 3.11**. scikit-learn is pinned to 1.3.x.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e .
```

On macOS/Linux, activate with `source .venv/bin/activate`.

```bash
curl -L -o data/raw/cms_Hbb.csv "https://zenodo.org/records/22310265/files/cms_Hbb.csv"
python -m scripts.prepare_data --source data/raw/cms_Hbb.csv
python -m scripts.train --config configs/default.yaml
```

This writes `artifacts/metrics.json` and `artifacts/figures/`. Random Forest is the slow step; skip it with `--skip-random-forest`. Seed, split fractions, and paths live in `configs/default.yaml`.

To reproduce the original notebook numbers (balanced test set, known leakage):

```bash
python -m scripts.train --legacy --output artifacts/metrics_legacy.json
```

## Project layout

```text
src/hbb_classification/   data checks, P1 experiment, legacy notebook pipeline
scripts/                  python -m entry points
configs/                  default seed, split, and paths
tests/                    schema, leakage, Punzi, and smoke tests on a toy table
data/raw/                 cms_Hbb.csv (gitignored) and data instructions
artifacts/                metrics.json and figures from a local run
reports/original/         course notebook and PDF, frozen
.github/workflows/        CI: ruff + pytest (no 105 MB download)
```

## Tests

CI never fetches `cms_Hbb.csv`. Tests build a 60-row table with the same columns and label rules.

```bash
pip install -e ".[dev]"
pytest
```

## Method (P1)

1. Load and hash-check `cms_Hbb.csv`. Drop `Unnamed: 0` and both label columns from `X`.
2. Stratified **60 / 20 / 20** train / validation / test split on the raw labels. No undersampling of evaluation data.
3. Replace energy-ratio `-1` sentinels with a missingness indicator and a train-only median.
4. Fit each model as a sklearn `Pipeline` so scalers and PCA see training rows only:
   - Gaussian Naive Bayes (no scaling)
   - LDA and logistic regression (`class_weight='balanced'`, scaled)
   - Logistic regression on PCA components that explain 85% of **training** variance
   - Random forest (`class_weight='balanced'`), scored as a classifier
5. Compare against a one-feature physics cut selected on validation AUC.
6. Report ROC AUC, average precision, and background rejection at 50% and 80% signal efficiency.
7. Maximize Punzi significance \(\varepsilon / (1 + \sqrt{B})\) on the **validation** scores (if \(B=0\), Punzi is \(\varepsilon\), not 0). Freeze that threshold and report test once.

## Limitations

- Monte Carlo jets only; no detector data and no systematic uncertainties.
- This is not a CMS physics result. Punzi here uses raw test counts, not a luminosity-scaled background estimate.
- `Unnamed: 0` is not a unique event key, so there is no group-aware split.
- No hyperparameter search beyond model defaults.
- The 102 sentinel rows are rare; handling them changes logistic AUC only in the fourth decimal place.

## Original course work

The submitted notebook and PDF are in [`reports/original/`](reports/original/). That analysis undersampled before splitting, fit PCA on the full balanced matrix, and chose the Punzi threshold on the test set. Use `--legacy` if you need those numbers.

## Citation

Cite this repository (`CITATION.cff`), the derived table, and the parent Open Data record:

- Kaluiji, J. (2026). *cms_Hbb: Derived High-Level Feature Table for H→bb̄ vs QCD Jet Tagging*. Zenodo. [DOI: 10.5281/zenodo.22310265](https://doi.org/10.5281/zenodo.22310265).
- Duarte, J. (2019). *Sample with jet, track and secondary vertex properties for Hbb tagging ML studies*. CERN Open Data Portal. [DOI: 10.7483/OPENDATA.CMS.JGJX.MS7Q](https://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q).

## Environment

- Python `>=3.11,<3.13`
- `pandas`, `numpy<2`, `matplotlib`, `seaborn`, `scikit-learn>=1.3,<1.4`, `pyyaml`
- Dev extras: `pytest`, `ruff`

See `pyproject.toml`.
