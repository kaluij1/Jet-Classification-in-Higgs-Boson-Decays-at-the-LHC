# Model card

Binary \(H \rightarrow b\bar{b}\) vs QCD jet classifier on a derived CMS Open Data table.

This is **not** a CMS physics result. It is a portfolio / methods demonstration on Monte Carlo jets.

## Intended use

- Compare linear, tree, and one-feature physics baselines under a leakage-controlled split.
- Report HEP-style working points (AUC, average precision, background rejection, Punzi).

Out of scope: luminosity-weighted significance, systematic uncertainties, detector data, or official tagging performance.

## Data

| | |
|---|---|
| Table | `cms_Hbb.csv` (225,868 × 29) |
| Labels | `isSignal` / `isBackground` (complementary) |
| Features | 26 high-level track / SV / subjet quantities |
| Hosted extract | [10.5281/zenodo.22310265](https://doi.org/10.5281/zenodo.22310265) (CC-BY-4.0) |
| Parent sample | CERN Open Data [record 12102](http://opendata.cern.ch/record/12102), [10.7483/OPENDATA.CMS.JGJX.MS7Q](https://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q) (CC0-1.0) |

`Unnamed: 0` is not a unique event ID. Splits are stratified by label only.

102 rows have `tau_vertexEnergyRatio_{0,1} = -1`. Training uses a missingness indicator plus a train-only median.

## Training and evaluation

- Stratified 60 / 20 / 20 train / validation / test on the **natural** 35% / 65% mix.
- Scalers, PCA, and imputers fit on train only.
- Punzi threshold \(\varepsilon / (1 + \sqrt{B})\) chosen on validation, applied to test once.
- `class_weight='balanced'` on logistic regression, LDA pipelines that support it, and random forest.
- Seed `42` (`configs/default.yaml`).

## Metrics (test set)

From `python -m scripts.train` with the SHA-256-checked CSV. Background rejection is \(1/\mathrm{FPR}\).

| Model | Test AUC | Test AP | Rejection @ 50% \(\varepsilon_s\) | @ 80% \(\varepsilon_s\) |
|---|---|---|---|---|
| Random forest (selected by val AUC) | 0.906 | 0.827 | 23.5 | 6.79 |
| Logistic regression | 0.856 | 0.735 | 13.1 | 4.31 |
| Logistic regression + PCA | 0.851 | 0.720 | 11.9 | 4.16 |
| LDA | 0.846 | 0.726 | 12.3 | 3.84 |
| Naive Bayes | 0.821 | 0.658 | 8.63 | 3.56 |
| Physics cut (`trackSip2dSigAboveBottom_0`) | 0.757 | 0.628 | 7.88 | 2.22 |

Random forest at the validation Punzi threshold (0.724): test Punzi 0.0137, signal precision 0.876, signal recall 0.450. At score 0.5: accuracy 0.836, signal F1 0.756.

Punzi here uses raw test counts, not a scaled expected background.

## Controls against leakage

- Evaluation sets are not undersampled.
- Feature selection / PCA / scaling never see test (or validation) during `fit`.
- Model ranking uses validation AUC; test numbers are reported once.

The original course notebook (undersample-then-split, PCA on the full balanced matrix, Punzi on test) is archived under `reports/original/` and reproduced only with `--legacy`.

## Limitations

- Simulation only (Pythia 8 / CMS Open Data MC).
- No trigger, jet-energy, or luminosity systematics.
- No hyperparameter search beyond library defaults plus the stated pipeline choices.
- Sentinel handling changes logistic validation AUC only in the fourth decimal place.

## Software license

MIT (see `LICENSE`). The derived CSV is CC-BY-4.0 on Zenodo; the parent ROOT sample is CC0-1.0.
