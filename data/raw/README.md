# Raw dataset

This project expects a single tabular extract:

| File | `cms_Hbb.csv` |
|---|---|
| Expected size | 105,219,354 bytes |
| SHA-256 | `79455ebf15da90a9a28d5f816bee56d428b8799aa78d9439a18dc8119a93c751` |
| Rows × columns | 225,868 × 29 |
| Place this file at | `data/raw/cms_Hbb.csv` |

The CSV is **not** stored in git (~105 MB). Copy your local file here, then validate it:

```bash
python -m scripts.prepare_data --source "C:\path\to\cms_Hbb.csv"
```

`prepare_data` copies the file into this directory (unless it is already here) and checks the size and SHA-256.

## What the table is

Each row is one jet. `isSignal = 1` marks a jet associated with a Higgs boson decaying to bottom quarks (\(H \rightarrow b\bar{b}\)). `isBackground = 1` marks a QCD multijet jet. The two flags are complementary.

There are 26 physics features (tracking, secondary vertices, and subjet quantities) plus an export index (`Unnamed: 0`) and the two labels.

This table is a **derived extract**, not the full CERN Open Data sample. Feature names match the high-level `fj_*` variables in:

- Duarte, J. (2019). *Sample with jet, track and secondary vertex properties for Hbb tagging ML studies* (`HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC`). CERN Open Data Portal. [DOI: 10.7483/OPENDATA.CMS.JGJX.MS7Q](https://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q) ([record 12102](http://opendata.cern.ch/record/12102)).

The published Open Data record is ~228 GiB of ROOT files. This repository does **not** currently ship a script that rebuilds `cms_Hbb.csv` from those files. If you do not already have the CSV, you need the same derived table that was used for the original course project.

Do not commit `cms_Hbb.csv` to git.
