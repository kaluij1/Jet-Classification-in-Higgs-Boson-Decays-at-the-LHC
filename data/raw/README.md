# Raw dataset

This project uses a derived tabular extract of CMS Open Data:

| | |
|---|---|
| File | `cms_Hbb.csv` |
| Size | 105,219,354 bytes |
| SHA-256 | `79455ebf15da90a9a28d5f816bee56d428b8799aa78d9439a18dc8119a93c751` |
| MD5 (Zenodo) | `c3c1a0a52ddcd590c10f512e5cc8c79c` |
| Rows × columns | 225,868 × 29 |
| Source | Derived from CERN Open Data record [12102](http://opendata.cern.ch/record/12102) (`HiggsToBBNTuple`), DOI: [10.7483/OPENDATA.CMS.JGJX.MS7Q](https://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q) |
| Parent license | Creative Commons Zero v1.0 Universal (CC0-1.0), confirmed on the record 12102 page |
| Derived-table license | CC-BY-4.0 |
| Download | [DOI: 10.5281/zenodo.22310265](https://doi.org/10.5281/zenodo.22310265) (v2; [record](https://zenodo.org/records/22310265)) |

v2 was checked against the local CSV: size 105,219,354 bytes and MD5 `c3c1a0a52ddcd590c10f512e5cc8c79c` both match the [Zenodo API](https://zenodo.org/api/records/22310265). Cite the concept DOI [10.5281/zenodo.22310118](https://doi.org/10.5281/zenodo.22310118) if you want all versions.

## Getting the data

```bash
curl -L -o data/raw/cms_Hbb.csv "https://zenodo.org/records/22310265/files/cms_Hbb.csv"
python -m scripts.prepare_data --source data/raw/cms_Hbb.csv
```

`prepare_data` copies the file here if needed and checks size plus SHA-256.

If you already have the CSV locally:

```bash
python -m scripts.prepare_data --source "C:\path\to\cms_Hbb.csv"
```

## What the table is

Each row is one jet. `isSignal = 1` marks a jet associated with a Higgs boson decaying to bottom quarks (H → bb̄). `isBackground = 1` marks a QCD multijet jet. The two flags are complementary. There are 26 physics features (tracking, secondary-vertex, and subjet quantities) plus an export index (`Unnamed: 0`) and the two labels.

This table is a derived extract, not the full CERN Open Data sample — the published record is ~228 GiB of ROOT files. Feature names correspond to a subset of the `fj_*` branches in:

> Duarte, J. (2019). *Sample with jet, track and secondary vertex properties for Hbb tagging ML studies* (HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC). CERN Open Data Portal. DOI: [10.7483/OPENDATA.CMS.JGJX.MS7Q](https://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q) (record 12102).

**Reproducibility note.** The extract is hosted on Zenodo (not by CERN) because the Open Data Portal publishes the experiment-level ROOT sample, not this convenience table. This repository does not currently ship the ROOT-to-CSV extraction script; if you need to rebuild the table from the raw ROOT files rather than use the hosted extract, open an issue.

Neither CMS nor CERN endorse works produced from these data (statement from record 12102).

Do not commit `cms_Hbb.csv` to git.
