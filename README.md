# Spatial IHC Feature Lab

A reproducible computational pathology repository demonstrating how to engineer spatial features from multiplex imaging data, train interpretable machine learning models, and evaluate predictive markers.

## Project Goals

* **Primary goal:** Spatial feature engineering and classical machine learning. The core workflow uses the MIBI-TNBC dataset to convert cell coordinates, phenotypes, and marker intensities into engineered biological features (distances, neighborhood densities, graphs) for interpretable prediction models.
* **Secondary goal:** Computer vision module. The MIHIC dataset supports a smaller image-classification pipeline for classical feature extraction and modest CNN transfer learning.

*Disclaimer: This is a portfolio and learning project, not a clinical product.*

For the rationale behind dataset choices, see [`reports/dataset_rationale.md`](reports/dataset_rationale.md).

For prediction targets and split strategy, see [`reports/targets_and_splits.md`](reports/targets_and_splits.md).

## Repository Layout

```text
spatial-ihc-feature-lab/
  data/                  Raw and processed data (local only; see data/README.md)
  notebooks/             Exploratory analysis and QC notebooks
  reports/
    figures/             Generated plots from scripts and notebooks
    tables/              QC summaries, EDA tables, and split summaries
    dataset_rationale.md Project note on MIBI vs MIHIC strategy
    targets_and_splits.md Prediction targets and leakage-safe split rules
  src/
    data/                MIBI-TNBC data preparation and QC scripts
    utils/               Shared utilities (train/val/test splits)
  environment.yml        Conda environment definition
```

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate spatial-ihc
```

Download the MIBI-TNBC processed data locally and place files under `data/raw/mibi_tnbc/` as described in [`data/README.md`](data/README.md). Raw CSVs, TIFF masks, and processed tables are gitignored.

## MIBI-TNBC Workflow

Run the scripts in order after raw data is in place:

```bash
python src/data/extract_mibi_centroids.py
python src/data/make_mibi_spatial_table.py
python src/data/check_mibi_master_table.py
python src/data/make_mibi_splits.py
```

| Step | Script | Output |
|------|--------|--------|
| 1 | `extract_mibi_centroids.py` | `data/processed/mibi_cell_centroids.csv` |
| 2 | `make_mibi_spatial_table.py` | `data/processed/mibi_cellData_with_patient_class_and_centroids.csv` |
| 3 | `check_mibi_master_table.py` | QC tables in `reports/tables/` and summary figures in `reports/figures/` |
| 4 | `make_mibi_splits.py` | `data/processed/mibi_cells_with_splits.csv` and `reports/tables/split_summary.csv` |

Shared label mappings (`patient_class`, cell groups, immune groups) live in `src/data/mibi_constants.py`. Leakage-safe split utilities live in `src/utils/splits.py`.

### Master table

The analysis table merges:

* per-cell marker expression and annotations from `cellData.csv`
* patient spatial immune phenotype from `patient_class.csv`
* spatial coordinates from labeled segmentation masks

**Sample counts:**

* `patient_class.csv` lists **41** SampleIDs (1–41)
* The master table contains **40** samples with cell-level data
* **Sample 30** has a segmentation mask but **zero rows in `cellData.csv`**, so it is excluded from the merged table
* **Samples 42–44** appear in `cellData.csv` but have no patient-class label or TIFF masks and are dropped during merge

Patient phenotype codes:

| Code | Label |
|------|-------|
| 0 | mixed |
| 1 | compartmentalized |
| 2 | cold |

### Targets and splits

The primary modeling target is **`patient_class`** (sample-level spatial immune architecture). A secondary target is **cell phenotype** (`Group`, `immuneGroup`).

Splits are performed at the **`SampleID` level** so all cells from the same sample stay in one split. The default train/validation/test assignment uses stratified group splitting on `patient_class` (20% test, 20% validation, `random_state=42`).

Current split sizes (40 samples → 24 train / 8 val / 8 test):

| Split | Cells | Samples | mixed | compartmentalized | cold |
|-------|------:|--------:|------:|------------------:|-----:|
| train | 114,670 | 24 | 11 | 9 | 4 |
| val | 38,311 | 8 | 4 | 3 | 1 |
| test | 44,697 | 8 | 4 | 3 | 1 |

See [`reports/targets_and_splits.md`](reports/targets_and_splits.md) for target definitions, leakage rules, and planned evaluation metrics.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `00_dataset_checks.ipynb` | Early dataset inspection and cleaning experiments |
| `01_mibi_spatial_eda.ipynb` | Raw MIBI TIFF channel inspection |
| `02_centroids_overlap_check.ipynb` | QC overlap between `cellData.csv`, centroids, and masks; includes mask-only sample checks |
| `03_mibi_comprehensive_eda.ipynb` | Extensive EDA on the master table (phenotypes, markers, spatial layout) |
| `04_split_summary.ipynb` | Inspect and validate the stratified sample-level train/val/test split |

Run notebooks from the `notebooks/` directory or ensure the project root resolves correctly so paths to `data/` and `src/` work.

## Generated Reports

**Figures** (`reports/figures/`): cells per sample, patient-class and cell-type distributions, tumor status, spatial sanity checks, and notebook-specific EDA plots under `reports/figures/eda_notebook/`.

**Tables** (`reports/tables/`): missing-value summaries, per-sample summaries, cell–mask overlap QC, and `split_summary.csv`.

## Current Status

Completed for MIBI-TNBC:

* Centroid extraction from labeled cell masks
* Master table construction with patient labels and spatial coordinates
* Overlap QC between tabular cell data and segmentation masks
* Summary QC figures and comprehensive EDA notebook
* Prediction target definition and leakage-safe split strategy (`reports/targets_and_splits.md`)
* Stratified sample-level train/val/test splits (`make_mibi_splits.py`, `src/utils/splits.py`)

Planned next steps:

* Spatial feature engineering (distances, neighborhood densities, graphs)
* Interpretable ML models for `patient_class` prediction using sample-level spatial features
