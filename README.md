# Spatial IHC Feature Lab

A reproducible computational pathology repository demonstrating how to engineer spatial features from multiplex imaging data, train interpretable machine learning models, and evaluate predictive markers.

## Project Goals

* **Primary goal:** Spatial feature engineering and classical machine learning. The core workflow uses the MIBI-TNBC dataset to convert cell coordinates, phenotypes, and marker intensities into engineered biological features (distances, neighborhood densities, graphs) for interpretable prediction models.
* **Secondary goal:** Computer vision module. The MIHIC dataset supports a smaller image-classification pipeline for classical feature extraction and modest CNN transfer learning.

*Disclaimer: This is a portfolio and learning project, not a clinical product.*

For the rationale behind dataset choices, see [`reports/dataset_rationale.md`](reports/dataset_rationale.md).

For prediction targets and split strategy, see [`reports/targets_and_splits.md`](reports/targets_and_splits.md).

For immune–tumor distance feature results, see [`reports/distance_features_report.md`](reports/distance_features_report.md).

For contact, density, and composition ROI features, see [`reports/contact_density_features_report.md`](reports/contact_density_features_report.md).

## Repository Layout

```text
spatial-ihc-feature-lab/
  data/                  Raw and processed data (local only; see data/README.md)
  notebooks/             Exploratory analysis and QC notebooks
  reports/
    figures/             Generated plots from scripts and notebooks
    metrics/             Model evaluation metrics (CSV)
    tables/              QC summaries, EDA tables, and split summaries
    cell_classifier_report.md  Summary of cell phenotype baseline results
    distance_features_report.md  Summary of immune–tumor distance features
    contact_density_features_report.md  Contact, density, and composition ROI features
    dataset_rationale.md Project note on MIBI vs MIHIC strategy
    targets_and_splits.md Prediction targets and leakage-safe split rules
  src/
    data/                MIBI-TNBC data preparation and QC scripts
    features/            Spatial feature engineering (distances, contacts, densities)
    models/              Supervised ML training scripts
    utils/               Shared utilities (train/val/test splits)
  tests/                 Unit tests for spatial feature helpers
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
python src/models/train_mibi_cell_classifier.py
python src/features/build_mibi_distance_features.py
python src/features/build_mibi_roi_features.py
```

| Step | Script | Output |
|------|--------|--------|
| 1 | `extract_mibi_centroids.py` | `data/processed/mibi_cell_centroids.csv` |
| 2 | `make_mibi_spatial_table.py` | `data/processed/mibi_cellData_with_patient_class_and_centroids.csv` |
| 3 | `check_mibi_master_table.py` | QC tables in `reports/tables/` and summary figures in `reports/figures/` |
| 4 | `make_mibi_splits.py` | `data/processed/mibi_cells_with_splits.csv` and `reports/tables/split_summary.csv` |
| 5 | `train_mibi_cell_classifier.py` | Baseline metrics in `reports/metrics/`, confusion matrices in `reports/figures/`, top confusions in `reports/tables/` |
| 6 | `build_mibi_distance_features.py` | Sample-level distance features, cell-level distances, summary table, and per-sample boxplots |
| 7 | `build_mibi_roi_features.py` | Sample-level ROI feature table (contacts, composition, densities + merged distances), summary table, and feature heatmap |

Shared label mappings (`patient_class`, cell groups, immune groups) live in `src/data/mibi_constants.py`. Leakage-safe split utilities live in `src/utils/splits.py`. Nearest-neighbor distance helpers live in `src/features/spatial_distances.py`. Radius-neighbor contact helpers live in `src/features/contact_features.py`.

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

### Cell phenotype classifier

The secondary modeling target is **cell phenotype** (`Group`): predict broad MIBI cell groups from per-cell marker expression only. This is a tabular ML baseline and pipeline validation step, not independent biological discovery. Because the labels are marker-derived, strong performance mainly confirms that the cleaned table, splits, and features are coherent.

Run after splits are created:

```bash
python src/models/train_mibi_cell_classifier.py
```

The script trains four baselines on the train split and evaluates on the held-out test split:

| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|-------|---------:|------------------:|---------:|--------------:|
| Dummy most frequent | 0.502 | 0.167 | 0.111 | 0.335 |
| Logistic regression | 0.930 | 0.956 | 0.790 | 0.936 |
| Random forest | 0.975 | 0.964 | 0.954 | 0.975 |
| HistGradientBoosting | **0.992** | **0.978** | **0.972** | **0.992** |

Cell groups (`Group`):

| Code | Label |
|------|-------|
| 1 | Unidentified |
| 2 | Immune |
| 3 | Endothelial |
| 4 | Mesenchymal-like |
| 5 | Tumor |
| 6 | Keratin-positive tumor |

Outputs:

* `reports/metrics/cell_classifier_baseline.csv` — model comparison metrics
* `reports/metrics/cell_classifier_classification_report.csv` — per-class precision/recall/F1
* `reports/tables/cell_classifier_top_confusions.csv` — largest off-diagonal confusion pairs
* `reports/figures/cell_classifier_confusion_matrix_*.png` — confusion matrix plots per model

See [`reports/cell_classifier_report.md`](reports/cell_classifier_report.md) for interpretation of remaining errors and model comparison notes.

### Immune–tumor distance features

After splits exist, build sample-level nearest-neighbor distances from immune populations to tumor cells:

```bash
python src/features/build_mibi_distance_features.py
```

For each of the 40 samples, the script measures distance from every CD8 T cell, macrophage, and B cell to the nearest tumor cell (`tumorYN == 1`), then aggregates mean, median, and percentiles (p10–p90) in pixels and approximate micrometers (`0.39 µm/pixel`).

Median CD8→tumor distance is about **11 µm** overall; compartmentalized samples sit farther from tumor (median ≈ 14 µm) than mixed (≈ 10 µm) or cold samples with any CD8 cells (≈ 9 µm). Macrophage→tumor distances follow the same pattern. B cells are sparse in many samples (11 samples have none), so those features are often missing and should be used carefully downstream.

Outputs:

* `data/processed/mibi_distance_features.csv` — one feature row per sample
* `data/processed/mibi_cell_to_tumor_distances.csv` — one row per immune cell distance
* `reports/tables/mibi_distance_features_summary.csv` — copy of the sample-level table for reporting
* `reports/figures/cd8_to_tumor_distance_by_sample.png`
* `reports/figures/macrophage_to_tumor_distance_by_sample.png`
* `reports/figures/b_cell_to_tumor_distance_by_sample.png`

See [`reports/distance_features_report.md`](reports/distance_features_report.md) for a short interpretation of these results.

### Contact, density, and composition ROI features

After distance features exist, build the combined sample-level ROI feature table:

```bash
python src/features/build_mibi_roi_features.py
```

For each of the 40 samples, the script computes:

* **Radius-neighbor contacts** at 20, 50, and 100 µm for directional pairs (immune↔tumor, CD8→tumor, macrophage→tumor, B cell→tumor), including total neighbor pairs, fraction of source cells with a neighbor, and mean neighbors per source cell
* **Cell composition** counts, proportions, FOV-approximate densities, and selected ratios (immune/tumor, CD8/CD4, macrophage/T cell)
* **Merged micrometer distance summaries** from `mibi_distance_features.csv` (pixel distance columns are dropped to avoid duplicate correlated features)

A radius neighbor means cell centroids fall within the chosen distance; it is not proof of membrane contact. Densities use approximate full FOV area (`density_per_fov_mm2`), not exact tissue-covered area. Missing values occur when a source population is absent (for example, no B cells in 11 samples) and are left as missing rather than zero.

Outputs:

* `data/processed/mibi_roi_features.csv` — final sample-level spatial feature table (107 columns)
* `reports/tables/mibi_roi_features_summary.csv` — reporting copy of the feature table
* `reports/figures/mibi_roi_feature_heatmap.png` — standardized feature heatmap across samples

See [`reports/contact_density_features_report.md`](reports/contact_density_features_report.md) for feature definitions, missing-value notes, and QC checks. Contact helpers are covered by `tests/test_contact_features.py`.

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

**Figures** (`reports/figures/`): cells per sample, patient-class and cell-type distributions, tumor status, spatial sanity checks, cell-classifier confusion matrices, immune–tumor distance boxplots by sample, ROI feature heatmap, and notebook-specific EDA plots under `reports/figures/eda_notebook/`.

**Metrics** (`reports/metrics/`): cell phenotype classifier baseline and per-class classification reports.

**Tables** (`reports/tables/`): missing-value summaries, per-sample summaries, cell–mask overlap QC, `split_summary.csv`, cell-classifier top confusions, `mibi_distance_features_summary.csv`, and `mibi_roi_features_summary.csv`.

## Current Status

Completed for MIBI-TNBC:

* Centroid extraction from labeled cell masks
* Master table construction with patient labels and spatial coordinates
* Overlap QC between tabular cell data and segmentation masks
* Summary QC figures and comprehensive EDA notebook
* Prediction target definition and leakage-safe split strategy (`reports/targets_and_splits.md`)
* Stratified sample-level train/val/test splits (`make_mibi_splits.py`, `src/utils/splits.py`)
* Cell phenotype baseline classifier from marker expression (`train_mibi_cell_classifier.py`)
* Nearest-neighbor immune–tumor distance features (`spatial_distances.py`, `build_mibi_distance_features.py`)
* Contact, density, and composition ROI features (`contact_features.py`, `build_mibi_roi_features.py`)

Planned next steps:

* Additional spatial features (graphs, richer neighborhood descriptors)
* Interpretable ML models for `patient_class` prediction using sample-level spatial features
