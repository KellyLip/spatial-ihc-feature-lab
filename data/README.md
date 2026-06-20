# Data

This folder documents the datasets used by this repository and explains how to obtain them manually.

Raw data files are not tracked by Git. Large CSV files, TIFF images, raw image folders, and derived processed files should remain local under `data/raw/` or `data/processed/`.

## MIBI-TNBC processed data

This repository uses the processed MIBI-TNBC dataset from Angelolab.

The processed dataset is used because it contains the smallest practical set of files needed for cell-level and spatial analysis:

* per-cell marker expression values
* per-cell biological annotations
* patient/sample identifiers
* patient-level labels
* segmented cell images where each cell has a unique object ID

This avoids downloading the full raw marker-image data at the start. The raw marker images are much larger and are not required for workflows that start from already-processed cell-level expression tables and segmentation masks.

## Manual download steps

1. Go to the Angelolab MIBI data page:

   ```text
   https://www.angelolab.com/mibi-data
   ```

2. Find the dataset section:

   ```text
   Triple negative breast cancer (TNBC)
   ```

3. Click:

   ```text
   Processed data (expression values, segmented images, patient data)
   ```

4. Download the processed data package.

5. Extract the downloaded files locally.

6. Move the relevant files into the expected local paths shown below.

## Expected local paths

Organize the downloaded files as follows:

```text
data/
  raw/
    mibi_tnbc/
      cellData.csv
      patient_class.csv
      labeled_cell_masks/
        p1_labeledcellData.tiff
        p2_labeledcellData.tiff
        ...
      cell_type_label_masks/
        p1_cellType.tiff
        p2_cellType.tiff
        ...
  processed/
    mibi_cell_centroids.csv
    mibi_cellData_with_patient_class_and_centroids.csv
    mibi_cells_with_splits.csv
```

Processed files under `data/processed/` are generated locally by the pipeline in `src/data/` and are not committed to Git.

## Expected raw files

### `cellData.csv`

Cell-level table containing information for cells across patients.

Expected contents include:

* patient/sample identifier (`SampleID`)
* cell object identifier within the image (`cellLabelInImage`)
* cell size
* marker expression values
* broad cell classification (`Group`, `tumorYN`, etc.)
* immune-cell subclassification, where available

The cell object identifier links rows in the table to segmented objects in the labeled cell TIFF images.

**Note:** The raw file contains **43** SampleIDs (1–41 except 30, plus 42–44). Sample **30** has no cell rows. Samples **42–44** have cell rows but no entry in `patient_class.csv`.

### `patient_class.csv`

Headerless two-column file: `SampleID`, `patient_class`.

Patient-level labels for the three spatial immune phenotypes:

| Code | Label |
|------|-------|
| 0 | mixed |
| 1 | compartmentalized |
| 2 | cold |

This file lists **41** samples (SampleIDs 1–41).

### Labeled cell segmentation TIFFs

TIFF images where each segmented cell has a unique integer ID.

These images are used to extract cell centroids and connect tabular cell-level data back to the physical layout of cells in the tissue image. The repository expects **41** mask files (`p1`–`p41`).

### Cell-type label TIFFs

TIFF images where each cell is assigned a biological cell-type label.

These files are useful for visual inspection, sanity checks, and validating that cell-type annotations align with the segmentation masks. They are not required for the current tabular/spatial pipeline.

## Processed outputs

After raw data is in place, run from the repository root:

```bash
python src/data/extract_mibi_centroids.py
python src/data/make_mibi_spatial_table.py
python src/data/make_mibi_splits.py
```

| File | Description |
|------|-------------|
| `mibi_cell_centroids.csv` | Centroids and mask areas per `(sample_id, cell_label)` from labeled masks |
| `mibi_cellData_with_patient_class_and_centroids.csv` | Master analysis table: expression, annotations, patient class, and coordinates |
| `mibi_cells_with_splits.csv` | Master table plus a `split` column (`train` / `val` / `test`) assigned by `SampleID` |

The master table retains **40** samples: all labeled samples except sample 30 (no `cellData` rows) and samples 42–44 (excluded explicitly; no patient-class label).

Label mappings used during analysis are defined in `src/data/mibi_constants.py`.

### Train/validation/test splits

Created by `src/data/make_mibi_splits.py` using `src/utils/splits.py`.

* **Group column:** `SampleID` — all cells from a sample share the same split
* **Stratification target:** `patient_class` (constant within each sample)
* **Default sizes:** 20% test, 20% validation, remainder train (`random_state=42`)

Split balance is summarized in `reports/tables/split_summary.csv`. Target definitions and leakage rules are documented in [`reports/targets_and_splits.md`](../reports/targets_and_splits.md).

You can also inspect the split interactively in `notebooks/04_split_summary.ipynb`.

## Data QC

Overlap between `cellData.csv`, centroids, and masks is checked in `notebooks/02_centroids_overlap_check.ipynb`. Summary tables are written to `reports/tables/`, including:

* `cell_mask_overlap_summary.csv` — row-level overlap counts
* `cell_rows_without_centroids_by_sample.csv` — `cellData` rows missing a mask match (samples 42–44)
* `mask_objects_without_cellData_by_sample.csv` — mask objects not present in `cellData.csv`
* `mask_only_samples_no_cellData.csv` — samples with a mask but zero `cellData` rows (sample 30)

Run `python src/data/check_mibi_master_table.py` after building the master table to regenerate QC tables and summary figures in `reports/tables/` and `reports/figures/`.

## Files not required initially

The full raw marker-image data is not required for the initial processed-data workflow.

The processed data already contains per-cell marker expression values, so the raw marker images should only be downloaded if they are needed for:

* visual validation
* figure generation
* comparison between raw marker signal and processed cell-level values
* image-level analysis
