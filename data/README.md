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
```

The exact TIFF file names may vary depending on the downloaded package. If the names differ, keep the original names and update the data-loading code accordingly rather than renaming files manually.

## Expected files

### `cellData.csv`

Cell-level table containing information for cells across patients.

Expected contents include:

* patient/sample identifier
* cell object identifier within the image
* cell size
* marker expression values
* broad cell classification
* immune-cell subclassification, where available

The cell object identifier links rows in the table to segmented objects in the labeled cell TIFF images.

### `patient_class.csv`

Patient-level class labels.

This file maps each patient/sample to a class label. These labels are useful for patient-level or sample-level modeling tasks.

### Labeled cell segmentation TIFFs

TIFF images where each segmented cell has a unique integer ID.

These images are used to connect tabular cell-level data back to the physical layout of cells in the tissue image.

### Cell-type label TIFFs

TIFF images where each cell is assigned a biological cell-type label.

These files are useful for visual inspection, sanity checks, and validating that cell-type annotations align with the segmentation masks.

## Files not required initially

The full raw marker-image data is not required for the initial processed-data workflow.

The processed data already contains per-cell marker expression values, so the raw marker images should only be downloaded if they are needed for:

* visual validation
* figure generation
* comparison between raw marker signal and processed cell-level values
* image-level analysis

## Git policy

Do not commit raw data or large generated files.

The following paths and file types should remain ignored by Git:

```text
data/raw/
data/processed/
*.tif
*.tiff
*.svs
*.ndpi
*.qptiff
*.pt
*.pth
*.h5
*.pkl
```

Only lightweight documentation, small example files, and code should be tracked in Git.
