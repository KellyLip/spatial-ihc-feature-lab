# Prediction Targets and Split Strategy

## Purpose

This document defines the prediction targets used in this project and the rules used to split the data safely.

The main risk in this project is data leakage. The dataset contains many cells from the same patient/sample. If individual cells are randomly split into train and test sets, the model may see cells from the same patient in both sets. This would make the test performance look better than it really is.

Therefore, splits must be performed at the patient/sample level whenever possible.

---

## Available candidate targets

### 1. Patient/sample tumor-immune architecture class

Column/source: `patient_class.csv`

Classes:

| Numeric label | Biological label |
|---:|---|
| 0 | mixed |
| 1 | compartmentalized |
| 2 | cold |

Interpretation:

- Cold: low immune infiltration.
- Mixed: immune and tumor cells are spatially mixed.
- Compartmentalized: immune and tumor cells are spatially separated.

This is the primary target for the main spatial ML model.

---

### 2. Cell phenotype / cell type

Columns/source:

- `Group`
- `immuneGroup`
- possible mapped label from the cell-label hierarchy

This is the secondary target for a cell-level classifier.

Example task:

Marker intensities per cell → cell phenotype.

Important limitation:

This model learns whether marker expression can reproduce cell labels.

---

### 3. Tumor / non-tumor label

Column/source:

- `tumorYN`

---

## Chosen targets

### Primary model

Question:

Can engineered spatial features classify tumor-immune architecture?

Target:

`patient_class`

Unit of prediction:

patient/sample

Input features:

future ROI/sample-level spatial features, such as:

- immune/tumor cell proportions
- immune-tumor nearest-neighbor distances
- immune-tumor contact counts
- CD8-to-tumor proximity
- macrophage-to-tumor proximity
- graph/neighborhood features

Split rule:

patient/sample-level split.

---

### Secondary model

Question:

Can marker expression classify cell phenotype?

Target:

cell phenotype label

Unit of prediction:

cell

Input features:

marker intensity columns only.

Excluded from features:

- patient ID
- sample ID
- coordinates
- cell label ID
- target columns
- derived labels that directly encode the answer

Split rule:

split by patient/sample, not by random cells.

---

## Split rules

1. No patient/sample ID may appear in more than one split.
2. Train, validation, and test splits must be created from patient/sample IDs first.
3. Cell rows inherit the split from their patient/sample.
4. Class balance should be checked after splitting.
5. Because the dataset is small, cross-validation should be preferred for final reporting.
6. Never report only accuracy. Use macro F1, class-wise precision/recall, and confusion matrix.

---

## Metrics

### For patient architecture classification

Use:

- macro F1
- balanced accuracy
- class-wise precision
- class-wise recall
- confusion matrix

Reason:

The classes are imbalanced, especially the cold class.

### For cell phenotype classification

Use:

- macro F1
- weighted F1
- class-wise precision
- class-wise recall
- confusion matrix

Reason:

Cell populations are strongly imbalanced. Common cell types can dominate accuracy.

---

## Current decision

Primary target:

`patient_class`

Secondary target:

cell phenotype label

Primary split strategy:

patient/sample-level split

Main leakage rule:

No patient/sample can appear in both train and test.