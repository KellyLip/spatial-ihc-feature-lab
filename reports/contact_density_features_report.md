# Contact, density, and composition features

## Purpose

This feature set describes tumor–immune organization at the sample level using cell coordinates and broad cell-phenotype labels from the MIBI-TNBC dataset.

Each row in the final feature table represents one sample. The features are intended as inputs for classifying tumor–immune spatial architecture (`mixed`, `compartmentalized`, or `cold`).

## Radius-neighbor features

Radius-neighbor calculations were performed at 20, 50, and 100 micrometers.

A radius neighbor means that the centroid of one cell lies within the selected distance of another cell. It should not be interpreted as proof of direct membrane contact.

The following directional population pairs were measured:

- immune cells to tumor cells
- tumor cells to immune cells
- CD8 T cells to tumor cells
- macrophages to tumor cells
- B cells to tumor cells

For each population pair and radius, the feature table includes:

- total source–target neighbor pairs
- fraction of source cells with at least one target neighbor
- mean target neighbors per source cell

Directional features are kept separate because immune-to-tumor and tumor-to-immune measurements answer different biological questions.

## Cell composition features

The table contains counts, proportions, and approximate field-of-view densities for:

- immune cells
- tumor cells
- mesenchymal-like cells
- endothelial cells
- unidentified cells
- CD8 T cells
- CD4 T cells
- regulatory T cells
- B cells
- macrophages

It also contains selected ratios such as:

- immune-to-tumor cell count ratio
- CD8-to-CD4 T-cell count ratio
- macrophage-to-T-cell count ratio

## Density limitation

Density is calculated using the approximate full field-of-view area rather than the exact tissue-covered area.

The resulting columns are therefore named `density_per_fov_mm2`. Empty regions and variation in tissue coverage may affect these values. They should not be described as exact tissue densities.

## Distance features

Previously generated nearest-neighbor distance summaries were merged into the feature table.

Only micrometer columns were retained. Pixel-distance columns were excluded because they contain the same information in another unit and would create perfectly correlated duplicate features.

## Missing values

The final table contains 158 missing feature values.

These values are fully explained by absent source populations:

- 11 samples contain no B cells
- 2 samples contain no CD8 T cells

Distance summaries, mean neighbor counts, and fractions per source cell are undefined when the source population is absent.

These values were not replaced with zero because:

- zero distance represents overlapping or immediately adjacent centroids
- a missing value represents an unavailable measurement because the population is absent

Missing-value handling will be performed later inside a leakage-safe preprocessing pipeline fitted only on training data.

## Quality control

The final table contains:

- 197,678 source cell rows
- 40 unique samples
- 40 sample-level feature rows
- 107 columns

Automated tests verify:

- known radius-neighbor calculations
- distance calculations with known answers
- pixel-to-micrometer conversion
- behavior with empty source or target populations
- summary calculations
- increasing contact counts with increasing radius

The complete test suite passes with 11 tests.

## Outputs

| Path | Description |
|---|---|
| `data/processed/mibi_roi_features.csv` | Final sample-level spatial feature table |
| `reports/tables/mibi_roi_features_summary.csv` | Reporting copy of the feature table |
| `reports/figures/mibi_roi_feature_heatmap.png` | Standardized feature heatmap across samples |
| `src/features/contact_features.py` | Reusable radius-neighbor functions |
| `src/features/build_mibi_roi_features.py` | Sample-level feature-building pipeline |
| `tests/test_contact_features.py` | Synthetic-coordinate contact tests |