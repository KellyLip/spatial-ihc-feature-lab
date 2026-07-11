## Immune–tumor nearest-neighbor distance features

We built sample-level spatial features measuring how close CD8 T cells, macrophages, and B cells sit to the nearest tumor cell. Distances use cell centroids from the cleaned MIBI master table and are summarized per sample (mean, median, and percentiles) in pixels and approximate micrometers (`0.39 µm/pixel`).

Across all 40 samples, median CD8→tumor and macrophage→tumor distances are roughly **11 µm**. Compartmentalized samples tend to have larger median distances (about **14–15 µm**) than mixed samples (about **10 µm**), which matches the expected spatial separation of immune and tumor compartments. Cold samples often have few or no CD8/B cells; when immune cells are present, their nearest-tumor distances are typically short, but those features are sparse and less stable.

B-cell features are missing for 11 samples (no B cells), and CD8 features are missing for 2 samples. Macrophages are present in every sample. These features are intended as inputs for later `patient_class` models, not as a finished predictive result on their own.

## How to regenerate

```bash
python src/features/build_mibi_distance_features.py
```

Requires `data/processed/mibi_cells_with_splits.csv`.

## Outputs

| Path | Description |
|------|-------------|
| `data/processed/mibi_distance_features.csv` | One row per sample with distance summaries |
| `data/processed/mibi_cell_to_tumor_distances.csv` | One row per immune cell → nearest tumor distance |
| `reports/tables/mibi_distance_features_summary.csv` | Reporting copy of the sample-level table |
| `reports/figures/cd8_to_tumor_distance_by_sample.png` | CD8→tumor distance boxplots by sample |
| `reports/figures/macrophage_to_tumor_distance_by_sample.png` | Macrophage→tumor distance boxplots by sample |
| `reports/figures/b_cell_to_tumor_distance_by_sample.png` | B cell→tumor distance boxplots by sample |

Population definitions are in `src/features/spatial_distances.py` (`tumorYN` / `immuneGroup` filters).
