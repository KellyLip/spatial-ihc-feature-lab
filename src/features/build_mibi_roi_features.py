"""Build sample-level spatial features for the MIBI-TNBC dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.contact_features import build_radius_neighbor_features
from src.features.spatial_distances import (
    MIBI_PIXEL_SIZE_UM,
    get_cells,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

INPUT_PATH = PROCESSED_DIR / "mibi_cells_with_splits.csv"
DISTANCE_FEATURES_PATH = PROCESSED_DIR / "mibi_distance_features.csv"

OUTPUT_PATH = PROCESSED_DIR / "mibi_roi_features.csv"
REPORT_TABLE_PATH = TABLES_DIR / "mibi_roi_features_summary.csv"
HEATMAP_PATH = FIGURES_DIR / "mibi_roi_feature_heatmap.png"


# ---------------------------------------------------------------------
# Spatial configuration
# ---------------------------------------------------------------------

RADII_UM = (20.0, 50.0, 100.0)

# Original MIBI fields are approximately 2048 x 2048 pixels.
FOV_WIDTH_PX = 2048
FOV_HEIGHT_PX = 2048

FOV_WIDTH_MM = FOV_WIDTH_PX * MIBI_PIXEL_SIZE_UM / 1000
FOV_HEIGHT_MM = FOV_HEIGHT_PX * MIBI_PIXEL_SIZE_UM / 1000
FOV_AREA_MM2 = FOV_WIDTH_MM * FOV_HEIGHT_MM


# These statistics are retained in the final model table.
#
# Other summaries remain available in contact_features.py but are omitted
# here to reduce unnecessary duplicate and highly correlated columns.
CONTACT_STATS_TO_KEEP = (
    "total_neighbor_pairs",
    "fraction_source_with_neighbor",
    "mean_neighbors_per_source",
)


PAIR_DEFINITIONS = {
    "immune_to_tumor": ("immune", "tumor"),
    "tumor_to_immune": ("tumor", "immune"),
    "cd8_to_tumor": ("cd8_t", "tumor"),
    "macrophage_to_tumor": ("macrophage", "tumor"),
    "b_cell_to_tumor": ("b_cell", "tumor"),
}


def get_single_sample_value(
    sample_df: pd.DataFrame,
    column: str,
) -> object:
    """
    Return the one non-missing value assigned to a sample.

    Columns such as patient_class and split must be constant inside
    each SampleID.
    """
    if column not in sample_df.columns:
        raise KeyError(f"Missing required column: {column}")

    values = sample_df[column].dropna().unique()

    if len(values) == 0:
        return np.nan

    if len(values) > 1:
        sample_id = sample_df["SampleID"].iloc[0]

        raise ValueError(
            f"Column {column!r} has multiple values inside "
            f"SampleID {sample_id}: {values.tolist()}"
        )

    return values[0]


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """
    Divide two values without creating infinite results.
    """
    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def select_population(
    sample_df: pd.DataFrame,
    population: str,
) -> pd.DataFrame:
    """
    Select a named cell population and retain valid coordinates.
    """
    if population == "immune":
        selected = sample_df[sample_df["Group"] == 2].copy()

    elif population == "tumor":
        selected = get_cells(
            sample_df,
            cell_type="tumor",
        )

    elif population in {"cd8_t", "macrophage", "b_cell"}:
        selected = get_cells(
            sample_df,
            cell_type=population,
        )

    else:
        raise ValueError(f"Unknown population: {population}")

    selected = selected.dropna(
        subset=["centroid_x", "centroid_y"]
    ).copy()

    return selected


def add_contact_features(
    feature_row: dict[str, object],
    *,
    prefix: str,
    source_cells: pd.DataFrame,
    target_cells: pd.DataFrame,
) -> None:
    """
    Add selected radius-contact summaries to one sample feature row.
    """
    contact_features = build_radius_neighbor_features(
        source_cells,
        target_cells,
        radii_um=RADII_UM,
        pixel_size_um=MIBI_PIXEL_SIZE_UM,
    )

    for feature_name, value in contact_features.items():
        if feature_name.endswith(CONTACT_STATS_TO_KEEP):
            feature_row[f"{prefix}_{feature_name}"] = value


def build_sample_feature_row(
    sample_id: int,
    sample_df: pd.DataFrame,
) -> dict[str, object]:
    """
    Build one feature row for one MIBI sample.
    """
    group_values = pd.to_numeric(
        sample_df["Group"],
        errors="coerce",
    )

    immune_group_values = pd.to_numeric(
        sample_df["immuneGroup"],
        errors="coerce",
    )

    n_total = int(len(sample_df))

    # Broad mutually exclusive cell groups.
    n_unidentified = int(group_values.eq(1).sum())
    n_immune = int(group_values.eq(2).sum())
    n_endothelial = int(group_values.eq(3).sum())
    n_mesenchymal_like = int(group_values.eq(4).sum())
    n_tumor = int(group_values.isin([5, 6]).sum())

    n_named_groups = (
        n_unidentified
        + n_immune
        + n_endothelial
        + n_mesenchymal_like
        + n_tumor
    )

    n_other_or_missing_group = n_total - n_named_groups

    # Selected immune populations.
    n_tregs = int(immune_group_values.eq(1).sum())
    n_cd4_t = int(immune_group_values.eq(2).sum())
    n_cd8_t = int(immune_group_values.eq(3).sum())
    n_cd3_t = int(immune_group_values.eq(4).sum())
    n_b_cells = int(immune_group_values.eq(6).sum())
    n_macrophages = int(immune_group_values.eq(8).sum())

    n_t_cells = n_tregs + n_cd4_t + n_cd8_t + n_cd3_t

    feature_row: dict[str, object] = {
        "SampleID": int(sample_id),
        "patient_class": get_single_sample_value(
            sample_df,
            "patient_class",
        ),
        "split": get_single_sample_value(
            sample_df,
            "split",
        ),
        "fov_area_mm2": FOV_AREA_MM2,
        "n_total_cells": n_total,
        "n_unidentified_cells": n_unidentified,
        "n_immune_cells": n_immune,
        "n_endothelial_cells": n_endothelial,
        "n_mesenchymal_like_cells": n_mesenchymal_like,
        "n_tumor_cells": n_tumor,
        "n_other_or_missing_group_cells": n_other_or_missing_group,
        "n_treg_cells": n_tregs,
        "n_cd4_t_cells": n_cd4_t,
        "n_cd8_t_cells": n_cd8_t,
        "n_cd3_t_cells": n_cd3_t,
        "n_t_cells": n_t_cells,
        "n_b_cells": n_b_cells,
        "n_macrophages": n_macrophages,
    }

    # Broad cell proportions.
    broad_counts = {
        "unidentified": n_unidentified,
        "immune": n_immune,
        "endothelial": n_endothelial,
        "mesenchymal_like": n_mesenchymal_like,
        "tumor": n_tumor,
        "other_or_missing_group": n_other_or_missing_group,
    }

    for population_name, count in broad_counts.items():
        feature_row[
            f"{population_name}_proportion"
        ] = safe_ratio(count, n_total)

        feature_row[
            f"{population_name}_density_per_fov_mm2"
        ] = safe_ratio(count, FOV_AREA_MM2)

    feature_row["total_cell_density_per_fov_mm2"] = safe_ratio(
        n_total,
        FOV_AREA_MM2,
    )

    # Selected immune-cell proportions and densities.
    selected_immune_counts = {
        "treg": n_tregs,
        "cd4_t": n_cd4_t,
        "cd8_t": n_cd8_t,
        "b_cell": n_b_cells,
        "macrophage": n_macrophages,
    }

    for population_name, count in selected_immune_counts.items():
        feature_row[
            f"{population_name}_proportion"
        ] = safe_ratio(count, n_total)

        feature_row[
            f"{population_name}_density_per_fov_mm2"
        ] = safe_ratio(count, FOV_AREA_MM2)

    # Biologically interpretable count ratios.
    feature_row["immune_to_tumor_count_ratio"] = safe_ratio(
        n_immune,
        n_tumor,
    )

    feature_row["cd8_to_cd4_count_ratio"] = safe_ratio(
        n_cd8_t,
        n_cd4_t,
    )

    feature_row["macrophage_to_t_cell_count_ratio"] = safe_ratio(
        n_macrophages,
        n_t_cells,
    )

    # Build each population only once for this sample.
    populations = {
        population_name: select_population(
            sample_df,
            population_name,
        )
        for population_name in {
            population
            for pair in PAIR_DEFINITIONS.values()
            for population in pair
        }
    }

    for prefix, (
        source_population,
        target_population,
    ) in PAIR_DEFINITIONS.items():
        add_contact_features(
            feature_row,
            prefix=prefix,
            source_cells=populations[source_population],
            target_cells=populations[target_population],
        )

    return feature_row


def build_feature_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build exactly one feature row per SampleID.
    """
    rows: list[dict[str, object]] = []

    for sample_id, sample_df in df.groupby(
        "SampleID",
        sort=True,
    ):
        print(f"Processing SampleID {sample_id}")

        rows.append(
            build_sample_feature_row(
                sample_id=int(sample_id),
                sample_df=sample_df,
            )
        )

    return pd.DataFrame(rows)


def merge_distance_features(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the previously generated nearest-neighbor distance features.

    Only micrometer columns are retained. Pixel columns would contain
    the same information in another unit and would create perfect
    correlations.
    """
    if not DISTANCE_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing distance feature table: "
            f"{DISTANCE_FEATURES_PATH}\n"
            "Run:\n"
            "python -m src.features.build_mibi_distance_features"
        )

    distance_df = pd.read_csv(DISTANCE_FEATURES_PATH)

    distance_columns = [
        column
        for column in distance_df.columns
        if column.endswith("_um")
    ]

    distance_subset = distance_df[
        ["SampleID", *distance_columns]
    ].copy()

    merged = feature_df.merge(
        distance_subset,
        on="SampleID",
        how="left",
        validate="one_to_one",
    )

    return merged


def check_radius_monotonicity(
    feature_df: pd.DataFrame,
) -> None:
    """
    Check that larger radii never produce fewer contacts.

    For example, every cell within 20 µm must also be within 50 µm.
    """
    statistics = (
        "total_neighbor_pairs",
        "fraction_source_with_neighbor",
        "mean_neighbors_per_source",
    )

    for pair_name in PAIR_DEFINITIONS:
        for statistic in statistics:
            columns = [
                (
                    f"{pair_name}_radius_20um_"
                    f"{statistic}"
                ),
                (
                    f"{pair_name}_radius_50um_"
                    f"{statistic}"
                ),
                (
                    f"{pair_name}_radius_100um_"
                    f"{statistic}"
                ),
            ]

            values = feature_df[columns]

            invalid = (
                (values[columns[0]] > values[columns[1]])
                | (values[columns[1]] > values[columns[2]])
            )

            if invalid.any():
                invalid_samples = feature_df.loc[
                    invalid,
                    "SampleID",
                ].tolist()

                raise ValueError(
                    f"Radius monotonicity failed for "
                    f"{pair_name}, {statistic}. "
                    f"Samples: {invalid_samples}"
                )


def run_qc(
    input_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    """
    Validate the final sample-level feature table.
    """
    expected_samples = input_df["SampleID"].nunique()

    if len(feature_df) != expected_samples:
        raise ValueError(
            "Feature table must contain one row per sample. "
            f"Expected {expected_samples}, found {len(feature_df)}."
        )

    if feature_df["SampleID"].duplicated().any():
        duplicated = feature_df.loc[
            feature_df["SampleID"].duplicated(),
            "SampleID",
        ].tolist()

        raise ValueError(
            f"Duplicate SampleID values: {duplicated}"
        )

    numeric_df = feature_df.select_dtypes(
        include=[np.number]
    )

    if np.isinf(numeric_df.to_numpy()).any():
        raise ValueError(
            "Infinite values were found in the feature table."
        )

    fraction_columns = [
        column
        for column in feature_df.columns
        if (
            column.endswith("_proportion")
            or column.endswith(
                "_fraction_source_with_neighbor"
            )
        )
    ]

    for column in fraction_columns:
        valid_values = feature_df[column].dropna()

        outside_range = ~valid_values.between(0, 1)

        if outside_range.any():
            raise ValueError(
                f"Values outside [0, 1] found in {column}."
            )

    check_radius_monotonicity(feature_df)

    print("\nQC summary")
    print("Input cell rows:", len(input_df))
    print("Unique input samples:", expected_samples)
    print("Feature-table rows:", len(feature_df))
    print("Feature-table columns:", feature_df.shape[1])
    print(
        "Missing feature values:",
        int(feature_df.isna().sum().sum()),
    )


def plot_feature_heatmap(
    feature_df: pd.DataFrame,
) -> None:
    """
    Plot a standardized subset of interpretable spatial features.
    """
    heatmap_columns = [
        "immune_proportion",
        "tumor_proportion",
        "mesenchymal_like_proportion",
        "cd8_t_proportion",
        "macrophage_proportion",
        "immune_to_tumor_count_ratio",
        (
            "immune_to_tumor_radius_20um_"
            "fraction_source_with_neighbor"
        ),
        (
            "immune_to_tumor_radius_50um_"
            "fraction_source_with_neighbor"
        ),
        (
            "immune_to_tumor_radius_100um_"
            "fraction_source_with_neighbor"
        ),
        (
            "tumor_to_immune_radius_20um_"
            "fraction_source_with_neighbor"
        ),
        (
            "tumor_to_immune_radius_50um_"
            "fraction_source_with_neighbor"
        ),
        (
            "tumor_to_immune_radius_100um_"
            "fraction_source_with_neighbor"
        ),
        "cd8_to_tumor_median_um",
        "macrophage_to_tumor_median_um",
        "b_cell_to_tumor_median_um",
    ]

    missing_columns = [
        column
        for column in heatmap_columns
        if column not in feature_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Heatmap columns are missing: {missing_columns}"
        )

    plot_df = (
        feature_df
        .set_index("SampleID")[heatmap_columns]
        .copy()
    )

    means = plot_df.mean(axis=0)
    standard_deviations = plot_df.std(axis=0).replace(0, 1)

    standardized = (
        plot_df - means
    ) / standard_deviations

    matrix = np.ma.masked_invalid(
        standardized.to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(15, 12))

    image = ax.imshow(
        matrix,
        aspect="auto",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Standardized feature value",
    )

    ax.set_xticks(
        np.arange(len(heatmap_columns))
    )
    ax.set_xticklabels(
        heatmap_columns,
        rotation=75,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(plot_df.index))
    )
    ax.set_yticklabels(
        plot_df.index.astype(str)
    )

    ax.set_xlabel("Spatial feature")
    ax.set_ylabel("Sample ID")
    ax.set_title(
        "MIBI sample-level spatial feature heatmap"
    )

    fig.tight_layout()
    fig.savefig(
        HEATMAP_PATH,
        dpi=300,
    )
    plt.close(fig)

    print("Saved heatmap:", HEATMAP_PATH)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input table: {INPUT_PATH}\n"
            "Run:\n"
            "python -m src.data.make_mibi_splits"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "SampleID",
        "Group",
        "immuneGroup",
        "tumorYN",
        "centroid_x",
        "centroid_y",
        "patient_class",
        "split",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Input table is missing columns: {missing_columns}"
        )

    print("Loaded:", INPUT_PATH)
    print("Shape:", df.shape)
    print("Unique SampleID:", df["SampleID"].nunique())
    print("Approximate field-of-view area:", FOV_AREA_MM2, "mm²")

    feature_df = build_feature_table(df)
    feature_df = merge_distance_features(feature_df)

    run_qc(
        input_df=df,
        feature_df=feature_df,
    )

    feature_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    feature_df.to_csv(
        REPORT_TABLE_PATH,
        index=False,
    )

    print("\nSaved feature tables:")
    print(OUTPUT_PATH)
    print(REPORT_TABLE_PATH)

    plot_feature_heatmap(feature_df)

    print("\nSample-level spatial feature build complete.")


if __name__ == "__main__":
    main()