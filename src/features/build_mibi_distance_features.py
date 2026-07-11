"""Build sample-level nearest-neighbor distance features for MIBI-TNBC."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.spatial_distances import (
    MIBI_PIXEL_SIZE_UM,
    get_cells,
    nearest_neighbor_distance,
    summarize_distances,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

INPUT_PATH = PROCESSED_DIR / "mibi_cells_with_splits.csv"

FEATURE_OUTPUT_PATH = (
    PROCESSED_DIR / "mibi_distance_features.csv"
)

REPORT_TABLE_PATH = (
    TABLES_DIR / "mibi_distance_features_summary.csv"
)

CELL_DISTANCE_OUTPUT_PATH = (
    PROCESSED_DIR / "mibi_cell_to_tumor_distances.csv"
)


# ---------------------------------------------------------------------
# Population definitions
# ---------------------------------------------------------------------

POPULATIONS = {
    "cd8_t": "cd8_to_tumor",
    "macrophage": "macrophage_to_tumor",
    "b_cell": "b_cell_to_tumor",
}

DISTANCE_STATS = [
    "mean",
    "median",
    "p10",
    "p25",
    "p75",
    "p90",
]


def get_single_sample_value(
    sample_df: pd.DataFrame,
    column: str,
) -> object:
    """
    Return the one unique non-missing value for a sample.

    patient_class and split should each be constant within one SampleID.
    """
    if column not in sample_df.columns:
        return np.nan

    values = sample_df[column].dropna().unique()

    if len(values) == 0:
        return np.nan

    if len(values) > 1:
        raise ValueError(
            f"Column {column!r} has multiple values inside "
            f"SampleID {sample_df['SampleID'].iloc[0]}: "
            f"{values.tolist()}"
        )

    return values[0]


def prepare_population(
    sample_df: pd.DataFrame,
    cell_type: str,
) -> pd.DataFrame:
    """
    Select one population and keep cells with valid coordinates.
    """
    population_df = get_cells(
        sample_df,
        cell_type=cell_type,
    )

    population_df = population_df.dropna(
        subset=["centroid_x", "centroid_y"]
    ).copy()

    return population_df


def build_distance_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:

    1. One summary row per SampleID.
    2. One detailed row per immune-cell distance.
    """
    feature_rows: list[dict[str, object]] = []
    cell_distance_rows: list[dict[str, object]] = []

    grouped = df.groupby("SampleID", sort=True)

    for sample_id, sample_df in grouped:
        print(f"Processing SampleID {sample_id}")

        tumor_cells = prepare_population(
            sample_df,
            cell_type="tumor",
        )

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
            "n_total_cells": int(len(sample_df)),
            "n_tumor_cells": int(len(tumor_cells)),
        }

        for cell_type, feature_prefix in POPULATIONS.items():
            source_cells = prepare_population(
                sample_df,
                cell_type=cell_type,
            )

            distances_px = nearest_neighbor_distance(
                source_cells=source_cells,
                target_cells=tumor_cells,
            )

            summary_px = summarize_distances(distances_px)

            feature_row[
                f"{feature_prefix}_n_source_cells"
            ] = int(len(source_cells))

            for statistic in DISTANCE_STATS:
                pixel_value = summary_px[statistic]

                feature_row[
                    f"{feature_prefix}_{statistic}_px"
                ] = pixel_value

                if np.isfinite(pixel_value):
                    micrometer_value = (
                        pixel_value * MIBI_PIXEL_SIZE_UM
                    )
                else:
                    micrometer_value = np.nan

                feature_row[
                    f"{feature_prefix}_{statistic}_um"
                ] = micrometer_value

            # Detailed cell-level table for plotting and QC.
            if len(distances_px) != len(source_cells):
                raise ValueError(
                    f"Distance count mismatch for SampleID {sample_id}, "
                    f"population {cell_type}."
                )

            for cell_label, distance_px in zip(
                source_cells["cellLabelInImage"],
                distances_px,
            ):
                cell_distance_rows.append(
                    {
                        "SampleID": int(sample_id),
                        "patient_class": get_single_sample_value(
                            sample_df,
                            "patient_class",
                        ),
                        "split": get_single_sample_value(
                            sample_df,
                            "split",
                        ),
                        "source_population": cell_type,
                        "source_cell_label": int(cell_label),
                        "distance_to_nearest_tumor_px": float(
                            distance_px
                        ),
                        "distance_to_nearest_tumor_um": float(
                            distance_px * MIBI_PIXEL_SIZE_UM
                        ),
                    }
                )

        feature_rows.append(feature_row)

    features_df = pd.DataFrame(feature_rows)
    cell_distances_df = pd.DataFrame(cell_distance_rows)

    return features_df, cell_distances_df


def plot_distance_distribution(
    cell_distances_df: pd.DataFrame,
    population: str,
    output_filename: str,
) -> None:
    """
    Plot one boxplot per sample for a selected immune population.
    """
    population_df = cell_distances_df[
        cell_distances_df["source_population"] == population
    ].copy()

    if population_df.empty:
        print(
            f"No distances available for {population}. "
            "Skipping plot."
        )
        return

    sample_ids = sorted(
        population_df["SampleID"].unique().tolist()
    )

    distance_groups = []
    plotted_sample_ids = []

    for sample_id in sample_ids:
        values = population_df.loc[
            population_df["SampleID"] == sample_id,
            "distance_to_nearest_tumor_um",
        ].dropna()

        if len(values) > 0:
            distance_groups.append(values.to_numpy())
            plotted_sample_ids.append(str(sample_id))

    if not distance_groups:
        print(
            f"No valid distance groups for {population}. "
            "Skipping plot."
        )
        return

    plt.figure(figsize=(16, 6))

    plt.boxplot(
        distance_groups,
        labels=plotted_sample_ids,
        showfliers=False,
    )

    plt.xlabel("Sample ID")
    plt.ylabel("Nearest tumor distance (approximate µm)")
    plt.title(
        f"{population.replace('_', ' ').title()} "
        "distance to nearest tumor cell"
    )
    plt.xticks(rotation=90)
    plt.tight_layout()

    output_path = FIGURES_DIR / output_filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved figure:", output_path)


def run_qc(
    input_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> None:
    """
    Run simple checks on the completed feature table.
    """
    expected_samples = input_df["SampleID"].nunique()

    if len(features_df) != expected_samples:
        raise ValueError(
            "Feature table must contain exactly one row per sample. "
            f"Expected {expected_samples}, found {len(features_df)}."
        )

    if features_df["SampleID"].duplicated().any():
        duplicated = features_df.loc[
            features_df["SampleID"].duplicated(),
            "SampleID",
        ].tolist()

        raise ValueError(
            f"Duplicate SampleID values found: {duplicated}"
        )

    if (features_df["n_tumor_cells"] == 0).any():
        no_tumor_samples = features_df.loc[
            features_df["n_tumor_cells"] == 0,
            "SampleID",
        ].tolist()

        print(
            "Warning: samples without tumor cells:",
            no_tumor_samples,
        )

    print("\nQC summary")
    print("Input rows:", len(input_df))
    print("Unique input samples:", expected_samples)
    print("Feature-table rows:", len(features_df))

    source_count_columns = [
        column
        for column in features_df.columns
        if column.endswith("_n_source_cells")
    ]

    print("\nSamples with zero source cells:")

    for column in source_count_columns:
        zero_count = int((features_df[column] == 0).sum())
        print(f"{column}: {zero_count}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}\n"
            "Run make_mibi_splits.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    print("Loaded:", INPUT_PATH)
    print("Shape:", df.shape)
    print("Unique SampleID:", df["SampleID"].nunique())

    required_columns = [
        "SampleID",
        "cellLabelInImage",
        "centroid_x",
        "centroid_y",
        "tumorYN",
        "immuneGroup",
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
            f"Input table is missing required columns: "
            f"{missing_columns}"
        )

    features_df, cell_distances_df = build_distance_features(df)

    run_qc(
        input_df=df,
        features_df=features_df,
    )

    features_df.to_csv(
        FEATURE_OUTPUT_PATH,
        index=False,
    )

    # The same summary is also saved under reports/tables so it is easy
    # to find as a visible project result.
    features_df.to_csv(
        REPORT_TABLE_PATH,
        index=False,
    )

    cell_distances_df.to_csv(
        CELL_DISTANCE_OUTPUT_PATH,
        index=False,
    )

    print("\nSaved tables:")
    print(FEATURE_OUTPUT_PATH)
    print(REPORT_TABLE_PATH)
    print(CELL_DISTANCE_OUTPUT_PATH)

    plot_distance_distribution(
        cell_distances_df,
        population="cd8_t",
        output_filename=(
            "cd8_to_tumor_distance_by_sample.png"
        ),
    )

    plot_distance_distribution(
        cell_distances_df,
        population="macrophage",
        output_filename=(
            "macrophage_to_tumor_distance_by_sample.png"
        ),
    )

    plot_distance_distribution(
        cell_distances_df,
        population="b_cell",
        output_filename=(
            "b_cell_to_tumor_distance_by_sample.png"
        ),
    )

    print("\nDistance-feature build complete.")


if __name__ == "__main__":
    main()