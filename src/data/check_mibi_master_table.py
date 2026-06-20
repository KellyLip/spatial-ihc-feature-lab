from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

MASTER_TABLE_PATH = PROCESSED_DIR / "mibi_cellData_with_patient_class_and_centroids.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mibi_constants import GROUP_LABELS, IMMUNE_GROUP_LABELS, PATIENT_CLASS_LABELS


def load_master_table() -> pd.DataFrame:
    if not MASTER_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {MASTER_TABLE_PATH}")

    return pd.read_csv(MASTER_TABLE_PATH)


def save_qc_tables(df: pd.DataFrame) -> None:
    missing_summary = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "n_missing"})
    )

    missing_summary["pct_missing"] = (
        missing_summary["n_missing"] / len(df) * 100
    ).round(3)

    missing_summary = missing_summary.sort_values(
        "n_missing",
        ascending=False
    )

    missing_summary.to_csv(
        TABLES_DIR / "mibi_master_missing_values.csv",
        index=False
    )

    dtype_summary = (
        df.dtypes
        .astype(str)
        .reset_index()
        .rename(columns={"index": "column", 0: "dtype"})
    )

    dtype_summary.to_csv(
        TABLES_DIR / "mibi_master_column_types.csv",
        index=False
    )

    sample_summary = (
        df.groupby("SampleID")
        .agg(
            n_cells=("cellLabelInImage", "count"),
            patient_class=("patient_class", "first"),
            n_tumor_cells=("tumorYN", lambda x: (x == 1).sum()),
            n_non_tumor_cells=("tumorYN", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    sample_summary.to_csv(
        TABLES_DIR / "mibi_sample_summary.csv",
        index=False
    )

    print("Saved QC tables:")
    print(TABLES_DIR / "mibi_master_missing_values.csv")
    print(TABLES_DIR / "mibi_master_column_types.csv")
    print(TABLES_DIR / "mibi_sample_summary.csv")


def plot_cells_per_sample(df: pd.DataFrame) -> None:
    counts = df["SampleID"].value_counts().sort_index()

    plt.figure(figsize=(12, 5))
    counts.plot(kind="bar")
    plt.xlabel("Sample ID")
    plt.ylabel("Number of cells")
    plt.title("Number of cells per sample")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cells_per_sample.png", dpi=300)
    plt.close()


def plot_patient_class_distribution(df: pd.DataFrame) -> None:
    sample_level = df[["SampleID", "patient_class"]].drop_duplicates()

    counts = sample_level["patient_class"].value_counts().sort_index()
    labels = [PATIENT_CLASS_LABELS.get(int(x), str(x)) for x in counts.index]

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.xticks(range(len(labels)), labels, rotation=0)
    plt.xlabel("Patient class")
    plt.ylabel("Number of samples")
    plt.title("Patient class distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "patient_class_distribution.png", dpi=300)
    plt.close()


def plot_group_distribution(df: pd.DataFrame) -> None:
    counts = df["Group"].value_counts().sort_index()
    labels = [GROUP_LABELS.get(int(x), str(x)) for x in counts.index]

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("Group")
    plt.ylabel("Number of cells")
    plt.title("Broad cell-group distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "group_distribution.png", dpi=300)
    plt.close()


def plot_immune_group_distribution(df: pd.DataFrame) -> None:
    immune_df = df[df["immuneGroup"] != 0].copy()

    counts = immune_df["immuneGroup"].value_counts().sort_index()
    labels = [IMMUNE_GROUP_LABELS.get(int(x), str(x)) for x in counts.index]

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("Immune group")
    plt.ylabel("Number of immune cells")
    plt.title("Immune-cell group distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "immune_group_distribution.png", dpi=300)
    plt.close()


def plot_tumor_status_distribution(df: pd.DataFrame) -> None:
    counts = df["tumorYN"].value_counts().sort_index()

    plt.figure(figsize=(5, 4))
    counts.plot(kind="bar")
    plt.xlabel("tumorYN")
    plt.ylabel("Number of cells")
    plt.title("Tumor vs non-tumor cell distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tumor_status_distribution.png", dpi=300)
    plt.close()


def plot_spatial_sanity_check(df: pd.DataFrame, sample_id: int = 1) -> None:
    sample_df = df[df["SampleID"] == sample_id].copy()

    if sample_df.empty:
        print(f"No cells found for SampleID {sample_id}. Skipping spatial plot.")
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(
        sample_df["centroid_x"],
        sample_df["centroid_y"],
        s=1,
        alpha=0.5,
    )
    plt.gca().invert_yaxis()
    plt.xlabel("centroid_x")
    plt.ylabel("centroid_y")
    plt.title(f"Spatial sanity check: SampleID {sample_id}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"sample_{sample_id}_spatial_sanity_check.png", dpi=300)
    plt.close()


def main() -> None:
    df = load_master_table()

    print("Loaded master table:", df.shape)
    print("Unique samples:", df["SampleID"].nunique())
    print("Columns:", df.shape[1])

    print("\nCritical missing values:")
    critical_cols = ["patient_class", "centroid_x", "centroid_y"]
    print(df[critical_cols].isna().sum())

    save_qc_tables(df)

    plot_cells_per_sample(df)
    plot_patient_class_distribution(df)
    plot_group_distribution(df)
    plot_immune_group_distribution(df)
    plot_tumor_status_distribution(df)
    plot_spatial_sanity_check(df, sample_id=1)

    print("\nSaved figures:")
    print(FIGURES_DIR / "cells_per_sample.png")
    print(FIGURES_DIR / "patient_class_distribution.png")
    print(FIGURES_DIR / "group_distribution.png")
    print(FIGURES_DIR / "immune_group_distribution.png")
    print(FIGURES_DIR / "tumor_status_distribution.png")
    print(FIGURES_DIR / "sample_1_spatial_sanity_check.png")

    print("\nDone.")


if __name__ == "__main__":
    main()