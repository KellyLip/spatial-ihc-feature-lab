from pathlib import Path

import pandas as pd


# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mibi_tnbc"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CELL_DATA_PATH = RAW_DIR / "cellData.csv"
PATIENT_CLASS_PATH = RAW_DIR / "patient_class.csv"
CENTROIDS_PATH = PROCESSED_DIR / "mibi_cell_centroids.csv"

OUTPUT_PATH = PROCESSED_DIR / "mibi_cellData_with_patient_class_and_centroids.csv"


def main() -> None:
    print("Project root:", PROJECT_ROOT)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load original cellData.csv
    # -----------------------------

    if not CELL_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CELL_DATA_PATH}")

    cell_df = pd.read_csv(CELL_DATA_PATH)

    print("Loaded cellData.csv:", cell_df.shape)

    # Make sure merge keys are integers
    cell_df["SampleID"] = cell_df["SampleID"].astype(int)
    cell_df["cellLabelInImage"] = cell_df["cellLabelInImage"].astype(int)

    # -----------------------------
    # Remove samples 42, 43, 44
    # -----------------------------

    excluded_samples = [42, 43, 44]

    before_rows = cell_df.shape[0]

    cell_df = cell_df[~cell_df["SampleID"].isin(excluded_samples)].copy()

    after_rows = cell_df.shape[0]

    print(f"Removed samples {excluded_samples}")
    print("Rows before filtering:", before_rows)
    print("Rows after filtering:", after_rows)
    print("Rows removed:", before_rows - after_rows)

    # -----------------------------
    # Load patient_class.csv
    # -----------------------------
    # patient_class.csv has no header:
    # Column 1 = SampleID
    # Column 2 = patient_class (0=mixed, 1=compartmentalized, 2=cold)

    if not PATIENT_CLASS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PATIENT_CLASS_PATH}")

    patient_class_df = pd.read_csv(
        PATIENT_CLASS_PATH,
        header=None,
        names=["SampleID", "patient_class"]
    )

    patient_class_df["SampleID"] = patient_class_df["SampleID"].astype(int)
    patient_class_df["patient_class"] = patient_class_df["patient_class"].astype(int)

    print("Loaded patient_class.csv:", patient_class_df.shape)

    # -----------------------------
    # Load centroids
    # -----------------------------
    # Created by extract_mibi_centroids.py

    if not CENTROIDS_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {CENTROIDS_PATH}\n"
            "Run extract_mibi_centroids.py first."
        )

    centroids_df = pd.read_csv(CENTROIDS_PATH)

    required_centroid_cols = [
        "sample_id",
        "cell_label",
        "centroid_x",
        "centroid_y",
        "area_pixels",
    ]

    missing_cols = [
        col for col in required_centroid_cols
        if col not in centroids_df.columns
    ]

    if missing_cols:
        raise ValueError(f"Centroids file is missing columns: {missing_cols}")

    centroids_df = centroids_df.rename(columns={
        "sample_id": "SampleID",
        "cell_label": "cellLabelInImage",
        "area_pixels": "mask_area_pixels",
    })

    centroids_df["SampleID"] = centroids_df["SampleID"].astype(int)
    centroids_df["cellLabelInImage"] = centroids_df["cellLabelInImage"].astype(int)

    print("Loaded mibi_cell_centroids.csv:", centroids_df.shape)

    # -----------------------------
    # Merge patient class
    # -----------------------------

    merged_df = cell_df.merge(
        patient_class_df,
        on="SampleID",
        how="left",
        validate="many_to_one",
    )

    # After removing samples 42–44, this should not contain missing values.
    merged_df["patient_class"] = merged_df["patient_class"].astype("Int64")

    # -----------------------------
    # Merge centroids
    # -----------------------------

    merged_df = merged_df.merge(
        centroids_df[
            [
                "SampleID",
                "cellLabelInImage",
                "centroid_x",
                "centroid_y",
                "mask_area_pixels",
            ]
        ],
        on=["SampleID", "cellLabelInImage"],
        how="left",
        validate="one_to_one",
    )

    # -----------------------------
    # QC
    # -----------------------------

    print("\nFinal merged table:", merged_df.shape)
    print("Unique samples in final table:", merged_df["SampleID"].nunique())
    print("Sample IDs:", sorted(merged_df["SampleID"].unique().tolist()))

    print("\nMissing patient_class rows:", merged_df["patient_class"].isna().sum())
    print("Missing centroid_x rows:", merged_df["centroid_x"].isna().sum())
    print("Missing centroid_y rows:", merged_df["centroid_y"].isna().sum())

    if merged_df["patient_class"].isna().sum() == 0:
        merged_df["patient_class"] = merged_df["patient_class"].astype(int)

    # -----------------------------
    # Save one output CSV only
    # -----------------------------

    merged_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved single merged file:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()