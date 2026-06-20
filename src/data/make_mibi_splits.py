from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

MASTER_TABLE_PATH = PROCESSED_DIR / "mibi_cellData_with_patient_class_and_centroids.csv"
OUTPUT_PATH = PROCESSED_DIR / "mibi_cells_with_splits.csv"
SUMMARY_PATH = TABLES_DIR / "split_summary.csv"

GROUP_COL = "SampleID"
TARGET_COL = "patient_class"
TEST_SIZE = 0.20
VAL_SIZE = 0.20
RANDOM_STATE = 42

sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.splits import (  # noqa: E402
    make_stratified_group_split_for_group_target,
    split_summary,
)


def main() -> None:
    print("Project root:", PROJECT_ROOT)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if not MASTER_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {MASTER_TABLE_PATH}\n"
            "Run make_mibi_spatial_table.py first."
        )

    df = pd.read_csv(MASTER_TABLE_PATH)
    print("Loaded master table:", df.shape)

    df_split = make_stratified_group_split_for_group_target(
        df=df,
        group_col=GROUP_COL,
        target_col=TARGET_COL,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    summary = split_summary(
        df_split,
        split_col="split",
        group_col=GROUP_COL,
        target_col=TARGET_COL,
    )

    df_split.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nSplit summary:")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
