"""Rank MIBI spatial-classifier features using permutation importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mibi_distance_features.csv"
)

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_TABLE_PATH = (
    TABLES_DIR
    / "spatial_classifier_feature_ranking.csv"
)

OUTPUT_FIGURE_PATH = (
    FIGURES_DIR
    / "spatial_classifier_feature_importance.png"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET_COL = "patient_class"
SPLIT_COL = "split"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3
N_PERMUTATION_REPEATS = 20


RAW_COUNT_FEATURES = [
    "n_total_cells",
    "n_tumor_cells",
    "cd8_to_tumor_n_source_cells",
    "macrophage_to_tumor_n_source_cells",
    "b_cell_to_tumor_n_source_cells",
]

DERIVED_COUNT_FEATURES = [
    "tumor_fraction",
    "cd8_fraction",
    "macrophage_fraction",
    "b_cell_fraction",
]

ABSENCE_FEATURES = [
    "cd8_absent",
    "macrophage_absent",
    "b_cell_absent",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cell fractions and population-absence indicators."""

    out = df.copy()

    total_cells = out["n_total_cells"].astype(float)

    out["tumor_fraction"] = (
        out["n_tumor_cells"] / total_cells
    )

    out["cd8_fraction"] = (
        out["cd8_to_tumor_n_source_cells"]
        / total_cells
    )

    out["macrophage_fraction"] = (
        out["macrophage_to_tumor_n_source_cells"]
        / total_cells
    )

    out["b_cell_fraction"] = (
        out["b_cell_to_tumor_n_source_cells"]
        / total_cells
    )

    out["cd8_absent"] = (
        out["cd8_to_tumor_n_source_cells"]
        .eq(0)
        .astype(int)
    )

    out["macrophage_absent"] = (
        out["macrophage_to_tumor_n_source_cells"]
        .eq(0)
        .astype(int)
    )

    out["b_cell_absent"] = (
        out["b_cell_to_tumor_n_source_cells"]
        .eq(0)
        .astype(int)
    )

    return out


def get_combined_features(df: pd.DataFrame) -> list[str]:
    """Return the feature set selected during model comparison."""

    distance_features = sorted(
        column
        for column in df.columns
        if column.endswith("_um")
        and "_to_tumor_" in column
    )

    features = (
        RAW_COUNT_FEATURES
        + DERIVED_COUNT_FEATURES
        + distance_features
        + ABSENCE_FEATURES
    )

    if len(features) != 30:
        raise ValueError(
            "Expected 30 combined input features, "
            f"but found {len(features)}."
        )

    return features


def get_feature_family(feature_name: str) -> str:
    """Assign each feature to a broad biological feature family."""

    if feature_name.endswith("_absent"):
        return "population_absence"

    if feature_name.endswith("_fraction"):
        return "cell_fraction"

    if feature_name in RAW_COUNT_FEATURES:
        return "cell_count"

    if "_to_tumor_" in feature_name:
        return "distance"

    return "other"


def make_model() -> Pipeline:
    """Create the selected preprocessing and model pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=100,
                    max_leaf_nodes=5,
                    min_samples_leaf=3,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def calculate_cross_validated_importance(
    development_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Calculate permutation importance separately within each validation fold.

    The model is refitted for every fold. Importance is therefore measured
    on samples that were not used to fit that particular model.
    """

    X = development_df[feature_columns]
    y = development_df[TARGET_COL].astype(int)

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    importance_rows: list[dict[str, object]] = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        print(
            f"Calculating permutation importance for fold "
            f"{fold_number}/{N_SPLITS * N_REPEATS}"
        )

        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]

        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]

        model = make_model()

        model.fit(
            X_train,
            y_train,
        )

        result = permutation_importance(
            estimator=model,
            X=X_validation,
            y=y_validation,
            scoring="f1_macro",
            n_repeats=N_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE + fold_number,
            n_jobs=1,
        )

        for feature_index, feature_name in enumerate(feature_columns):
            for repeat_index, importance_value in enumerate(
                result.importances[feature_index]
            ):
                importance_rows.append(
                    {
                        "fold": fold_number,
                        "permutation_repeat": repeat_index + 1,
                        "feature": feature_name,
                        "feature_family": get_feature_family(
                            feature_name
                        ),
                        "importance": float(importance_value),
                    }
                )

    raw_importance_df = pd.DataFrame(importance_rows)

    ranking_df = (
        raw_importance_df
        .groupby(
            ["feature", "feature_family"],
            as_index=False,
        )
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            median_importance=("importance", "median"),
            positive_fraction=(
                "importance",
                lambda values: float((values > 0).mean()),
            ),
            n_measurements=("importance", "size"),
        )
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ranking_df.insert(
        0,
        "rank",
        np.arange(1, len(ranking_df) + 1),
    )

    return ranking_df


def plot_feature_importance(
    ranking_df: pd.DataFrame,
    top_n: int = 15,
) -> None:
    """Plot the highest-ranked permutation importance values."""

    plot_df = (
        ranking_df
        .head(top_n)
        .sort_values(
            "mean_importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        plot_df["feature"],
        plot_df["mean_importance"],
        xerr=plot_df["std_importance"],
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Decrease in validation macro F1 after permutation"
    )

    ax.set_ylabel("Feature")

    ax.set_title(
        "Cross-validated permutation importance"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURE_PATH,
        dpi=300,
    )

    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    df = add_derived_features(df)

    feature_columns = get_combined_features(df)

    development_df = df[
        df[SPLIT_COL].isin(["train", "val"])
    ].copy()

    if len(development_df) != 32:
        raise ValueError(
            "Expected 32 development samples, "
            f"but found {len(development_df)}."
        )

    print("Development samples:", len(development_df))
    print("Input features:", len(feature_columns))
    print(
        "Validation folds:",
        N_SPLITS * N_REPEATS,
    )
    print(
        "Permutations per feature per fold:",
        N_PERMUTATION_REPEATS,
    )

    ranking_df = calculate_cross_validated_importance(
        development_df=development_df,
        feature_columns=feature_columns,
    )

    ranking_df.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    plot_feature_importance(
        ranking_df=ranking_df,
        top_n=15,
    )

    print("\nTop 15 features:")
    print(
        ranking_df[
            [
                "rank",
                "feature",
                "feature_family",
                "mean_importance",
                "std_importance",
                "positive_fraction",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_TABLE_PATH)
    print(OUTPUT_FIGURE_PATH)

    print("\nFeature ranking complete.")


if __name__ == "__main__":
    main()