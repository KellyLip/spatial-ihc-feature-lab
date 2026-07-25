"""Train sample-level MIBI spatial architecture classifiers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CV_METRICS_PATH = (
    METRICS_DIR
    / "spatial_classifier_cv_metrics.csv"
)

TEST_METRICS_PATH = (
    METRICS_DIR
    / "spatial_classifier_test_metrics.csv"
)

CLASSIFICATION_REPORT_PATH = (
    METRICS_DIR
    / "spatial_classifier_classification_report.csv"
)

PREDICTIONS_PATH = (
    TABLES_DIR
    / "spatial_classifier_predictions.csv"
)

CONFUSION_MATRIX_PATH = (
    FIGURES_DIR
    / "spatial_classifier_confusion_matrix.png"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ID_COL = "SampleID"
TARGET_COL = "patient_class"
SPLIT_COL = "split"

RANDOM_STATE = 42

# The development data contains five cold samples.
# Five folds is therefore the largest sensible stratified split.
N_SPLITS = 5
N_REPEATS = 3

PATIENT_CLASS_LABELS = {
    0: "mixed",
    1: "compartmentalized",
    2: "cold",
}


# ---------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------

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

# These indicators preserve the biological meaning of missing distances.
#
# For example:
# b_cell_absent = 1 means the sample contains no B cells, so a
# B-cell-to-tumor distance cannot be calculated.
ABSENCE_FEATURES = [
    "cd8_absent",
    "macrophage_absent",
    "b_cell_absent",
]


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def validate_input_table(df: pd.DataFrame) -> None:
    """Check that the sample-level modeling table is valid."""

    required_columns = [
        ID_COL,
        TARGET_COL,
        SPLIT_COL,
        *RAW_COUNT_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Input table is missing required columns: "
            f"{missing_columns}"
        )

    if df[ID_COL].isna().any():
        raise ValueError("SampleID contains missing values.")

    if df[ID_COL].duplicated().any():
        duplicated_samples = df.loc[
            df[ID_COL].duplicated(keep=False),
            ID_COL,
        ].tolist()

        raise ValueError(
            "The modeling table must contain exactly one row "
            "per sample. "
            f"Duplicated SampleIDs: {duplicated_samples}"
        )

    if df[TARGET_COL].isna().any():
        raise ValueError(
            "patient_class contains missing values."
        )

    unexpected_targets = sorted(
        set(df[TARGET_COL].unique())
        - set(PATIENT_CLASS_LABELS)
    )

    if unexpected_targets:
        raise ValueError(
            "Unexpected patient_class values: "
            f"{unexpected_targets}"
        )

    expected_splits = {"train", "val", "test"}
    observed_splits = set(
        df[SPLIT_COL].dropna().unique()
    )

    if observed_splits != expected_splits:
        raise ValueError(
            f"Expected splits {expected_splits}, "
            f"but found {observed_splits}."
        )

    if (df["n_total_cells"] <= 0).any():
        raise ValueError(
            "Every sample must contain at least one cell."
        )

    if (
        df["n_tumor_cells"]
        > df["n_total_cells"]
    ).any():
        raise ValueError(
            "A sample contains more tumor cells "
            "than total cells."
        )


# ---------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------

def add_derived_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cell fractions and population-absence indicators.

    Fractions make cell counts more comparable between samples
    containing different total numbers of cells.
    """
    out = df.copy()

    total_cells = out["n_total_cells"].astype(float)

    out["tumor_fraction"] = (
        out["n_tumor_cells"]
        / total_cells
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


def get_feature_sets(
    df: pd.DataFrame,
) -> dict[str, list[str]]:
    """Define the feature groups used in the comparison."""

    distance_features = sorted(
        column
        for column in df.columns
        if column.endswith("_um")
        and "_to_tumor_" in column
    )

    count_features = (
        RAW_COUNT_FEATURES
        + DERIVED_COUNT_FEATURES
    )

    feature_sets = {
        "count_only": count_features,

        # Absence indicators are included because an undefined
        # distance caused by an absent population is meaningful.
        "distance_only": (
            distance_features
            + ABSENCE_FEATURES
        ),

        "combined": (
            count_features
            + distance_features
            + ABSENCE_FEATURES
        ),
    }

    for feature_set_name, columns in feature_sets.items():
        missing_columns = [
            column
            for column in columns
            if column not in df.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Feature set {feature_set_name!r} "
                f"is missing columns: {missing_columns}"
            )

        if len(columns) != len(set(columns)):
            raise ValueError(
                f"Feature set {feature_set_name!r} "
                "contains duplicate columns."
            )

    return feature_sets


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def get_sklearn_major_minor() -> tuple[int, int]:
    """Return the installed scikit-learn major/minor version."""

    parts = SKLEARN_VERSION.split(".")

    return int(parts[0]), int(parts[1])


def make_elastic_net_classifier() -> LogisticRegression:
    """
    Create an elastic-net multinomial classifier.

    The constructor changed in recent scikit-learn versions,
    so this small compatibility check prevents deprecation warnings.
    """
    common_arguments = {
        "solver": "saga",
        "l1_ratio": 0.5,
        "C": 1.0,
        "max_iter": 10_000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }

    if get_sklearn_major_minor() >= (1, 8):
        return LogisticRegression(
            **common_arguments,
        )

    return LogisticRegression(
        penalty="elasticnet",
        **common_arguments,
    )


def make_hist_gradient_boosting_classifier(
) -> HistGradientBoostingClassifier:
    """
    Create a constrained gradient-boosting model.

    Shallow trees and regularization reduce overfitting risk
    on the small sample-level dataset.
    """
    arguments = {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 5,
        "min_samples_leaf": 3,
        "l2_regularization": 1.0,
        "random_state": RANDOM_STATE,
    }

    # class_weight was added to this classifier in newer
    # scikit-learn versions.
    if get_sklearn_major_minor() >= (1, 2):
        arguments["class_weight"] = "balanced"

    return HistGradientBoostingClassifier(
        **arguments,
    )


def get_models() -> dict[str, Pipeline]:
    """Create preprocessing and modeling pipelines."""

    return {
        "dummy_most_frequent": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "classifier",
                    DummyClassifier(
                        strategy="most_frequent",
                    ),
                ),
            ]
        ),

        "elastic_net_logistic_regression": Pipeline(
            steps=[
                # Imputation is fitted inside each CV training fold.
                # This prevents validation information from leaking
                # into preprocessing.
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    make_elastic_net_classifier(),
                ),
            ]
        ),

        "random_forest": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=4,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "hist_gradient_boosting": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "classifier",
                    make_hist_gradient_boosting_classifier(),
                ),
            ]
        ),
    }


# ---------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------

def evaluate_with_cross_validation(
    development_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    models: dict[str, Pipeline],
) -> pd.DataFrame:
    """Compare fixed model and feature-set configurations."""

    y_development = (
        development_df[TARGET_COL]
        .astype(int)
    )

    minimum_class_count = int(
        y_development.value_counts().min()
    )

    if minimum_class_count < N_SPLITS:
        raise ValueError(
            "Repeated stratified CV needs at least "
            f"{N_SPLITS} development samples in every class. "
            f"The smallest class has {minimum_class_count}."
        )

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }

    # The dummy classifier ignores biological features,
    # so it only needs to be evaluated once.
    evaluation_jobs = [
        (
            "not_applicable",
            "dummy_most_frequent",
            ["n_total_cells"],
        )
    ]

    model_names = [
        "elastic_net_logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]

    for feature_set_name, columns in feature_sets.items():
        for model_name in model_names:
            evaluation_jobs.append(
                (
                    feature_set_name,
                    model_name,
                    columns,
                )
            )

    rows: list[dict[str, object]] = []

    for (
        feature_set_name,
        model_name,
        columns,
    ) in evaluation_jobs:
        print(
            "Cross-validating: "
            f"{model_name} | {feature_set_name}"
        )

        results = cross_validate(
            estimator=models[model_name],
            X=development_df[columns],
            y=y_development,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )

        row: dict[str, object] = {
            "model": model_name,
            "feature_set": feature_set_name,
            "n_input_features": len(columns),
            "n_development_samples": len(
                development_df
            ),
            "n_cv_splits": N_SPLITS,
            "n_cv_repeats": N_REPEATS,
        }

        for metric_name in scoring:
            values = results[
                f"test_{metric_name}"
            ]

            row[
                f"mean_{metric_name}"
            ] = float(np.mean(values))

            row[
                f"std_{metric_name}"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

        row["mean_fit_time_seconds"] = float(
            np.mean(results["fit_time"])
        )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "mean_macro_f1",
                "mean_balanced_accuracy",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def select_best_configuration(
    cv_results: pd.DataFrame,
) -> pd.Series:
    """
    Select the best non-dummy configuration.

    Primary selection metric:
    mean macro F1

    Tie-breaker:
    mean balanced accuracy
    """
    eligible_results = cv_results[
        cv_results["model"]
        != "dummy_most_frequent"
    ].copy()

    best_row = (
        eligible_results
        .sort_values(
            [
                "mean_macro_f1",
                "mean_balanced_accuracy",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return best_row


# ---------------------------------------------------------------------
# Held-out test evaluation
# ---------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_ids: list[int],
) -> None:
    """Save the held-out test confusion matrix."""

    labels = [
        PATIENT_CLASS_LABELS[class_id]
        for class_id in class_ids
    ]

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)

    ax.set_xticks(
        np.arange(len(labels))
    )
    ax.set_yticks(
        np.arange(len(labels))
    )

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(
        "Spatial architecture classifier"
    )

    threshold = (
        cm.max() / 2
        if cm.size
        else 0
    )

    for row_index in range(cm.shape[0]):
        for column_index in range(cm.shape[1]):
            value = cm[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
            )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
    )

    plt.close(fig)


def evaluate_selected_model(
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    models: dict[str, Pipeline],
    best_row: pd.Series,
) -> tuple[
    Pipeline,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Fit the selected model and evaluate it once on test data."""

    model_name = str(
        best_row["model"]
    )

    feature_set_name = str(
        best_row["feature_set"]
    )

    columns = feature_sets[
        feature_set_name
    ]

    model = models[model_name]

    X_development = development_df[columns]
    y_development = (
        development_df[TARGET_COL]
        .astype(int)
    )

    X_test = test_df[columns]
    y_test = (
        test_df[TARGET_COL]
        .astype(int)
    )

    model.fit(
        X_development,
        y_development,
    )

    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "feature_set": feature_set_name,
        "n_input_features": len(columns),
        "n_development_samples": len(
            development_df
        ),
        "n_test_samples": len(test_df),
        "accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_test,
                y_pred,
            )
        ),
        "macro_f1": f1_score(
            y_test,
            y_pred,
            average="macro",
        ),
        "weighted_f1": f1_score(
            y_test,
            y_pred,
            average="weighted",
        ),
    }

    predictions = test_df[
        [
            ID_COL,
            TARGET_COL,
            SPLIT_COL,
        ]
    ].copy()

    predictions = predictions.rename(
        columns={
            TARGET_COL: "true_class",
        }
    )

    predictions["true_label"] = (
        predictions["true_class"]
        .map(PATIENT_CLASS_LABELS)
    )

    predictions["predicted_class"] = y_pred

    predictions["predicted_label"] = (
        predictions["predicted_class"]
        .map(PATIENT_CLASS_LABELS)
    )

    class_ids = sorted(
        PATIENT_CLASS_LABELS
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=class_ids,
        target_names=[
            PATIENT_CLASS_LABELS[class_id]
            for class_id in class_ids
        ],
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "label",
            }
        )
    )

    report_df.insert(
        0,
        "feature_set",
        feature_set_name,
    )

    report_df.insert(
        0,
        "model",
        model_name,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_ids,
    )

    plot_confusion_matrix(
        cm=cm,
        class_ids=class_ids,
    )

    return (
        model,
        pd.DataFrame([metrics]),
        predictions,
        report_df,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            f"Missing input file: {INPUT_PATH}\n"
            "Run build_mibi_distance_features.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    validate_input_table(df)

    df = add_derived_features(df)

    feature_sets = get_feature_sets(df)
    models = get_models()

    # The test split stays completely untouched during
    # cross-validation and model selection.
    development_df = df[
        df[SPLIT_COL].isin(
            ["train", "val"]
        )
    ].copy()

    test_df = df[
        df[SPLIT_COL] == "test"
    ].copy()

    print(
        "scikit-learn version:",
        SKLEARN_VERSION,
    )

    print(
        "Development samples:",
        len(development_df),
    )

    print(
        "Held-out test samples:",
        len(test_df),
    )

    print("\nDevelopment class counts:")

    class_counts = (
        development_df[TARGET_COL]
        .value_counts()
        .sort_index()
        .rename(
            index=PATIENT_CLASS_LABELS
        )
    )

    print(class_counts)

    print("\nFeature-set sizes:")

    for name, columns in feature_sets.items():
        print(
            f"  {name}: {len(columns)}"
        )

    cv_results = evaluate_with_cross_validation(
        development_df=development_df,
        feature_sets=feature_sets,
        models=models,
    )

    cv_results.to_csv(
        CV_METRICS_PATH,
        index=False,
    )

    print("\nCross-validation ranking:")

    ranking_columns = [
        "model",
        "feature_set",
        "mean_macro_f1",
        "std_macro_f1",
        "mean_balanced_accuracy",
    ]

    print(
        cv_results[ranking_columns]
        .to_string(index=False)
    )

    best_row = select_best_configuration(
        cv_results
    )

    print("\nSelected configuration:")
    print(
        "Model:",
        best_row["model"],
    )
    print(
        "Feature set:",
        best_row["feature_set"],
    )
    print(
        "Mean CV macro F1:",
        round(
            float(
                best_row["mean_macro_f1"]
            ),
            3,
        ),
    )

    (
        _,
        test_metrics,
        predictions,
        classification_report_df,
    ) = evaluate_selected_model(
        development_df=development_df,
        test_df=test_df,
        feature_sets=feature_sets,
        models=models,
        best_row=best_row,
    )

    test_metrics.to_csv(
        TEST_METRICS_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    classification_report_df.to_csv(
        CLASSIFICATION_REPORT_PATH,
        index=False,
    )

    print("\nHeld-out test metrics:")

    print(
        test_metrics.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(CV_METRICS_PATH)
    print(TEST_METRICS_PATH)
    print(CLASSIFICATION_REPORT_PATH)
    print(PREDICTIONS_PATH)
    print(CONFUSION_MATRIX_PATH)

    print(
        "\nSpatial classifier baseline complete."
    )


if __name__ == "__main__":
    main()