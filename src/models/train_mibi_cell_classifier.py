from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "mibi_cells_with_splits.csv"

METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Config
# -----------------------------

TARGET_COL = "Group"
SPLIT_COL = "split"
GROUP_COL = "SampleID"
RANDOM_STATE = 42

# Biological marker / protein-expression columns only.
# We intentionally exclude IDs, coordinates, patient labels, and target labels.
MARKER_COLS = [
    "dsDNA",
    "Vimentin",
    "SMA",
    "B7H3",
    "FoxP3",
    "Lag3",
    "CD4",
    "CD16",
    "CD56",
    "OX40",
    "PD1",
    "CD31",
    "PD-L1",
    "EGFR",
    "Ki67",
    "CD209",
    "CD11c",
    "CD138",
    "CD163",
    "CD68",
    "CSF-1R",
    "CD8",
    "CD3",
    "IDO",
    "Keratin17",
    "CD63",
    "CD45RO",
    "CD20",
    "p53",
    "Beta catenin",
    "HLA-DR",
    "CD11b",
    "CD45",
    "H3K9ac",
    "Pan-Keratin",
    "H3K27me3",
    "phospho-S6",
    "MPO",
    "Keratin6",
    "HLA_Class_1",
]

GROUP_LABELS = {
    1: "Unidentified",
    2: "Immune",
    3: "Endothelial",
    4: "Mesenchymal-like",
    5: "Tumor",
    6: "Keratin-positive tumor",
}


def check_no_group_leakage(df: pd.DataFrame) -> None:
    """
    Make sure no SampleID appears in more than one split.
    This is critical because cells from the same patient are not independent.
    """
    split_counts = df.groupby(GROUP_COL)[SPLIT_COL].nunique()
    leaking_samples = split_counts[split_counts > 1]

    if not leaking_samples.empty:
        raise ValueError(
            f"Leakage detected. These samples appear in multiple splits: "
            f"{leaking_samples.index.tolist()}"
        )


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], model_name: str) -> None:
    """
    Save a confusion matrix plot.
    Rows = true labels.
    Columns = predicted labels.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix: {model_name}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.tight_layout()

    output_path = FIGURES_DIR / f"cell_classifier_confusion_matrix_{model_name}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved confusion matrix: {output_path}")


def get_top_confusions(cm: np.ndarray, class_ids: list[int], model_name: str) -> pd.DataFrame:
    """
    Return the biggest off-diagonal confusion pairs.
    Off-diagonal means true class != predicted class.
    """
    rows = []

    for i, true_class in enumerate(class_ids):
        for j, pred_class in enumerate(class_ids):
            if i == j:
                continue

            rows.append(
                {
                    "model": model_name,
                    "true_class": true_class,
                    "true_label": GROUP_LABELS.get(true_class, str(true_class)),
                    "predicted_class": pred_class,
                    "predicted_label": GROUP_LABELS.get(pred_class, str(pred_class)),
                    "n_cells": cm[i, j],
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("n_cells", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_model(model, model_name: str, X_test, y_test, class_ids: list[int]) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate one trained model and return:
    1. one metrics row
    2. classification report table
    """
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
    }

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=class_ids,
        target_names=[GROUP_LABELS.get(x, str(x)) for x in class_ids],
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )
    report_df.insert(0, "model", model_name)

    cm = confusion_matrix(y_test, y_pred, labels=class_ids)
    label_names = [GROUP_LABELS.get(x, str(x)) for x in class_ids]

    plot_confusion_matrix(cm, label_names, model_name)

    top_confusions = get_top_confusions(cm, class_ids, model_name)

    return metrics, report_df, top_confusions


def main() -> None:
    print("Project root:", PROJECT_ROOT)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {DATA_PATH}\n"
            "Run: python src/data/make_mibi_splits.py"
        )

    df = pd.read_csv(DATA_PATH)
    print("Loaded:", DATA_PATH)
    print("Shape:", df.shape)

    required_cols = MARKER_COLS + [TARGET_COL, SPLIT_COL, GROUP_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    check_no_group_leakage(df)

    # Keep only rows with valid target and split.
    df = df.dropna(subset=[TARGET_COL, SPLIT_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Remove any rows with missing marker values.
    # In this dataset this should usually be zero, but the check is explicit.
    before = len(df)
    df = df.dropna(subset=MARKER_COLS).copy()
    after = len(df)

    print(f"Rows removed because of missing marker values: {before - after}")

    print("\nSplit counts:")
    print(df[SPLIT_COL].value_counts())

    print("\nTarget counts by split:")
    print(pd.crosstab(df[SPLIT_COL], df[TARGET_COL]))

    # Train on train split.
    # Evaluate final reported metrics on test split.
    train_df = df[df[SPLIT_COL] == "train"].copy()
    test_df = df[df[SPLIT_COL] == "test"].copy()

    X_train = train_df[MARKER_COLS]
    y_train = train_df[TARGET_COL]

    X_test = test_df[MARKER_COLS]
    y_test = test_df[TARGET_COL]

    class_ids = sorted(df[TARGET_COL].unique().tolist())

    print("\nClasses:")
    for class_id in class_ids:
        print(class_id, GROUP_LABELS.get(class_id, str(class_id)))

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),

        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            random_state=RANDOM_STATE,
        ),
    }

    metrics_rows = []
    report_tables = []
    confusion_tables = []

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")

        model.fit(X_train, y_train)

        metrics, report_df, top_confusions = evaluate_model(
            model=model,
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
            class_ids=class_ids,
        )

        metrics_rows.append(metrics)
        report_tables.append(report_df)
        confusion_tables.append(top_confusions)

        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    metrics_df = pd.DataFrame(metrics_rows)
    report_df = pd.concat(report_tables, ignore_index=True)
    confusions_df = pd.concat(confusion_tables, ignore_index=True)

    metrics_path = METRICS_DIR / "cell_classifier_baseline.csv"
    report_path = METRICS_DIR / "cell_classifier_classification_report.csv"
    confusions_path = TABLES_DIR / "cell_classifier_top_confusions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    report_df.to_csv(report_path, index=False)
    confusions_df.to_csv(confusions_path, index=False)

    print("\nSaved:")
    print(metrics_path)
    print(report_path)
    print(confusions_path)

    print("\nDone.")


if __name__ == "__main__":
    main()