"""
Utilities for leakage-safe train/validation/test splits.

Main rule:
Do not randomly split cells when many cells come from the same patient/sample.
Split by patient/sample first, then assign all cells from that patient/sample
to the same split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitConfig:
    group_col: str
    target_col: Optional[str] = None
    test_size: float = 0.20
    val_size: float = 0.20
    random_state: int = 42


def assert_no_group_leakage(
    df: pd.DataFrame,
    split_col: str,
    group_col: str,
) -> None:
    """
    Raise an error if the same group appears in more than one split.

    Example:
    If group_col is patient_id, then patient 7 must not appear in both train and test.
    """
    group_split_counts = df.groupby(group_col)[split_col].nunique()
    leaking_groups = group_split_counts[group_split_counts > 1]

    if not leaking_groups.empty:
        raise ValueError(
            f"Data leakage detected. These groups appear in multiple splits: "
            f"{leaking_groups.index.tolist()[:20]}"
        )


def make_group_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
    split_col: str = "split",
) -> pd.DataFrame:
    """
    Create train/val/test split by group only.

    Use this when the target varies inside the group, for example:
    cell phenotype labels inside the same patient.

    The split is not stratified. Always check class balance afterward.
    """
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    out = df.copy()

    groups = pd.Series(out[group_col].dropna().unique(), name=group_col)

    train_val_groups, test_groups = train_test_split(
        groups,
        test_size=test_size,
        random_state=random_state,
    )

    relative_val_size = val_size / (1.0 - test_size)

    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=relative_val_size,
        random_state=random_state,
    )

    out[split_col] = "unassigned"
    out.loc[out[group_col].isin(train_groups), split_col] = "train"
    out.loc[out[group_col].isin(val_groups), split_col] = "val"
    out.loc[out[group_col].isin(test_groups), split_col] = "test"

    assert_no_group_leakage(out, split_col=split_col, group_col=group_col)

    return out


def make_stratified_group_split_for_group_target(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
    split_col: str = "split",
) -> pd.DataFrame:
    """
    Create train/val/test split for a target that is constant per group.

    Example:
    patient_class is one label per patient/sample.

    This function first creates a one-row-per-group table, splits that table,
    then maps the split back to all rows.
    """
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    out = df.copy()

    group_target = (
        out[[group_col, target_col]]
        .drop_duplicates()
        .sort_values(group_col)
        .reset_index(drop=True)
    )

    target_counts_per_group = group_target.groupby(group_col)[target_col].nunique()
    invalid_groups = target_counts_per_group[target_counts_per_group > 1]

    if not invalid_groups.empty:
        raise ValueError(
            f"Target is not constant within these groups: "
            f"{invalid_groups.index.tolist()[:20]}"
        )

    train_val, test = train_test_split(
        group_target,
        test_size=test_size,
        random_state=random_state,
        stratify=group_target[target_col],
    )

    relative_val_size = val_size / (1.0 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val[target_col],
    )

    split_map = {}
    split_map.update({g: "train" for g in train[group_col]})
    split_map.update({g: "val" for g in val[group_col]})
    split_map.update({g: "test" for g in test[group_col]})

    out[split_col] = out[group_col].map(split_map)

    if out[split_col].isna().any():
        raise ValueError("Some rows were not assigned to a split.")

    assert_no_group_leakage(out, split_col=split_col, group_col=group_col)

    return out


def split_summary(
    df: pd.DataFrame,
    split_col: str,
    group_col: str,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a compact summary table for the split.
    """
    rows = []

    for split_name, split_df in df.groupby(split_col):
        row = {
            "split": split_name,
            "n_rows": len(split_df),
            "n_groups": split_df[group_col].nunique(),
        }

        if target_col is not None:
            counts = split_df[[group_col, target_col]].drop_duplicates()[target_col].value_counts()
            for class_id, count in counts.items():
                row[f"class_{class_id}_groups"] = count

        rows.append(row)

    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)