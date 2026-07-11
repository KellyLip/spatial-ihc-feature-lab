"""Nearest-neighbor distance features for MIBI-TNBC cell data."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# Approximate conversion reported for the original MIBI-TNBC images:
# 100 pixels ≈ 39 micrometers.
MIBI_PIXEL_SIZE_UM = 0.39


# Dataset-specific population definitions.
CELL_TYPE_FILTERS = {
    "tumor": ("tumorYN", 1),
    "cd8_t": ("immuneGroup", 3),
    "b_cell": ("immuneGroup", 6),
    "macrophage": ("immuneGroup", 8),
}


def get_cells(
    df: pd.DataFrame,
    cell_type: Optional[str] = None,
    marker_positive: Optional[tuple[str, float]] = None,
) -> pd.DataFrame:
    """
    Select a cell population from the MIBI cell table.

    Parameters
    ----------
    df:
        Cell-level MIBI DataFrame.

    cell_type:
        One of:
        - "tumor"
        - "cd8_t"
        - "b_cell"
        - "macrophage"

    marker_positive:
        Optional tuple containing:

            (marker_column, positivity_threshold)

        Example:

            marker_positive=("PD1", 0.5)

        Marker positivity is supported for later work, but the initial
        features use the existing cell labels rather than arbitrary marker
        thresholds.

    Returns
    -------
    pd.DataFrame
        Filtered copy of the input DataFrame.
    """
    selected = df.copy()

    if cell_type is not None:
        normalized_type = cell_type.strip().lower()

        if normalized_type not in CELL_TYPE_FILTERS:
            valid_types = sorted(CELL_TYPE_FILTERS)
            raise ValueError(
                f"Unknown cell type: {cell_type}. "
                f"Expected one of: {valid_types}"
            )

        column, expected_value = CELL_TYPE_FILTERS[normalized_type]

        if column not in selected.columns:
            raise KeyError(
                f"Column required for {normalized_type} is missing: {column}"
            )

        selected = selected[selected[column] == expected_value]

    if marker_positive is not None:
        marker_column, threshold = marker_positive

        if marker_column not in selected.columns:
            raise KeyError(f"Marker column is missing: {marker_column}")

        selected = selected[selected[marker_column] >= threshold]

    return selected.copy()


def _extract_coordinates(
    df: pd.DataFrame,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> np.ndarray:
    """
    Extract finite x/y coordinates from a cell DataFrame.
    """
    missing_columns = [
        column
        for column in (x_col, y_col)
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing coordinate columns: {missing_columns}")

    coordinates = (
        df[[x_col, y_col]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    finite_rows = np.isfinite(coordinates).all(axis=1)

    return coordinates[finite_rows]


def nearest_neighbor_distance(
    source_cells: pd.DataFrame,
    target_cells: pd.DataFrame,
    *,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    pixel_size_um: Optional[float] = None,
    exclude_self: bool = False,
) -> np.ndarray:
    """
    Calculate each source cell's distance to its nearest target cell.

    This calculation is directional:

        CD8 -> tumor

    is not necessarily the same feature as:

        tumor -> CD8

    Parameters
    ----------
    source_cells:
        Population for which one distance is calculated per cell.

    target_cells:
        Population searched for the nearest neighbor.

    pixel_size_um:
        When supplied, convert distances from pixels to micrometers.

    exclude_self:
        Use True when the source and target represent the same population.
        The nearest cell would otherwise usually be the cell itself.

    Returns
    -------
    np.ndarray
        One nearest-neighbor distance per valid source cell.

        An empty array is returned when either population is empty.
    """
    source_coordinates = _extract_coordinates(
        source_cells,
        x_col=x_col,
        y_col=y_col,
    )

    target_coordinates = _extract_coordinates(
        target_cells,
        x_col=x_col,
        y_col=y_col,
    )

    if len(source_coordinates) == 0 or len(target_coordinates) == 0:
        return np.array([], dtype=float)

    tree = cKDTree(target_coordinates)

    if exclude_self:
        if len(target_coordinates) < 2:
            return np.array([], dtype=float)

        distances, _ = tree.query(source_coordinates, k=2)
        nearest_distances = distances[:, 1]
    else:
        nearest_distances, _ = tree.query(source_coordinates, k=1)

    nearest_distances = np.asarray(nearest_distances, dtype=float)

    if pixel_size_um is not None:
        if pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be greater than zero.")

        nearest_distances = nearest_distances * pixel_size_um

    return nearest_distances


def summarize_distances(distances: np.ndarray) -> dict[str, float]:
    """
    Summarize a nearest-neighbor distance distribution.

    Returns count, mean, median, and selected percentiles.
    """
    distances = np.asarray(distances, dtype=float)
    distances = distances[np.isfinite(distances)]

    if len(distances) == 0:
        return {
            "n_source_cells": 0,
            "mean": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
        }

    return {
        "n_source_cells": int(len(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p10": float(np.percentile(distances, 10)),
        "p25": float(np.percentile(distances, 25)),
        "p75": float(np.percentile(distances, 75)),
        "p90": float(np.percentile(distances, 90)),
    }