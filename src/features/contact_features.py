"""Radius-based spatial-neighbor features for MIBI-TNBC cell data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.features.spatial_distances import MIBI_PIXEL_SIZE_UM


def _extract_coordinates(
    df: pd.DataFrame,
    *,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> np.ndarray:
    """
    Extract valid numeric x/y coordinates from a cell table.

    Rows with missing or non-finite coordinates are excluded.
    """
    missing_columns = [
        column
        for column in (x_col, y_col)
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing coordinate columns: {missing_columns}"
        )

    coordinates = (
        df[[x_col, y_col]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    finite_rows = np.isfinite(coordinates).all(axis=1)

    return coordinates[finite_rows]


def micrometers_to_pixels(
    distance_um: float,
    *,
    pixel_size_um: float = MIBI_PIXEL_SIZE_UM,
) -> float:
    """
    Convert a distance in micrometers into image pixels.
    """
    if distance_um < 0:
        raise ValueError("distance_um must be zero or greater.")

    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be greater than zero.")

    return distance_um / pixel_size_um


def neighbor_counts_within_radius(
    source_cells: pd.DataFrame,
    target_cells: pd.DataFrame,
    *,
    radius_um: float,
    pixel_size_um: float = MIBI_PIXEL_SIZE_UM,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> np.ndarray:
    """
    Count target cells found within a radius of each source cell.

    This calculation is directional.

    Examples
    --------
    immune -> tumor:
        One count for every immune cell, indicating how many tumor
        cells occur within the specified radius.

    tumor -> immune:
        One count for every tumor cell, indicating how many immune
        cells occur within the specified radius.

    Returns
    -------
    np.ndarray
        One neighbor count per valid source cell.

        If the target population is empty, a zero is returned for
        every source cell.

        If the source population is empty, an empty array is returned.
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

    if len(source_coordinates) == 0:
        return np.array([], dtype=int)

    if len(target_coordinates) == 0:
        return np.zeros(len(source_coordinates), dtype=int)

    radius_px = micrometers_to_pixels(
        radius_um,
        pixel_size_um=pixel_size_um,
    )

    tree = cKDTree(target_coordinates)

    neighbor_indices = tree.query_ball_point(
        source_coordinates,
        r=radius_px,
    )

    return np.asarray(
        [len(indices) for indices in neighbor_indices],
        dtype=int,
    )


def summarize_neighbor_counts(
    neighbor_counts: np.ndarray,
) -> dict[str, float]:
    """
    Summarize radius-neighbor counts for one source population.

    A total pair count counts every source-target pairing. Therefore,
    one source cell with three nearby target cells contributes three
    pairs.
    """
    counts = np.asarray(neighbor_counts, dtype=int)

    if len(counts) == 0:
        return {
            "n_source_cells": 0,
            "total_neighbor_pairs": 0,
            "n_source_with_neighbor": 0,
            "fraction_source_with_neighbor": np.nan,
            "mean_neighbors_per_source": np.nan,
            "median_neighbors_per_source": np.nan,
        }

    has_neighbor = counts > 0

    return {
        "n_source_cells": int(len(counts)),
        "total_neighbor_pairs": int(counts.sum()),
        "n_source_with_neighbor": int(has_neighbor.sum()),
        "fraction_source_with_neighbor": float(has_neighbor.mean()),
        "mean_neighbors_per_source": float(counts.mean()),
        "median_neighbors_per_source": float(np.median(counts)),
    }


def build_radius_neighbor_features(
    source_cells: pd.DataFrame,
    target_cells: pd.DataFrame,
    *,
    radii_um: Iterable[float] = (20.0, 50.0, 100.0),
    pixel_size_um: float = MIBI_PIXEL_SIZE_UM,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> dict[str, float]:
    """
    Build flat radius-neighbor features for several radii.

    The returned dictionary does not include population names.
    The calling script should add a prefix such as
    ``immune_to_tumor`` or ``tumor_to_immune``.
    """
    features: dict[str, float] = {}

    for radius_um in radii_um:
        if radius_um <= 0:
            raise ValueError("All radii must be greater than zero.")

        counts = neighbor_counts_within_radius(
            source_cells,
            target_cells,
            radius_um=radius_um,
            pixel_size_um=pixel_size_um,
            x_col=x_col,
            y_col=y_col,
        )

        summary = summarize_neighbor_counts(counts)

        if float(radius_um).is_integer():
            radius_label = str(int(radius_um))
        else:
            radius_label = str(radius_um).replace(".", "_")

        for statistic, value in summary.items():
            features[
                f"radius_{radius_label}um_{statistic}"
            ] = value

    return features