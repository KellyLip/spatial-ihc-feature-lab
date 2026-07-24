import numpy as np
import pandas as pd

from src.features.contact_features import (
    build_radius_neighbor_features,
    micrometers_to_pixels,
    neighbor_counts_within_radius,
    summarize_neighbor_counts,
)


def test_micrometers_to_pixels() -> None:
    pixels = micrometers_to_pixels(
        2.5,
        pixel_size_um=0.5,
    )

    assert pixels == 5.0


def test_neighbor_counts_have_known_answer() -> None:
    source = pd.DataFrame(
        {
            "centroid_x": [0.0, 10.0],
            "centroid_y": [0.0, 0.0],
        }
    )

    target = pd.DataFrame(
        {
            "centroid_x": [3.0, 30.0],
            "centroid_y": [4.0, 0.0],
        }
    )

    counts = neighbor_counts_within_radius(
        source,
        target,
        radius_um=5.0,
        pixel_size_um=1.0,
    )

    # The first source is exactly 5 units from the first target.
    # The second source has no target within 5 units.
    np.testing.assert_array_equal(counts, [1, 0])


def test_empty_target_returns_zero_for_each_source() -> None:
    source = pd.DataFrame(
        {
            "centroid_x": [0.0, 10.0],
            "centroid_y": [0.0, 0.0],
        }
    )

    target = pd.DataFrame(
        {
            "centroid_x": [],
            "centroid_y": [],
        }
    )

    counts = neighbor_counts_within_radius(
        source,
        target,
        radius_um=20.0,
    )

    np.testing.assert_array_equal(counts, [0, 0])


def test_empty_source_returns_empty_array() -> None:
    source = pd.DataFrame(
        {
            "centroid_x": [],
            "centroid_y": [],
        }
    )

    target = pd.DataFrame(
        {
            "centroid_x": [0.0],
            "centroid_y": [0.0],
        }
    )

    counts = neighbor_counts_within_radius(
        source,
        target,
        radius_um=20.0,
    )

    assert len(counts) == 0


def test_summarize_neighbor_counts() -> None:
    summary = summarize_neighbor_counts(
        np.array([2, 0, 1, 0])
    )

    assert summary["n_source_cells"] == 4
    assert summary["total_neighbor_pairs"] == 3
    assert summary["n_source_with_neighbor"] == 2
    assert summary["fraction_source_with_neighbor"] == 0.5
    assert summary["mean_neighbors_per_source"] == 0.75
    assert summary["median_neighbors_per_source"] == 0.5


def test_build_features_for_multiple_radii() -> None:
    source = pd.DataFrame(
        {
            "centroid_x": [0.0, 10.0],
            "centroid_y": [0.0, 0.0],
        }
    )

    target = pd.DataFrame(
        {
            "centroid_x": [3.0],
            "centroid_y": [4.0],
        }
    )

    features = build_radius_neighbor_features(
        source,
        target,
        radii_um=(5.0, 10.0),
        pixel_size_um=1.0,
    )

    assert features[
        "radius_5um_total_neighbor_pairs"
    ] == 1

    assert features[
        "radius_5um_fraction_source_with_neighbor"
    ] == 0.5

    assert features[
        "radius_10um_total_neighbor_pairs"
    ] == 2

    assert features[
        "radius_10um_fraction_source_with_neighbor"
    ] == 1.0