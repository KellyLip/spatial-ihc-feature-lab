import numpy as np
import pandas as pd

from src.features.spatial_distances import (
    get_cells,
    nearest_neighbor_distance,
    summarize_distances,
)


def test_get_cells_selects_expected_populations() -> None:
    df = pd.DataFrame(
        {
            "tumorYN": [1, 0, 0, 0],
            "immuneGroup": [0, 3, 6, 8],
            "centroid_x": [0, 1, 2, 3],
            "centroid_y": [0, 1, 2, 3],
        }
    )

    assert len(get_cells(df, cell_type="tumor")) == 1
    assert len(get_cells(df, cell_type="cd8_t")) == 1
    assert len(get_cells(df, cell_type="b_cell")) == 1
    assert len(get_cells(df, cell_type="macrophage")) == 1


def test_nearest_neighbor_distance_has_known_answer() -> None:
    source = pd.DataFrame(
        {
            "centroid_x": [0.0, 3.0],
            "centroid_y": [0.0, 4.0],
        }
    )

    target = pd.DataFrame(
        {
            "centroid_x": [0.0],
            "centroid_y": [4.0],
        }
    )

    distances = nearest_neighbor_distance(source, target)

    # Distance from (0, 0) to (0, 4) is 4.
    # Distance from (3, 4) to (0, 4) is 3.
    np.testing.assert_allclose(distances, [4.0, 3.0])


def test_distance_conversion_to_micrometers() -> None:
    source = pd.DataFrame(
        {
            "centroid_x": [0.0],
            "centroid_y": [0.0],
        }
    )

    target = pd.DataFrame(
        {
            "centroid_x": [3.0],
            "centroid_y": [4.0],
        }
    )

    distances = nearest_neighbor_distance(
        source,
        target,
        pixel_size_um=0.5,
    )

    # Pixel distance is 5. At 0.5 micrometers per pixel, result is 2.5.
    np.testing.assert_allclose(distances, [2.5])


def test_empty_target_population_returns_empty_array() -> None:
    source = pd.DataFrame(
        {
            "centroid_x": [0.0],
            "centroid_y": [0.0],
        }
    )

    target = pd.DataFrame(
        {
            "centroid_x": [],
            "centroid_y": [],
        }
    )

    distances = nearest_neighbor_distance(source, target)

    assert len(distances) == 0


def test_summarize_distances() -> None:
    distances = np.array([1.0, 2.0, 3.0, 4.0])

    summary = summarize_distances(distances)

    assert summary["n_source_cells"] == 4
    assert summary["mean"] == 2.5
    assert summary["median"] == 2.5