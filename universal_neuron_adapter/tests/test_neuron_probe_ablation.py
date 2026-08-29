import numpy as np

from universal_neuron_adapter.neuron_probe_ablation import (
    coordinate_feature_indices,
    same_layer_random,
    top_coordinates,
)


def test_coordinate_feature_indices_keep_both_statistics() -> None:
    coordinates = np.asarray([[0, 0], [1, 2]], dtype=np.int64)
    np.testing.assert_array_equal(
        coordinate_feature_indices(coordinates),
        np.asarray([0, 1, 1540, 1541]),
    )


def test_same_layer_random_preserves_budget_and_excludes_selected() -> None:
    ranking = np.tile(np.arange(768), (12, 1))
    selected = top_coordinates(ranking, 4)
    control = same_layer_random(selected, 4, seed=234)
    assert len(control) == len(selected) == 48
    for layer in range(12):
        selected_dimensions = set(selected[selected[:, 0] == layer, 1])
        control_dimensions = set(control[control[:, 0] == layer, 1])
        assert len(control_dimensions) == 4
        assert selected_dimensions.isdisjoint(control_dimensions)
