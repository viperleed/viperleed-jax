import pytest
from pytest_cases import parametrize_with_cases

from tests.structures import CaseStatesAfterInit, ParameterSpaceSize, Tag
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)


def _compare_parameter_space_size(parameter_space, layer, expected_size):
    size_at_layer = parameter_space._free_params_up_to_layer(layer)
    size_at_layer = ParameterSpaceSize(*size_at_layer)
    assert size_at_layer == expected_size


@parametrize_with_cases(
    'test_case',
    cases=CaseStatesAfterInit,
    has_tag=Tag.PARAMETER_SPACE_SIZE_TOTAL,
)
def test_total_parameter_size(test_case):
    parameter_space, _, info = test_case
    _compare_parameter_space_size(
        parameter_space,
        layer=DisplacementTreeLayers.Base,
        expected_size=info.total_size,
    )


@pytest.mark.xfail(
    reason='Symmetry constraints for vibs and occs are wrong at the moment'
)
@parametrize_with_cases(
    'test_case',
    cases=CaseStatesAfterInit,
    has_tag=Tag.PARAMETER_SPACE_SIZE_SYMMETRY,
)
def test_symmetrized_parameter_size(test_case):
    parameter_space, _, info = test_case
    _compare_parameter_space_size(
        parameter_space,
        layer=DisplacementTreeLayers.Symmetry,
        expected_size=info.symmetry_size,
    )
