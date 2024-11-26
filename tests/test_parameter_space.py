from pytest_cases import fixture, parametrize_with_cases

from tests.structures import CaseStatesAfterInit, ParameterSpaceSize
from viperleed_jax.base_scatterers import BaseScatterers
from viperleed_jax.parameter_space import ParameterSpace
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)


def _get_space(state):
    base_scatterers = BaseScatterers(state.slab)
    return ParameterSpace(base_scatterers, state.rpars)


def _compare_parameter_space_size(parameter_space, layer, expected_size):
    size_at_layer = parameter_space._free_params_up_to_layer(layer)
    size_at_layer = ParameterSpaceSize(*size_at_layer)
    assert size_at_layer == expected_size


@fixture
@parametrize_with_cases(
    'test_case',
    cases=CaseStatesAfterInit,
)
def space_with_info(test_case):
    state, info = test_case
    parameter_space = _get_space(state)
    return parameter_space, state, info


def test_parameter_space_size(space_with_info, subtests):
    parameter_space, _, info = space_with_info
    with subtests.test('Base layer parameter space size'):
        _compare_parameter_space_size(
            parameter_space,
            layer=DisplacementTreeLayers.Base,
            expected_size=info.total_size,
        )
    with subtests.test('Symmetry layer parameter space size'):
        _compare_parameter_space_size(
            parameter_space,
            layer=DisplacementTreeLayers.Symmetry,
            expected_size=info.symmetry_size,
        )
