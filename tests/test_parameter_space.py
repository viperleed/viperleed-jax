from pytest_cases import parametrize_with_cases

from tests.structures import CaseStatesAfterInit, ParameterSpaceSize, Tag
from viperleed_jax.parameter_space import ParameterSpace


def _compare_parameter_space_size(parameter_space, layer, expected_size):
    size_at_layer = parameter_space.__free_params_up_to_layer(layer)
    size_at_layer = ParameterSpaceSize(*size_at_layer)
    assert size_at_layer == expected_size


@parametrize_with_cases('test_case', cases=CaseStatesAfterInit)
def test_build_parameter_space(test_case):
    parameter_space, *_ = test_case
    assert isinstance(parameter_space, ParameterSpace)


@parametrize_with_cases(
    'test_case',
    cases=CaseStatesAfterInit,
    has_tag=Tag.PARAMETER_SPACE_SIZE_TOTAL,
)
def test_total_parameter_size(test_case):
    parameter_space, _, info = test_case
    assert isinstance(parameter_space, ParameterSpace)


@parametrize_with_cases(
    'test_case',
    cases=CaseStatesAfterInit,
    has_tag=Tag.PARAMETER_SPACE_SIZE_SYMMETRY,
)
def test_symmetrized_parameter_size(test_case):
    parameter_space, _, info = test_case
    assert isinstance(parameter_space, ParameterSpace)
