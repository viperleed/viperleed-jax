import pytest
from pytest_cases import fixture, parametrize_with_cases

from pathlib import Path

from tests.structures import CaseStatesAfterInit, ParameterSpaceSize
from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.parameter_space import ParameterSpace
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)

from viperleed.calc.files.new_displacements.file import DisplacementsFile

DISPLACEMENTS_PATH = (
    Path(__file__).parent / 'test_data' / 'displacements'
)



def _get_space(state):
    atom_basis = AtomBasis(state.slab)
    return ParameterSpace(atom_basis, state.rpars)


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


def test_unconstrained_parameter_space_size(space_with_info, subtests):
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


class TestFe2O3:
    """Tests using the Fe2O3(012) structure."""

    @fixture
    @pytest.mark.parametrize(
        'fname_postfix,n_param_split',
        [
            # (name, (v0r, geo, vib, occ))
            ('simple', (1, 1, 0, 0)),
            ('constrain', (1, 1, 1, 0)),
            ('wildcard', (1, 2, 2, 0)),
            ('complex', (1, 30, 1, 0)),
        ],
    )
    def fe2o3_valid_displacements_file(fname_postfix, n_param_split):
        """Fixture for valid displacements file."""
        path =  DISPLACEMENTS_PATH / 'Fe2O3_012' / f'DISPLACEMENTS_{fname_postfix}'
        disp_file = DisplacementsFile()
        disp_file.read(path)
        return disp_file, n_param_split

    @fixture
    @parametrize_with_cases(
        'case', cases=CaseStatesAfterInit.case_fe2o3_012_converged
    )
    def fe2o3_space(case):
        state, _ = case
        return _get_space(state)


    """Test the Fe2O3 structure."""
    def test_apply_displacements(self, fe2o3_space, fe2o3_valid_displacements_file):
        disp_file, expected_param_split = fe2o3_valid_displacements_file
        fe2o3_space.apply_displacements(
            search_block=disp_file.first_block()
        )
        # check that the number of parameters is as expected
        assert fe2o3_space.n_param_split == expected_param_split
