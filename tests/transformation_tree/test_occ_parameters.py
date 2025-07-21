import numpy as np
import pytest
from pytest_cases import fixture, parametrize_with_cases
from viperleed.calc.files.new_displacements.lines import (
    ConstraintLine,
    OccDeltaLine,
    OffsetsLine,
)
from viperleed_jax.lib.derived_quantities.normalized_occupations import (
    normalize_occ_vector,
)

from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)
from viperleed_jax.transformation_tree.occ_parameters import OccTree
from viperleed_jax.transformation_tree.reduced_space import (
    apply_affine_to_subspace,
)

from ..structures import CaseStatesAfterInit


def test_tree_creation(atom_basis, fully_constrained_tree_template):
    """Test tree creation."""
    fully_constrained_tree_template(atom_basis, OccTree)


@fixture
@parametrize_with_cases('case', cases=CaseStatesAfterInit.case_pt25rh75_o_3x1)
def ptrh_tree(case):
    state, _ = case
    atom_basis = AtomBasis(state.slab)
    return OccTree(atom_basis)


class TestPt25Rh75_O_3x1:
    """Tests with the Pt25Rh75_O_3x1 structure."""

    @fixture
    @pytest.mark.parametrize('total_occ', [1.0, 0.5, 0.25])
    def total_occ_tree(self, ptrh_tree, total_occ):
        line = ConstraintLine(f'occ Me* L(1-3) = total {total_occ}')
        ptrh_tree.apply_explicit_constraint(line)
        bounds_line = OccDeltaLine('Me* L(1-3) = Rh 0.10 0.90, Pt 0.90 0.10')
        ptrh_tree.apply_bounds(bounds_line)

        ptrh_tree.apply_implicit_constraints()
        ptrh_tree.finalize_tree()
        return ptrh_tree, total_occ

    def test_tree_is_not_centered(self, total_occ_tree):
        """Test that the tree is not centered.

        It should not be centered because the reference occupations are 25/75%
        while the range is symmetric around 50/50%!"""
        tree, _ = total_occ_tree
        assert not tree.is_centered

    @pytest.mark.parametrize('normalization_method', ['projection', 'mirror'])
    def test_tree_total_occ_sum(
        self, total_occ_tree, normalization_method, subtests
    ):
        """Test that the total occupation sum is correct."""
        tree, total_occ = total_occ_tree

        params = np.array([0.5] * tree.root.dof)
        atom_ids = tree.atom_basis.atom_ids
        non_normalized_occupations = tree(params)
        normalized_occupations = normalize_occ_vector(
            non_normalized_occupations, atom_ids, op_type=normalization_method
        )

        # The sum of the normalized occupations should sum up to
        # 8 + 9 * total_occ (8 100% occupied atoms + 9 atoms with total_occ%)
        sum_occ = normalized_occupations.sum()
        with subtests.test('sum of occupations'):
            assert sum_occ == pytest.approx(8 + 9 * total_occ)

    KNOWN_OCC_PARAM_TO_VALUES = {
        tuple([0.0] * 6): np.array(
            [
                1.0,
                1.0,
                0.9,
                0.1,
                0.1,
                0.9,
                0.1,
                0.9,
                0.9,
                0.1,
                0.1,
                0.9,
                0.9,
                0.1,
                0.9,
                0.1,
                0.9,
                0.1,
                0.9,
                0.1,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
            ],
        ),
        tuple([0.5] * 6): np.array(
            [
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
            ]
        ),
        tuple([1.0] * 6): np.array(
            [
                1.0,
                1.0,
                0.1,
                0.9,
                0.9,
                0.1,
                0.9,
                0.1,
                0.1,
                0.9,
                0.9,
                0.1,
                0.1,
                0.9,
                0.1,
                0.9,
                0.1,
                0.9,
                0.1,
                0.9,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
                0.75,
                0.25,
            ]
        ),
    }

    @pytest.mark.parametrize('input_params', KNOWN_OCC_PARAM_TO_VALUES.keys())
    def test_known_occ_params(
        self,
        ptrh_tree,
        input_params,
    ):
        """Test known occupation parameters."""
        line = ConstraintLine('occ Me* L(1-3) = total 1.0')
        ptrh_tree.apply_explicit_constraint(line)
        bounds_line = OccDeltaLine('Me* L(1-3) = Rh 0.10 0.90, Pt 0.90 0.10')
        ptrh_tree.apply_bounds(bounds_line)

        ptrh_tree.apply_implicit_constraints()
        ptrh_tree.finalize_tree()

        non_normalized_occupations = ptrh_tree(np.array(input_params))

        assert non_normalized_occupations == pytest.approx(
            self.KNOWN_OCC_PARAM_TO_VALUES[input_params]
        )
