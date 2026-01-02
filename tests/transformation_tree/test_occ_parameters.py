import numpy as np
import pytest
from pytest_cases import fixture, parametrize_with_cases
from viperleed.calc.files.new_displacements.lines import (
    ConstraintLine,
    OccDeltaLine,
    OffsetsLine,
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

    def test_simple_linked_constraints(self, ptrh_tree):
        """Test simple linked constraints."""

        # A constraint that links all occupations together
        line = ConstraintLine('occ Me* L(1-3) = linked')
        ptrh_tree.apply_explicit_constraint(line)
        bounds_line = OccDeltaLine('Me* L(1-3) = Rh 0.10 0.90, Pt 0.90 0.10')
        ptrh_tree.apply_bounds_line(bounds_line)
        ptrh_tree.apply_bounds()
        ptrh_tree.apply_implicit_constraints()
        ptrh_tree.finalize_tree()

        # all linked, so only one degree of freedom
        assert ptrh_tree.root.dof == 1

    def test_redundant_total_occ_constraint(self, ptrh_tree):
        """Test application of a redundant total occupation constraint."""

        # A constraint that links all occupations together
        line = ConstraintLine('occ Me* L(1-3) = linked')
        ptrh_tree.apply_explicit_constraint(line)

        line = ConstraintLine('occ Me* L(1-3) = total 1.0')
        with pytest.raises(ValueError):
            ptrh_tree.apply_explicit_constraint(line)

    @fixture
    @pytest.mark.parametrize('total_occ', [1.0, 0.5, 0.25])
    def total_occ_tree(self, ptrh_tree, total_occ):
        line = ConstraintLine(f'occ Me* L(1-3) = total {total_occ}')
        ptrh_tree.apply_explicit_constraint(line)
        bounds_line = OccDeltaLine('Me* L(1-3) = Rh 0.10 0.90, Pt 0.90 0.10')
        ptrh_tree.apply_bounds_line(bounds_line)
        ptrh_tree.apply_bounds()

        ptrh_tree.apply_implicit_constraints()
        ptrh_tree.finalize_tree()
        return ptrh_tree, total_occ

    def test_tree_is_not_centered(self, total_occ_tree):
        """Test that the tree is not centered.

        It should not be centered because the reference occupations are 25/75%
        while the range is symmetric around 50/50%!"""
        tree, _ = total_occ_tree
        assert not tree.is_centered()

    KNOWN_OCC_PARAM_TO_VALUES = {
        tuple([0.0] * 6): np.array(
            [
                1.0,
                1.0,
                0.1,
                0.9,
                0.1,
                0.9,
                0.1,
                0.9,
                0.1,
                0.9,
                0.1,
                0.9,
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
                0.9,
                0.1,
                0.9,
                0.1,
                0.9,
                0.1,
                0.9,
                0.1,
                0.9,
                0.1,
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
        ptrh_tree.apply_bounds_line(bounds_line)
        ptrh_tree.apply_bounds()

        ptrh_tree.apply_implicit_constraints()
        ptrh_tree.finalize_tree()

        non_normalized_occupations = ptrh_tree(np.array(input_params))

        assert non_normalized_occupations == pytest.approx(
            self.KNOWN_OCC_PARAM_TO_VALUES[input_params]
        )
