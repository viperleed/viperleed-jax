import numpy as np
import pytest
from pytest_cases import fixture, parametrize_with_cases
from viperleed.calc.files.new_displacements.lines import (
    ConstraintLine,
    OffsetsLine,
    VibDeltaLine,
)

from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)
from viperleed_jax.transformation_tree.reduced_space import (
    apply_affine_to_subspace,
)
from viperleed_jax.transformation_tree.vib_parameters import VibTree

from ..structures import CaseStatesAfterInit


def test_tree_creation(atom_basis, fully_constrained_tree_template):
    """Test tree creation."""
    fully_constrained_tree_template(atom_basis, VibTree)

class TestFe2O3:
    """Test the Fe2O3 structure."""

    @fixture
    @parametrize_with_cases(
        'case', cases=CaseStatesAfterInit.case_fe2o3_012_converged
    )
    def fe2o3_tree(self, case):
        state, _ = case
        atom_basis = AtomBasis(state.slab)
        return VibTree(atom_basis)

    def test_apply_simple_offsets(self, fe2o3_tree):
        fe2o3_tree.apply_offsets(OffsetsLine('vib Fe 1 = 0.1'))
        assert 'Offset(vib Fe 1 = 0.1)' in str(fe2o3_tree)
        last_node = fe2o3_tree.nodes[-1]
        assert len(last_node.children) == 1
        assert last_node.children[0].transformer.biases == pytest.approx(
            np.array([0.1])
        )
        assert last_node.children[0].transformer.weights == pytest.approx(
            np.eye(1)
        )

    def test_apply_offsets_twice(self, fe2o3_tree):
        fe2o3_tree.apply_offsets(OffsetsLine('vib Fe 1 = 0.1'))
        with pytest.raises(ValueError, match='already defined'):
            fe2o3_tree.apply_offsets(OffsetsLine('vib Fe 1 = 0.1'))

    @pytest.mark.parametrize(
        'constraint',
        [
            'vib Fe L(1) = linked',
            'vib Fe L(1-2) = linked',
            'vib O L(1) = linked',
            'vib Fe_surf, O_surf = linked',
            'vib Fe_surf, O_surf = 1 Fe_surf',
            'vib Fe_surf, O_surf = 1.5 Fe_surf',
        ],
    )
    def test_apply_single_constraints(self, fe2o3_tree, constraint):
        # Apply constraints to the tree
        fe2o3_tree.apply_explicit_constraint(ConstraintLine(constraint))
        assert constraint in str(fe2o3_tree)

    def test_apply_wrong_constraint_type(self, fe2o3_tree):
        """Test applying a constraint with the wrong type."""
        vib_constraint = ConstraintLine('occ Fe_surf, O_surf = linked')
        with pytest.raises(ValueError, match='Wrong constraint type'):
            fe2o3_tree.apply_explicit_constraint(vib_constraint)
