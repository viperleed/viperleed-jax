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
from viperleed_jax.transformation_tree.vib_parameters import (
    VibConstraintNode,
    VibLeafNode,
    VibSymmetryConstraint,
    VibTree,
)

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

    def test_more_than_one_dof_constraint(self, fe2o3_tree):
        """Test applying a constraint with more than one DOF."""
        with pytest.raises(
            ValueError, match='Vibrational constraints must have dof=1'
        ):
            VibConstraintNode(
                children=[fe2o3_tree.leaves[0]],
                name='test',
                layer=DisplacementTreeLayers.User_Constraints,
                dof=2,
            )

    def test_symmetry_node_with_non_leaf_children(self, fe2o3_tree):
        """Test that symmetry nodes can have non-leaf children."""
        # after tree creation, the roots are symmetry nodes (non-leaf)
        symmetry_node = fe2o3_tree.roots[0]

        with pytest.raises(ValueError, match='Children must be VibLeaf nodes.'):
            VibSymmetryConstraint(
                children=[symmetry_node],
            )

    def test_symmetry_node_with_different_site_elements(self, fe2o3_tree):
        """Test that symmetry nodes require the same site-element."""
        leaf_1 = VibLeafNode(fe2o3_tree.leaves[0].atom)
        leaf_2 = VibLeafNode(fe2o3_tree.leaves[6].atom)

        with pytest.raises(
            ValueError, match='Children must have the same site-element.'
        ):
            VibSymmetryConstraint(
                children=[leaf_1, leaf_2],
            )

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

    def test_apply_multiple_constraints(self, fe2o3_tree, subtests):
        """Test applying multiple interconnected constraints."""
        assert sum(root.dof for root in fe2o3_tree.roots) == 15
        fe2o3_tree.apply_explicit_constraint(
            ConstraintLine('vib Fe L(1) = linked')
        )
        assert sum(root.dof for root in fe2o3_tree.roots) == 14
        fe2o3_tree.apply_explicit_constraint(
            ConstraintLine('vib Fe L(2) = linked')
        )
        assert sum(root.dof for root in fe2o3_tree.roots) == 13
        with subtests.test('apply layered constraints'):
            fe2o3_tree.apply_explicit_constraint(
                ConstraintLine('vib Fe L(1-2) = linked')
            )
            assert sum(root.dof for root in fe2o3_tree.roots) == 12
        with subtests.test('finalize layered constraints'):
            fe2o3_tree.apply_implicit_constraints()
            fe2o3_tree.finalize_tree()

    @pytest.mark.parametrize(
        'constraints',
        [
            [
                'vib Fe L(1) = linked',
                'vib Fe 1-4 = linked',
            ],
            [
                'vib Fe L(1-2) = linked',
                'vib Fe L(1) = linked',
            ],
            [
                'vib Fe 1 = Fe 2',
            ],
        ],
    )
    def test_error_redundant_constraints(self, fe2o3_tree, constraints):
        """Test that redundant constraints raise an error."""
        with pytest.raises(ValueError, match='redundant'):  # noqa: PT012
            for constraint in constraints:
                fe2o3_tree.apply_explicit_constraint(ConstraintLine(constraint))

    @pytest.mark.parametrize(
        'bounds_line,implicit_dof',
        [
            ('Fe_surf = -0.1 0.1', 1),
            ('O_surf = 0.1 -0.1', 1),
            ('Fe L(1-2) = -0.1 0.2', 4),  # 4 indep. atoms, 1 dof each
        ],
    )
    def test_apply_single_vib_delta(
        self, fe2o3_tree, bounds_line, implicit_dof, subtests
    ):
        """Test applying a single VIB_DELTA line."""
        vib_delta_line = VibDeltaLine(bounds_line)
        fe2o3_tree.apply_bounds(vib_delta_line)
        assert bounds_line in str(fe2o3_tree)

        # apply implicit constraints
        fe2o3_tree.apply_implicit_constraints()
        with subtests.test('check dof'):
            # finalize the tree to apply the bounds
            assert sum(root.dof for root in fe2o3_tree.roots) == implicit_dof
