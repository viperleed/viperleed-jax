import numpy as np
import pytest
from pytest_cases import fixture, parametrize_with_cases
from viperleed.calc.files.new_displacements.lines import (
    ConstraintLine,
    GeoDeltaLine,
    OffsetsLine,
)

from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.geo_parameters import GeoTree
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)
from viperleed_jax.transformation_tree.reduced_space import (
    apply_affine_to_subspace,
)

from ..structures import CaseStatesAfterInit


def test_tree_creation(atom_basis, subtests):
    """Test tree creation."""
    # create the geometry tree
    tree = GeoTree(atom_basis)
    assert len(tree.leaves) == len(atom_basis)

    with subtests.test('tree root creation'):
        # apply implicit constraints to unmodified tree
        tree.apply_implicit_constraints()
        tree.finalize_tree()
        assert tree.root.is_root
        assert tree.root.is_leaf is False


# def test_symmetry_operations_determinant(atom_basis, subtests):
#     """The abs of the determinant of symmetry operations should be 1."""
#     # create the geometry tree
#     tree = GeoTree(atom_basis)
#     tree.finalize_tree()

#     symmetry_roots = tree.roots_up_to_layer(DisplacementTreeLayers.Symmetry)
#     z_only_roots = [root for root in symmetry_roots if root.dof == 1]
#     in_plane_1d_roots = [root for root in symmetry_roots if root.dof == 2]
#     free_roots = [root for root in symmetry_roots if root.dof == 3]

#     # TODO: figure out more through tests for the symmetry operations
#     with subtests.test('z_only_roots'):
#         for root in z_only_roots:
#             for leaf in root.leaves:
#                 sym_op = leaf.symmetry_operation_to_reference_propagator
#                 assert sym_op[0, 0] == pytest.approx(1.0)

#     with subtests.test('in_plane_1d_roots'):
#         for root in in_plane_1d_roots:
#             for leaf in root.leaves:
#                 sym_op = leaf.symmetry_operation_to_reference_propagator

#     with subtests.test('free roots'):
#         for root in free_roots:
#             for leaf in root.leaves:
#                 sym_op = leaf.symmetry_operation_to_reference_propagator
#                 sym_op_det = np.linalg.det(sym_op)
#                 assert abs(sym_op_det) == pytest.approx(1.0)

class TestFe2O3:
    """Test the Fe2O3 structure."""

    @fixture
    @parametrize_with_cases('case', cases=CaseStatesAfterInit.case_fe2o3_012_converged)
    def fe2o3_tree(self, case):
        state, _ = case
        atom_basis =  AtomBasis(state.slab)
        return GeoTree(atom_basis)

    def test_apply_offsets(self, fe2o3_tree):
        fe2o3_tree.apply_offsets(OffsetsLine('geo Fe 1 z = 0.1'))
        assert 'Offset(geo Fe 1 z = 0.1)' in str(fe2o3_tree)
        last_node = fe2o3_tree.nodes[-1]
        assert len(last_node.children) == 1
        assert last_node.children[0].transformer.biases == pytest.approx(
            np.array([0.1, 0.0, 0.0]))
        assert last_node.children[0].transformer.weights == pytest.approx(
            np.eye(3))

    def test_apply_offsets_twice(self, fe2o3_tree):
        fe2o3_tree.apply_offsets(OffsetsLine('geo Fe 1 z = 0.1'))
        with pytest.raises(ValueError, match='already defined'):
            fe2o3_tree.apply_offsets(OffsetsLine('geo Fe 1 z = 0.1'))

    @pytest.mark.parametrize(
        'constraint',
        [
            'geo Fe L(1) = linked',
            'geo Fe L(1-2) = linked',
            'geo O L(1) = linked',
            'geo Fe_surf, O_surf = linked',
            'geo Fe_surf, O_surf = 1 Fe_surf',
            'geo Fe_surf, O_surf = [[1 0 0] [0 0 1] [0 1 0]] Fe_surf',
        ],
    )
    def test_apply_single_constraints(self, fe2o3_tree, constraint):
        # Apply constraints to the tree
        fe2o3_tree.apply_explicit_constraint(ConstraintLine(constraint))
        assert constraint in str(fe2o3_tree)

    def test_apply_wrong_constraint_type(self, fe2o3_tree):
        """Test applying a constraint with the wrong type."""
        vib_constraint = ConstraintLine('vib Fe_surf, O_surf = linked')
        with pytest.raises(ValueError, match='Wrong constraint type'):
            fe2o3_tree.apply_explicit_constraint(vib_constraint)


    def test_apply_multiple_constraints(self, fe2o3_tree, subtests):
        """Test applying multiple interconnected constraints."""
        assert sum(root.dof for root in fe2o3_tree.roots) == 45
        fe2o3_tree.apply_explicit_constraint(
            ConstraintLine('geo Fe L(1) = linked'))
        assert sum(root.dof for root in fe2o3_tree.roots) == 42
        fe2o3_tree.apply_explicit_constraint(
            ConstraintLine('geo Fe L(2) = linked'))
        assert sum(root.dof for root in fe2o3_tree.roots) == 39
        with subtests.test('apply layered constraints'):
            fe2o3_tree.apply_explicit_constraint(
                ConstraintLine('geo Fe L(1-2) = linked'))
            assert sum(root.dof for root in fe2o3_tree.roots) == 36
        with subtests.test('finalize layered constraints'):
            fe2o3_tree.apply_implicit_constraints()
            fe2o3_tree.finalize_tree()

    @pytest.mark.parametrize(
        'constraints',
        [
            [
                'geo Fe L(1) = linked',
                'geo Fe 1-4 = linked',
            ],
            [
                'geo Fe L(1-2) = linked',
                'geo Fe L(1) = linked',
            ],
            [
                'geo Fe 1 = Fe 2',
            ],
        ],
    )
    def test_error_redundant_constraints(self, fe2o3_tree, constraints):
        """Test that redundant constraints raise an error."""
        with pytest.raises(ValueError, match='redundant'):                      # noqa: PT012
            for constraint in constraints:
                fe2o3_tree.apply_explicit_constraint(
                    ConstraintLine(constraint))


    @pytest.mark.parametrize(
        'bounds_line,implicit_dof',
        [
            ('Fe_surf xyz = -0.1 0.1', 3),
            ('Fe_surf xy = -0.1 0.1', 2),
            ('Fe_surf z = -5 5', 1),
        ]
    )
    def test_apply_single_geo_delta(
        self, fe2o3_tree, bounds_line, implicit_dof, subtests):
        """Test applying a single GeoDelta."""
        geo_delta_line = GeoDeltaLine(bounds_line)
        fe2o3_tree.apply_bounds(geo_delta_line)
        assert bounds_line in str(fe2o3_tree)

        # apply implicit constraints
        fe2o3_tree.apply_implicit_constraints()
        with subtests.test('check dof'):
            # finalize the tree to apply the bounds
            assert sum(root.dof for root in fe2o3_tree.roots) == implicit_dof

def test_reflection_preserves_ranges():
    # 1) Set up a 3×3 basis = identity
    basis = np.eye(3)
    # 2) Uniform ±0.1 range on each axis
    ranges = np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])
    # 3) Build a “mirror Y” + tiny cross‐coupling
    W = np.array(
        [
            [1.0,  0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0,  0.0, 1.0],
        ]
    )
    b = np.zeros(3)
    T = AffineTransformer(W, b)

    Bm, new_ranges, b_out = apply_affine_to_subspace(basis, ranges, T)

    # Expect Bm = W @ I = W
    assert np.allclose(Bm, W)
    # And ranges should be exactly the same as what we passed in
    assert np.allclose(abs(new_ranges), abs(ranges)), f'got {new_ranges}'
    # No bias in or out
    assert np.allclose(b_out, 0)


class TestBaseTree:
    """Test that tests methods of the base tree work as expected.

    These tests are not specific to the GeoTree, but rather test the
    methods of the base tree class.
    """

    def test_finalize_before_finalize(self, fe2o3_tree):
        """Test that creating a root before finalizing the tree works."""
        with pytest.raises(
            ValueError, match='Implicit constraints have not been applied'
        ):
            fe2o3_tree.finalize_tree()

    def test_create_root_again(self, fe2o3_tree):
        """Test that applying implicit constraints works."""
        # apply implicit constraints
        fe2o3_tree.apply_implicit_constraints()
        # finalize the tree to apply the bounds
        fe2o3_tree.finalize_tree()

        with pytest.raises(ValueError, match='has already been created'):
            fe2o3_tree.finalize_tree()

    def test_apply_not_a_constraints(self, fe2o3_tree):
        with pytest.raises(TypeError, match='must be a ConstraintLine'):
            fe2o3_tree.apply_explicit_constraint('this is not a constraint')
