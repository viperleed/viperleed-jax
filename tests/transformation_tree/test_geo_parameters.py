import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from viperleed_jax.base_scatterers import BaseScatterers
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.geo_parameters import GeoTree

from ..structures import CaseStatesAfterInit


@parametrize_with_cases('test_case', cases=CaseStatesAfterInit)
def test_tree_creation(test_case, subtests):
    """Test tree creation."""
    state, info = test_case
    base_scatterers = BaseScatterers(state.slab)
    # create the geometry tree
    tree = GeoTree(base_scatterers)
    assert len(tree.leaves) == len(base_scatterers)

    with subtests.test('tree root creation'):
        tree.create_root()
        assert tree.root.is_root
        assert tree.root.is_leaf is False


@parametrize_with_cases('test_case', cases=CaseStatesAfterInit)
def test_symmetry_operations_determinant(test_case, subtests):
    """The abs of the determinant of symmetry operations should be 1."""
    state, _ = test_case
    base_scatterers = BaseScatterers(state.slab)
    # create the geometry tree
    tree = GeoTree(base_scatterers)
    tree.create_root()

    symmetry_roots = tree.roots_up_to_layer(DisplacementTreeLayers.Symmetry)
    z_only_roots = [root for root in symmetry_roots if root.dof == 1]
    in_plane_1d_roots = [root for root in symmetry_roots if root.dof == 2]
    free_roots = [root for root in symmetry_roots if root.dof == 3]

    # TODO: figure out more through tests for the symmetry operations
    with subtests.test('z_only_roots'):
        for root in z_only_roots:
            for leaf in root.leaves:
                sym_op = leaf.symmetry_operation_to_reference_propagator
                assert sym_op[0, 0] == pytest.approx(1.0)

    with subtests.test('in_plane_1d_roots'):
        for root in in_plane_1d_roots:
            for leaf in root.leaves:
                sym_op = leaf.symmetry_operation_to_reference_propagator

    with subtests.test('free roots'):
        for root in free_roots:
            for leaf in root.leaves:
                sym_op = leaf.symmetry_operation_to_reference_propagator
                sym_op_det = np.linalg.det(sym_op)
                assert abs(sym_op_det) == pytest.approx(1.0)
