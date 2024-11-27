import pytest
import numpy as np
from pytest_cases import parametrize_with_cases

from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.base_scatterers import BaseScatterers
from viperleed_jax.transformation_tree.geo_parameters import GeoTree

from ..structures import CaseStatesAfterInit, Tag


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
        assert tree.subtree_root.is_root
        assert tree.subtree_root.is_leaf is False

@parametrize_with_cases('test_case', cases=CaseStatesAfterInit)
def test_symmetry_operations_determinant(test_case):
    """The abs of the determinant of symmetry operations should be 1."""
    state, _ = test_case
    base_scatterers = BaseScatterers(state.slab)
    # create the geometry tree
    tree = GeoTree(base_scatterers)

    sym_ops = [
        leaf.symmetry_operation_to_reference_propagator for leaf in tree.leaves
    ]
    sym_op_dets = np.array([np.linalg.det(sym_op) for sym_op in sym_ops])
    assert abs(sym_op_dets) == pytest.approx(1.0)
