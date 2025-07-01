"""Tests for the abstract transformation tree."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-25'

import pytest

from viperleed_jax.transformation_tree.tree import (
    LinearTree,
    TransformationTree,
)


# Test Abstract Base Class
def test_abstract_transformation_tree():
    with pytest.raises(TypeError):
        TransformationTree(name='Abstract Tree', root_node_name='Root')


# # Test LinearTree
# def test_linear_tree_create_subtree_root(linear_tree):
#     assert not linear_tree.finalized
#     linear_tree.create_subtree_root()
#     assert linear_tree.finalized
#     assert linear_tree.root.dof == 3


# Additional tests for LinearTree using real nodes and transformers
import numpy as np
import pytest

from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)
from viperleed_jax.transformation_tree.nodes import (
    LinearConstraintNode,
    LinearLeafNode,
)


def test_linear_tree_add_leaf_node_and_finalize():
    tree = LinearTree(name='Test Tree', root_node_name='Root')
    leaf1 = LinearLeafNode(dof=2, name='leaf1')
    leaf2 = LinearLeafNode(dof=1, name='leaf2')

    trafo1 = AffineTransformer(weights=np.eye(2), biases=np.zeros(2))
    trafo2 = AffineTransformer(weights=np.ones((1, 2)), biases=np.zeros(1))

    leaf1.set_transformer(trafo1)
    leaf2.set_transformer(trafo2)

    root = LinearConstraintNode(
        dof=2,
        layer=DisplacementTreeLayers.User_Constraints,
        name='constraint',
        children=[leaf1, leaf2],
        transformers=[trafo1, trafo2],
    )
    tree.add_node(root)
    tree.finalize_tree()
    assert tree.root == root
    assert tree.finalized
    assert tree.root.dof == 2


def test_linear_tree_finalize_fails_without_root():
    tree = LinearTree(name='Test Tree', root_node_name='Root')
    with pytest.raises(
        ValueError, match="Tree does not contain a node with name 'Root'"
    ):
        tree.finalize_tree()
