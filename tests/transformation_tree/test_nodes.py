"""Tests for the abstract linear tree node."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-23'

import numpy as np
import pytest

from viperleed_jax.atom_basis import SiteEl
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.errors import (
    InvalidNodeError,
    TransformationTreeError,
)
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
    LinearMap,
)
from viperleed_jax.transformation_tree.nodes import (
    AtomicLinearNode,
    LinearConstraintNode,
    LinearLeafNode,
    LinearTreeNode,
    TransformationTreeNode,
)


@pytest.fixture
def dummy_leaf_node():
    """Fixture for a dummy leaf node."""
    return LinearLeafNode(dof=3, name='dummy_leaf')


def test_transformation_tree_node(dummy_leaf_node):
    """Test basic functionality of TransformationTreeNode."""
    assert dummy_leaf_node._name == 'dummy_leaf'
    with pytest.raises(InvalidNodeError):
        _ = dummy_leaf_node.transformer  # Transformer is not set

    transformer = AffineTransformer(weights=np.eye(3), biases=np.zeros(3))
    dummy_leaf_node.set_transformer(transformer)
    assert dummy_leaf_node.transformer is transformer


def test_type_error_set_transformer(dummy_leaf_node):
    """Test basic functionality of TransformationTreeNode."""
    not_a_transformer = np.eye(3)
    with pytest.raises(TypeError):
        dummy_leaf_node.set_transformer(not_a_transformer)


def test_detaching_parent_not_allowed():
    """Test that detaching parents in TransformationTreeNode is not allowed."""
    # Create a mock node
    node = TransformationTreeNode(name='child', parent=None)
    with pytest.raises(
        TransformationTreeError,
        match='Transformation trees do not support detaching nodes.',
    ):
        node._pre_detach()


# Test the name property of LinearTreeNode
def test_linear_tree_node_name_property():
    node = LinearLeafNode(dof=3, name='foo')
    assert node._name == 'foo'
    assert node.name == '(3) foo'

class TestLinearLeafNode:
    # Test LinearLeafNode properties and transformer setting
    def test_linear_leaf_node_properties_and_transformer(self):

        node = LinearLeafNode(dof=2, name='leaf')
        assert node.dof == 2
        assert node.name == '(2) leaf'

        # test transformer setting
        valid_transformer = AffineTransformer(weights=np.eye(2), biases=np.zeros(2))
        node.set_transformer(valid_transformer)
        assert node.transformer == valid_transformer


def test_transformer_validation():
    """Test transformer validation in LinearTreeNode."""
    node = LinearLeafNode(dof=2, name='leaf')
    valid_transformer = AffineTransformer(weights=np.eye(2), biases=np.zeros(2))
    invalid_transformer = AffineTransformer(
        weights=np.eye(3), biases=np.zeros(3)
    )

    node.set_transformer(valid_transformer)
    assert node.transformer == valid_transformer

    with pytest.raises(ValueError):
        node.set_transformer(
            invalid_transformer
        )  # Mismatched transformer output dimension


def test_parent_child_relationship_in_tree_nodes():
    grandparent = TransformationTreeNode(name='grandparent', parent=None)
    parent = TransformationTreeNode(name='parent', parent=grandparent)
    child = TransformationTreeNode(name='child', parent=parent)
    assert child.parent is parent
    assert child in parent.children
    assert parent.parent is grandparent
    assert child in grandparent.descendants
    assert grandparent in child.ancestors


    def test_transformer_to_unrelated_node(dummy_leaf_node):
        """Test that transformer_to_descendent raises for unrelated nodes."""
        unrelated_node = LinearLeafNode(dof=3, name='unrelated')
        with pytest.raises(ValueError):
            dummy_leaf_node.transformer_to_descendent(unrelated_node)

class TestConstraintNode:
    @pytest.fixture
    def direct_map(self):
        """Fixture for a direct map."""
        return LinearMap(np.eye(3))


# Test parent-child relationship in tree nodes
    def test_parent_child_relationship_in_linear_tree_nodes(
        self, dummy_leaf_node, direct_map
    ):
        child = dummy_leaf_node
        parent = LinearConstraintNode(
            dof=3, layer=DisplacementTreeLayers.Symmetry,
            name='sym_parent',
            transformers=[direct_map],
            children=[child]
        )

        # test transformer to descendent
        assert parent.transformer_to_descendent(child) == direct_map

    def test_error_layer_mismatch(self, direct_map):
        """Test that a node cannot be added to a parent with a lower layer."""
        child = LinearTreeNode(dof=3, name='child', layer=DisplacementTreeLayers.Root)

        with pytest.raises(InvalidNodeError):
            LinearConstraintNode(
                dof=3, layer=DisplacementTreeLayers.Base,
                name='parent', children=[child], transformers=[direct_map]
            )

    def test_constraint_node_children(self, direct_map):
        """Test that a constraint node must have children."""
        with pytest.raises(InvalidNodeError):
            LinearConstraintNode(
                dof=3, layer=DisplacementTreeLayers.Symmetry,
                name='parent', children=[], transformers=[direct_map]
            )

    def test_constraint_node_dof_larger_than_children(self, direct_map):
        """Test that a constraint node must have dof <= dof of children."""
        child = LinearLeafNode(dof=3, name='child')
        with pytest.raises(InvalidNodeError):
            LinearConstraintNode(
                dof=4, layer=DisplacementTreeLayers.Symmetry,
                name='parent', children=[child], transformers=[direct_map]
            )


class TestAtomicLinearNode:
    class MockAtom:
        def __init__(self):
            self.atom = None
            self.num = 1
            self.site_element = SiteEl('site', 'Fe')
            self.site = self.site_element.site
            self.element = self.site_element.element

    def test_atomic_linear_node(self):
        """Test AtomicLinearNode properties and transformer setting."""
        atom = self.MockAtom()
        node = AtomicLinearNode(atom=atom, dof=3, name='atomic_node')
        assert node.dof == 3
        assert node.name == '(3) atomic_node'
        assert node.atom is atom
        assert node.element == 'Fe'
        assert node.site == 'site'
