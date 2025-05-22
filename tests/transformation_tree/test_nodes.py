"""Tests for the abstract linear tree node."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-23'

import numpy as np
import pytest

from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.errors import (
    InvalidNodeError,
    NodeCreationError,
    TransformationTreeError,
)
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)
from viperleed_jax.transformation_tree.nodes import (
    LinearLeafNode,
    LinearTreeNode,
    TransformationTreeNode,
)


class MockTransformationTreeNode(TransformationTreeNode):
    def set_transformer(self, transformer):
        self._transformer = transformer


class MockLinearTreeNode(LinearTreeNode):
    def __init__(self, dof, layer, name=None, parent=None, children=None):
        super().__init__(
            dof, layer, name=name, parent=parent, children=children
        )

    def set_transformer(self, transformer):
        if not isinstance(transformer, AffineTransformer):
            raise TypeError('Invalid transformer type.')
        if transformer.out_dim != self.dof:
            raise ValueError('Transformer output dimension mismatch.')
        self._transformer = transformer


@pytest.fixture
def dummy_leaf_node():
    """Fixture for a dummy leaf node."""
    return LinearLeafNode(dof=3, name='dummy_leaf')

def test_transformation_tree_node():
    """Test basic functionality of TransformationTreeNode."""
    node = MockTransformationTreeNode(name='root', parent=None)
    assert node._name == 'root'
    with pytest.raises(InvalidNodeError):
        _ = node.transformer  # Transformer is not set

    transformer = AffineTransformer(weights=np.eye(2), biases=np.zeros(2))
    node.set_transformer(transformer)
    assert node.transformer is transformer


def test_type_error_set_transformer(dummy_leaf_node):
    """Test basic functionality of TransformationTreeNode."""

    not_a_transformer = np.eye(2)
    with pytest.raises(TypeError):
        dummy_leaf_node.set_transformer(not_a_transformer)


def test_linear_tree_node_transformer_validation():
    """Test transformer validation in LinearTreeNode."""
    node = MockLinearTreeNode(dof=2, layer=DisplacementTreeLayers.Base)
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


def test_detaching_parent_not_allowed():
    """Test that detaching parents in TransformationTreeNode is not allowed."""
    # Create a mock node
    node = MockTransformationTreeNode(name='child', parent=None)
    with pytest.raises(
        TransformationTreeError,
        match='Transformation trees do not support detaching nodes.',
    ):
        node._pre_detach()


# Test the name property of LinearTreeNode
def test_linear_tree_node_name_property():
    node = MockLinearTreeNode(
        dof=3, layer=DisplacementTreeLayers.Base, name='foo')
    assert node._name == 'foo'
    assert node.name == '(3) foo'


# Test LinearLeafNode properties and transformer setting
def test_linear_leaf_node_properties_and_transformer():

    node = LinearLeafNode(dof=2, name='leaf')
    assert node.dof == 2
    assert node.name == '(2) leaf'

    # test transformer setting
    valid_transformer = AffineTransformer(weights=np.eye(2), biases=np.zeros(2))
    node.set_transformer(valid_transformer)
    assert node.transformer == valid_transformer


# Test parent-child relationship in tree nodes
def test_parent_child_relationship_in_tree_nodes():
    parent = MockTransformationTreeNode(name='parent', parent=None)
    child = MockTransformationTreeNode(name='child', parent=parent)
    assert child.parent is parent
    assert child in parent.children
