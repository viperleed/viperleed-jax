"""Tests for the abstract linear tree node."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-23'

import numpy as np
import pytest

from viperleed_jax.parameters.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.parameters.linear_transformer import LinearTransformer
from viperleed_jax.parameters.linear_tree_nodes import (
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
        if not isinstance(transformer, LinearTransformer):
            raise TypeError('Invalid transformer type.')
        if transformer.out_dim != self.dof:
            raise ValueError('Transformer output dimension mismatch.')
        self._transformer = transformer


def test_transformation_tree_node():
    """Test basic functionality of TransformationTreeNode."""
    node = MockTransformationTreeNode(name='root')
    assert node._name == 'root'
    with pytest.raises(ValueError):
        _ = node.transformer  # Transformer is not set

    transformer = LinearTransformer(weights=np.eye(2), biases=np.zeros(2))
    node.set_transformer(transformer)
    assert node.transformer is transformer


def test_linear_tree_node_transformer_validation():
    """Test transformer validation in LinearTreeNode."""
    node = MockLinearTreeNode(dof=2, layer=DisplacementTreeLayers.Base)
    valid_transformer = LinearTransformer(weights=np.eye(2), biases=np.zeros(2))
    invalid_transformer = LinearTransformer(
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
    node = MockTransformationTreeNode(name='child')
    with pytest.raises(
        RuntimeError,
        match='Transformation trees do not support detaching nodes.',
    ):
        node._pre_detach()
