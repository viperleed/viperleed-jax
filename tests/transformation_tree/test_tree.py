"""Tests for the abstract transformation tree."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-25'

import pytest

from viperleed_jax.transformation_tree.tree import (
    LinearTree,
    TransformationTree,
)


class MockAtom:
    def __init__(self):
        self.scatterers = ['Atom1', 'Atom2', 'Atom3']
        self.site_elements = ['H', 'O', 'C']


class MockNode:
    def __init__(self):
        pass

    def set_transformer(self, transformer):
        self.transformer = transformer


class MockRootNode(MockNode):
    def __init__(self, dof):
        self.dof = dof
        self.is_root = True
        self.is_leaf = False
        self.children = []


class MockLeafNode(MockNode):
    def __init__(self, dof, base_scatterer):
        self.dof = dof
        self.is_leaf = True
        self.is_root = True
        self.base_scatterer = base_scatterer


class MockLinearTree(LinearTree):
    """Mock implementation of LinearTree."""

    def build_tree(self):
        """Mock implementation of build_tree."""
        leaf1 = MockLeafNode(dof=1, base_scatterer='Atom1')
        leaf2 = MockLeafNode(dof=1, base_scatterer='Atom2')
        leaf3 = MockLeafNode(dof=1, base_scatterer='Atom3')

        self.nodes.extend([leaf1, leaf2, leaf3])

    def create_root(self):
        """Mock implementation of create_subtree_root."""
        super().create_root()


@pytest.fixture
def linear_tree():
    """Fixture for a mock linear tree."""
    return MockLinearTree(name='Linear Test Tree', root_node_name='Root')


# Test Abstract Base Class
def test_abstract_transformation_tree():
    with pytest.raises(TypeError):
        TransformationTree(name='Abstract Tree', root_node_name='Root')


# Test LinearTree
def test_linear_tree_create_subtree_root(linear_tree):
    assert not linear_tree._tree_root_has_been_created
    linear_tree.create_subtree_root()
    assert linear_tree._tree_root_has_been_created
    assert linear_tree.root.dof == 3
