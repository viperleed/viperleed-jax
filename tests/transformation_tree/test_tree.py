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
    def __init__(self, dof, atom):
        self.dof = dof
        self.is_leaf = True
        self.is_root = True
        self.atom = atom


class MockLinearTree(LinearTree):
    """Mock implementation of LinearTree."""

    def _initialize_tree(self):
        """Mock implementation of build_tree."""
        leaf1 = MockLeafNode(dof=1, atom='Atom1')
        leaf2 = MockLeafNode(dof=1, atom='Atom2')
        leaf3 = MockLeafNode(dof=1, atom='Atom3')

        self.nodes.extend([leaf1, leaf2, leaf3])

    def finalize_tree(self):
        """Mock implementation of create_subtree_root."""
        super().finalize_tree()


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
    assert not linear_tree.finalized
    linear_tree.create_subtree_root()
    assert linear_tree.finalized
    assert linear_tree.root.dof == 3
