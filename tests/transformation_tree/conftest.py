"""Conftest for tests.transformation_tree."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-06-11'


import pytest
from pytest_cases import fixture


@fixture
def fully_constrained_tree_template(subtests):
    """Test tree creation."""
    def fully_constrained_tree_test(atom_basis, tree_class):
        # create the geometry tree
        tree = tree_class(atom_basis)
        assert len(tree.leaves) == len(atom_basis)

        with subtests.test('tree root creation'):
            # apply implicit constraints to unmodified tree
            tree.apply_implicit_constraints()
            tree.finalize_tree()
            assert tree.root.is_root
            assert tree.root.is_leaf is False
            # no ranges set -> DOF should be 0
            assert sum(node.dof for node in tree.roots) == 0
    return fully_constrained_tree_test
