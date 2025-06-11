"""Tests for meta_parameters."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-06-11'


from types import SimpleNamespace

import numpy as np
import pytest

from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)
from viperleed_jax.transformation_tree.meta_parameters import MetaTree


class DummyRParams:
    """Simple dummy rparams class to simulate IV_SHIFT_RANGE."""
    def __init__(self, start, stop):
        self.IV_SHIFT_RANGE = SimpleNamespace(start=start, stop=stop)


def test_tree_initialization():
    """Test that the meta parameter tree initializes with expected structure."""
    tree = MetaTree()
    assert len(tree.nodes) == 2
    assert tree.nodes[0].name == '(1) V0r' # leaf node
    assert tree.nodes[1].name == '(1) V0r' # dummy symmetry node


def test_apply_bounds_creates_implicit_constraint():
    """Test that applying bounds adds an implicit constraint node."""
    tree = MetaTree()
    rpars = DummyRParams(start=-0.5, stop=0.5)
    tree.apply_bounds(rpars)

    assert any('V0r' in node.name and 'eV' in node.name for node in tree.nodes)
    implicit_nodes = [node for node in tree.nodes if 'eV' in node.name]
    assert len(implicit_nodes) == 1

    expected_transformer = AffineTransformer(
        weights=np.eye(1), biases=np.array([-0.5]), out_reshape=(1,)
    )
    assert tree.v0r_node.parent.transformer == expected_transformer


def test_explicit_constraint_raises():
    tree = MetaTree()
    with pytest.raises(NotImplementedError):
        tree.apply_explicit_constraint()


def test_implicit_constraint_raises():
    tree = MetaTree()
    with pytest.raises(NotImplementedError):
        tree.apply_implicit_constraints()


def test_apply_offsets_raises():
    tree = MetaTree()
    with pytest.raises(NotImplementedError):
        tree.apply_offsets()
