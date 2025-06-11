import numpy as np
import pytest
from pytest_cases import fixture, parametrize_with_cases
from viperleed.calc.files.new_displacements.lines import (
    ConstraintLine,
    OccDeltaLine,
    OffsetsLine,
)

from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)
from viperleed_jax.transformation_tree.occ_parameters import OccTree
from viperleed_jax.transformation_tree.reduced_space import (
    apply_affine_to_subspace,
)

from ..structures import CaseStatesAfterInit


def test_tree_creation(atom_basis, subtests):
    """Test tree creation."""
    # create the geometry tree
    tree = OccTree(atom_basis)
    assert len(tree.leaves) == len(atom_basis)

    with subtests.test('tree root creation'):
        # apply implicit constraints to unmodified tree
        tree.apply_implicit_constraints()
        tree.finalize_tree()
        assert tree.root.is_root
        assert tree.root.is_leaf is False
