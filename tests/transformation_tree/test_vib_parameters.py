import numpy as np
import pytest
from pytest_cases import fixture, parametrize_with_cases

from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.files.displacements.lines import (
    ConstraintLine,
    OffsetsLine,
    VibDeltaLine,
)
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)
from viperleed_jax.transformation_tree.reduced_space import (
    apply_affine_to_subspace,
)
from viperleed_jax.transformation_tree.vib_parameters import VibTree

from ..structures import CaseStatesAfterInit


class TestFe2O3:
    """Test the Fe2O3 structure."""

    @fixture
    @parametrize_with_cases(
        'case', cases=CaseStatesAfterInit.case_fe2o3_012_converged
    )
    def fe2o3_tree(self, case):
        state, _ = case
        atom_basis = AtomBasis(state.slab)
        return VibTree(atom_basis)

    def test_apply_offsets(self, fe2o3_tree):
        fe2o3_tree.apply_offsets(OffsetsLine('vib Fe 1 = 0.1'))
        assert 'Offset(vib Fe 1 = 0.1)' in str(fe2o3_tree)
        last_node = fe2o3_tree.nodes[-1]
        assert len(last_node.children) == 1
        assert last_node.children[0].transformer.biases == pytest.approx(
            np.array(0.1)
        )
        assert last_node.children[0].transformer.weights == pytest.approx(
            np.eye(1)
        )

