"""Module meta_parameters."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-10-01'

import numpy as np

from .displacement_tree_layers import DisplacementTreeLayers
from .linear_transformer import LinearMap
from .nodes import (
    ImplicitLinearConstraintNode,
    LinearConstraintNode,
    LinearLeafNode,
)
from .reduced_space import Zonotope
from .tree import (
    LinearTree,
)

# Note: currently, V0r is (and can only be) a single parameter, which makes
# it much simpler than the other parameters. Further constraints are not
# implemented since there does not seem to be any use for them.
# In future, this could be extended to try and vary other parameters
# (e.g. incidence angle, etc.).

class V0rLeafNode(LinearLeafNode):
    """Leaf node for the V0r meta parameter."""

    def __init__(self):
        super().__init__(
            dof=1,  # V0r is a single scalar parameter
            name='V0r',
        )


class MetaTree(LinearTree):
    """Subtree for meta parameters."""

    def __init__(self):
        super().__init__(
            name='Meta Parameters (V0r)', root_node_name='V0r (root)'
        )

    def _initialize_tree(self):
        """Build the meta parameters subtree."""
        self.v0r_node = V0rLeafNode()
        self.nodes.append(self.v0r_node)
        # we need a "dummy" symmetry node on top of V0r to match the expected
        # structure of the tree
        free_constraint_node = LinearConstraintNode(
            dof=1,
            layer=DisplacementTreeLayers.Symmetry,
            name='V0r',
            children=[self.v0r_node],
            transformers=[LinearMap(np.eye(1), (1,))],
        )
        self.nodes.append(free_constraint_node)


    def apply_bounds(self, rpars):
        self._bound_V0r_from_rpars(rpars)

    def apply_implicit_constraints(self):
        raise NotImplementedError(
            'Meta parameters do not have implicit constraints.'
        )

    def apply_explicit_constraint(self):
        raise NotImplementedError(
            'Meta parameters do not have explicit constraints.'
        )

    def apply_offsets(self):
        """Apply the offsets to the meta parameters."""
        raise NotImplementedError(
            'Meta parameters do not have offsets to apply.'
        )

    def _analyze_tree(self):
        """Apply the offsets to the meta parameters."""
        raise NotImplementedError(
            'Meta parameters do not have offsets to apply.'
        )

    def _bound_V0r_from_rpars(self, rpars):
        """Read and apply the bounds of V0r from Rparams."""
        lower, upper = rpars.IV_SHIFT_RANGE.start, rpars.IV_SHIFT_RANGE.stop

        v0r_range = np.array(
            [[lower, upper]]
        ).T

        v0r_range_zonotope = Zonotope(
            basis=np.array([[1.0]]),  # 1D zonotope
            ranges=v0r_range,
            offset=None,
        )

        implicit_constraint_node = ImplicitLinearConstraintNode(
            child=self.v0r_node.parent,
            name=f'V0r({lower:.2f} eV, {upper:.2f} eV)',
            child_zonotope=v0r_range_zonotope,
        )
        self.nodes.append(implicit_constraint_node)
