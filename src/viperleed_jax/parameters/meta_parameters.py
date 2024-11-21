"""Module meta_parameters."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-10-01'

from jax import numpy as jnp

from .hierarchical_linear_tree import (
    HLBound,
    HLConstraintNode,
    HLLeafNode,
    LinearTree,
    DisplacementTreeLayers,
)
from .linear_transformer import LinearTransformer

# Note: currently, V0r is (and can only be) a single parameter, which makes
# it much simpler than the other parameters. Further constraints are not
# implemented since there does not seem to be any use for them.
# In future, this could be extended to try and vary other parameters
# (e.g. incidence angle, etc.).


class MetaParameterSubtree(LinearTree):
    """Subtree for meta parameters."""

    def __init__(self):
        super().__init__()

    def build_subtree(self):
        # V0r
        self.v0r_node = V0rHLLeafNode()
        self.nodes.append(self.v0r_node)

    def read_from_rpars(self, rpars):
        # V0r
        self.v0r_node.update_bounds(rpars)
        bound_node = V0rBoundNode(self.v0r_node)
        self.nodes.append(bound_node)
        self.create_subtree_root()

    @property
    def name(self):
        return 'Meta Parameters (V0r)'

    @property
    def subtree_root_name(self):
        return 'V0r (root)'


class V0rHLLeafNode(HLLeafNode):
    def __init__(self):
        dof = 1  # V0r is a single scalar parameter
        name = 'V0r'
        self.bound = HLBound(1)
        self.num = 1
        super().__init__(dof=dof, name=name)

    def update_bounds(self, rpars):
        lower, upper = rpars.IV_SHIFT_RANGE.start, rpars.IV_SHIFT_RANGE.stop
        self.bound.update_range(
            _range=(lower, upper), offset=None, enforce=True
        )


class V0rBoundNode(HLConstraintNode):
    def __init__(self, child):
        if not isinstance(child, V0rHLLeafNode):
            raise ValueError('V0rBoundNode must have a single leaf child.')

        weights = [child.bound.upper - child.bound.lower]
        biases = child.bound.lower
        transformer = LinearTransformer(weights, biases, (1,))

        super().__init__(
            dof=1,
            name='Simple Bound',
            children=[child],
            transformers=[transformer],
            layer=DisplacementTreeLayers.Implicit_Constraints,
        )
