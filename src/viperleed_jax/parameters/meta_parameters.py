"""Module meta_parameters."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-10-01'


from .displacement_range import DisplacementRange
from .displacement_tree_layers import DisplacementTreeLayers
from .hierarchical_linear_tree import (
    LinearTree,
)
from .linear_transformer import LinearTransformer
from .linear_tree_nodes import LinearConstraintNode, LinearLeafNode

# Note: currently, V0r is (and can only be) a single parameter, which makes
# it much simpler than the other parameters. Further constraints are not
# implemented since there does not seem to be any use for them.
# In future, this could be extended to try and vary other parameters
# (e.g. incidence angle, etc.).


class MetaParameterSubtree(LinearTree):
    """Subtree for meta parameters."""

    def __init__(self):
        super().__init__()

    def build_tree(self):
        # called in init
        # V0r
        self.v0r_node = V0rHLLeafNode()
        self.nodes.append(self.v0r_node)

    def read_from_rpars(self, rpars):
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


class V0rHLLeafNode(LinearLeafNode):
    def __init__(self):
        dof = 1  # V0r is a single scalar parameter
        name = 'V0r'
        self.bound = DisplacementRange(1)
        self.num = 1
        super().__init__(dof=dof, name=name)

    def update_bounds(self, rpars):
        lower, upper = rpars.IV_SHIFT_RANGE.start, rpars.IV_SHIFT_RANGE.stop
        self.bound.update_range(
            _range=(lower, upper), offset=None, enforce=True
        )


class V0rBoundNode(LinearConstraintNode):
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
