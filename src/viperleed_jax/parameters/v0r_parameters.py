from jax import numpy as jnp

from viperleed_jax.base import LinearTransformer
from viperleed_jax.parameters.base_parameters import Bound
from viperleed_jax.parameters.hierarchical_linear_tree import HLSubtree
from viperleed_jax.parameters.hierarchical_linear_tree import HLLeafNode
from viperleed_jax.parameters.hierarchical_linear_tree import HLBound


# Note: currently, V0r is (and can only be) a single parameter, which makes
# it much simpler than the other parameters. Further constraints are not
# implemented since there does not seem to be any use for them.
# In future, this could be extended to try and vary other parameters
# (e.g. incidence angle, etc.).


class MetaParameterSubtree(HLSubtree):
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
        self.create_subtree_root()

    @property
    def name(self):
        return "Meta Parameters (V0r)"

    @property
    def subtree_root_name(self):
        return "V0r (root)"


class V0rHLLeafNode(HLLeafNode):

    def __init__(self):
        dof = 1  # V0r is a single scalar parameter
        name = "V0r"
        self.bound = HLBound(1)
        super().__init__(dof=dof, name=name)

    def update_bounds(self, rpars):
        lower, upper = rpars.IV_SHIFT_RANGE.start, rpars.IV_SHIFT_RANGE.stop
        self.bound.update_range(
            range=(lower, upper), offset=None, user_set=True
        )


class V0rParam():

    def __init__(self, delta_slab):

        self.n_free_params = 1
        self.n_base_params = 1
        self.n_symmetry_constrained_params = 1
        self.bound = None

    def set_bound(self, bound):
        if not isinstance(bound, V0rParamBound):
            raise ValueError("Bound must be of type V0rParamBound.")
        self.bound = bound

    def get_v0r_transformer(self):
        if self.bound is None:
            raise ValueError("Bound must be set before transformation.")
        # v0r is a relative value, so there is no offset
        offset = jnp.array([self.bound.min]).reshape(1, 1)
        weights = jnp.array(self.bound.max - self.bound.min).reshape(1, 1)
        return LinearTransformer(weights, offset, out_reshape=(1,))


class V0rParamBound(Bound):
    def __init__(self, min, max):
        super().__init__(min, max)
