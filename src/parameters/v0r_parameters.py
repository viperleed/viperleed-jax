from jax import numpy as jnp

from src.base import LinearTransformer
from src.parameters.base_parameters import Bound


class V0rParam():
    
    # Note: currently, V0r is (and can only be) a single parameter, which makes
    # it much simpler than the other parameters. Further constraints are not
    # implemented since there does not seem to be any use for them.
    # In future, this could be extended to try and vary other parameters
    # (e.g. incidence angle, etc.).
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
        offset = jnp.array([0.0]).reshape(1, 1)
        weights = jnp.array(self.bound.max - self.bound.min).reshape(1, 1)
        return LinearTransformer(weights, offset)


class V0rParamBound(Bound):
    def __init__(self, min, max):
        super().__init__(min, max)
