"""Hashable Array wrapper for JAX.

Helps to make JAX arrays hashable, so they can be used static args in jit.
This speeds up the computation of functions that use these arrays as arguments.

Adapted from https://github.com/google/jax/issues/4572#issuecomment-709809897
"""


import jax.numpy as jnp


def simple_sum_hash(x):
    return hash(x.tobytes())


class HashableArray:
    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return simple_sum_hash(self.val)

    def __eq__(self, other):
        return isinstance(other, HashableArray) and jnp.all(
            jnp.equal(self.val, other.val)
        )
