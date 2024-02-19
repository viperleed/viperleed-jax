"""Module interpolation

This module is a reworking of scipy's and my Bspline interpolation methods.
It can interpolate functions efficiently and in a JAX-compatible way."""

import jax
from jax import numpy as jnp
from scipy import interpolate

def find_interval(knots, x_val):
    """Return index of interval in knots that contains x_val"""
    # raise if knots are not sorted
    if not jnp.all(knots[:-1] <= knots[1:]):
        raise ValueError('knots must be sorted')
    return jnp.searchsorted(knots, x_val, side='left') -1

