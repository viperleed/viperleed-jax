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

## Natural knots boundary condition – currently unused – TODO: implement?
def get_natural_knots(x, deg):
    knots = np.concatenate(
        [np.full(shape=(deg,), fill_value=x[0]),
         x,
         np.full(shape=(deg,), fill_value=x[-1])])
    return knots

def natural_derivative_bc(deg):
    _supported_degrees = (3, 5)

    if deg == 3:
        derivs_l_ord = np.array([2])
        derivs_l_val = np.array([0])
        
        derivs_r_ord = np.array([2])
        derivs_r_val = np.array([0])

    elif deg == 5:
        derivs_l_ord = np.array([3, 4])
        derivs_l_val = np.array([0, 0])
        
        derivs_r_ord = np.array([3, 4])
        derivs_r_val = np.array([0, 0])

    else:
        raise ValueError(f"unsupported degree {deg}")
    return derivs_l_ord, derivs_l_val, derivs_r_ord, derivs_r_val
