"""Module interpolation

This module is a reworking of scipy's and my Bspline interpolation methods.
It can interpolate functions efficiently and in a JAX-compatible way."""

import numpy as np
import jax
from jax import numpy as jnp
from scipy import interpolate

def find_interval(knots, x_val):
    """Return index of interval in knots that contains x_val"""
    # raise if knots are not sorted
    if not jnp.all(knots[:-1] <= knots[1:]):
        raise ValueError('knots must be sorted')
    return jnp.searchsorted(knots, x_val, side='left') -1


## Set up left hand side (LHS)
def set_up_lhs(original_grid, target_grid, intpol_deg,
               boundary_condition='not-a-knot'):
    """Taken mostly from scipy interpolate.py, with some modifications"""
    if boundary_condition != 'not-a-knot':
        raise NotImplementedError("Only not-a-knot boundary conditions"
                                  "are currently supported")

    knots, colloc_matrix = _knots_and_colloc_matrix_not_a_knot(
        original_grid, intpol_deg)

    full_colloc_matrix = _banded_colloc_matrix_to_full(colloc_matrix,
                                                       intpol_deg)




def _knots_and_colloc_matrix_not_a_knot(original_grid, intpol_deg):
    """Return the knots and collocation matrix for a given grid and degree"""
    # get the knots (not-a-knot bc)
    knots = interpolate._bsplines._not_a_knot(original_grid, intpol_deg)

    # number of derivatives at boundaries (not-a-knot bc)
    deriv_l = None
    deriv_l_ords, deriv_l_vals = interpolate._bsplines._process_deriv_spec(deriv_l)
    deriv_r = None
    deriv_r_ords, deriv_r_vals = interpolate._bsplines._process_deriv_spec(deriv_r)

    nleft = deriv_l_ords.shape[0]
    nright = deriv_r_ords.shape[0]

    # basic collocation matrix
    kl = ku = intpol_deg
    nt = knots.size - intpol_deg - 1
    banded_colloc_matrix = np.zeros((2*kl + ku + 1, nt), dtype=np.float64, order='F')
    interpolate._bspl._colloc(np.array(original_grid, dtype=np.float64),
                np.array(knots, dtype=np.float64),
                intpol_deg, banded_colloc_matrix)

    # derivatives at boundaries
    if nleft > 0:
        interpolate._bspl._handle_lhs_derivatives(
            knots,
            intpol_deg,
            original_grid[0], 
            banded_colloc_matrix, kl, ku,
            deriv_l_ords.astype(np.dtype("long")))
    if nright > 0:
        interpolate._bspl._handle_lhs_derivatives(
            knots,
            intpol_deg,
            original_grid[-1],
            banded_colloc_matrix, kl, ku,
            deriv_r_ords.astype(np.dtype("long")),
            offset=nt-nright)
    return knots, banded_colloc_matrix

## Deal with collocation matrix

def _banded_colloc_matrix_to_full(colloc_matrix, intpol_deg):
    kl = intpol_deg
    center_row = 2*kl
    full_matrix = np.diag(colloc_matrix[center_row,:])
    for k in range(1, kl+1):
        full_matrix += np.diag(colloc_matrix[center_row-k,k:], k)
        full_matrix += np.diag(colloc_matrix[center_row+k,:-k], -k)
    return full_matrix


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
