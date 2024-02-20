"""Module interpolation

This module is a reworking of scipy's and my Bspline interpolation methods.
It can interpolate functions efficiently and in a JAX-compatible way."""

import numpy as np
import jax
from jax import numpy as jnp
from scipy import interpolate

def find_interval(knots, x_val, intpol_deg):
    """Return index of interval in knots that contains x_val"""
    # raise if knots are not sorted
    if not jnp.all(knots[:-1] <= knots[1:]):
        raise ValueError('knots must be sorted')
    return jnp.clip(jnp.searchsorted(knots, x_val, side='left'),
                    a_min=intpol_deg + 1,
                    a_max=knots.size - intpol_deg - 1) -1


def get_bspline_coeffs(lhs, rhs):
    """Return the coefficients of the B-spline interpolant.

    Solves the linear system lhs * coeffs = rhs for coeffs."""
    # TODO: we could do this more efficiently. One easy improvement would be to
    # pre-factorize lhs by splitting .solve() into .lu_factor() and .lu_solve()
    # parts. Only the solve part depends on the right hand side.
    return jnp.linalg.solve(lhs, rhs)



def not_a_knot_rhs(values):
    return values.reshape(-1, 1)


def calc_de_boor(knots, target_grid, deriv_order, intpol_deg):
    """Calculate the De Boor coefficients for the given knots and target grid"""
    de_boor_matrix = np.zeros((intpol_deg + 1, target_grid.size))
    intervals = find_interval(knots, target_grid, intpol_deg)
    for i, (interval, new_x) in enumerate(zip(intervals, target_grid)):
        beta_coeffs = interpolate._bspl.evaluate_all_bspl(knots,
                                                          intpol_deg,
                                                          new_x,
                                                          interval,
                                                          deriv_order)
        de_boor_matrix[:, i] = beta_coeffs
    return de_boor_matrix


def evaluate_spline(de_boor_coeffs, spline_coeffs, intervals, intpol_deg):
    """Evaluate the spline using the De Boor coefficients and the B-spline coefficients"""
    # Extract the relevant coefficients for each interval
    lower_indices = intervals - intpol_deg
    coeff_indices = lower_indices.reshape(-1,1) + np.arange(intpol_deg+1)
    coeff_subarrays = spline_coeffs[coeff_indices]
    coeff_subarrays = coeff_subarrays.reshape(-1, intpol_deg+1) # remove tailing 1 dimension

    # Element-wise multiplication between coefficients and de_boor values, sum over basis functions
    return np.einsum('ij,ji->i', coeff_subarrays, de_boor_coeffs)

## Set up left hand side (LHS)
def set_up_knots_and_lhs(original_grid, intpol_deg,
               boundary_condition='not-a-knot'):
    """Taken mostly from scipy interpolate.py, with some modifications"""
    if boundary_condition != 'not-a-knot':
        raise NotImplementedError("Only not-a-knot boundary conditions"
                                  "are currently supported")

    knots, colloc_matrix = _knots_and_colloc_matrix_not_a_knot(
        original_grid, intpol_deg)

    full_colloc_matrix = _banded_colloc_matrix_to_full(colloc_matrix,
                                                       intpol_deg)

    # we can now determine the coefficients by solving the linear system
    return knots, full_colloc_matrix




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
