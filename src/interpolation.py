"""Module interpolation

This module is a reworking of scipy's and my Bspline interpolation methods.
It can interpolate functions efficiently and in a JAX-compatible way."""
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from scipy import interpolate


class StaticGridSplineInterpolator(ABC):
    def __init__(self, origin_grid, target_grid, intpol_deg):
        self.origin_grid = origin_grid
        self.target_grid = target_grid
        self.intpol_deg = intpol_deg
        self.knots = self._get_knots()
        self.intervals = self._find_intervals()
        # TODO: If we find a suitable solver, we could also use the banded
        # and pre-factorized collocation matrix

        # calculate De Boor coefficients
        self.de_boor_coeffs = {
            deriv_order: self._calc_de_boor_coeffs(deriv_order)
            for deriv_order in range(intpol_deg)
        }

        # calcualte collocation matrix
        self.full_colloc_matrix = self.calculate_colloc_matrix()

        # Invert collocation matrix – this reduces the B-spline interpolation
        # to a simple matrix- vector multiplication
        self.inv_colloc_matrix = jnp.linalg.inv(self.full_colloc_matrix)

    def is_compatible(self, other):
        if not isinstance(other, type(self)):
            return False
        return (jnp.all(self.target_grid == other.target_grid) and
                self.intpol_deg == other.intpol_deg)

    @abstractmethod
    def _get_knots(self):
        raise NotImplementedError

    def _calc_de_boor_coeffs(self, deriv_order):
        """Calculate the De Boor coeffs for the given knots and target grid"""
        de_boor_coeffs = np.zeros((self.intpol_deg + 1, self.target_grid.size))
        for i, (interval, new_x) in enumerate(
            zip(self.intervals, self.target_grid)):
            beta_coeffs = interpolate._bspl.evaluate_all_bspl(
                self.knots,
                self.intpol_deg,
                new_x,
                interval,
                deriv_order)
            de_boor_coeffs[:, i] = beta_coeffs
        return de_boor_coeffs

    def _find_intervals(self):
        """Return index of interval in knots that contains x_val"""
        # raise if knots are not sorted
        if not jnp.all(self.knots[:-1] <= self.knots[1:]):
            raise ValueError('knots must be sorted')
        intervals = jnp.clip(jnp.searchsorted(self.knots,
                                              self.target_grid,
                                              side='left'),
                        a_min=self.intpol_deg + 1,
                        a_max=self.knots.size - self.intpol_deg - 1) - 1
        return intervals

    def _banded_colloc_matrix_to_full(self, banded_colloc_matrix):
        kl = self.intpol_deg
        center_row = 2*kl
        full_matrix = np.diag(banded_colloc_matrix[center_row,:])
        for k in range(1, kl+1):
            full_matrix += np.diag(banded_colloc_matrix[center_row-k,k:], k)
            full_matrix += np.diag(banded_colloc_matrix[center_row+k,:-k], -k)
        return full_matrix


class StaticNotAKnotSplineInterpolator(StaticGridSplineInterpolator):
    def _get_knots(self):
        return interpolate._bsplines._not_a_knot(self.origin_grid, self.intpol_deg)

    def calculate_colloc_matrix(self):
        # number of derivatives at boundaries (not-a-knot bc)
        deriv_l = None
        deriv_l_ords, deriv_l_vals = interpolate._bsplines._process_deriv_spec(deriv_l)
        deriv_r = None
        deriv_r_ords, deriv_r_vals = interpolate._bsplines._process_deriv_spec(deriv_r)

        nleft = deriv_l_ords.shape[0]
        nright = deriv_r_ords.shape[0]

        # basic collocation matrix
        kl = ku = self.intpol_deg
        nt = self.knots.size - self.intpol_deg - 1
        banded_colloc_matrix = np.zeros((2*kl + ku + 1, nt), dtype=np.float64, order='F')
        interpolate._bspl._colloc(np.array(self.origin_grid, dtype=np.float64),
                    np.array(self.knots, dtype=np.float64),
                    self.intpol_deg, banded_colloc_matrix)

        # derivatives at boundaries
        if nleft > 0:
            interpolate._bspl._handle_lhs_derivatives(
                self.knots,
                self.intpol_deg,
                self.origin_grid[0], 
                banded_colloc_matrix, kl, ku,
                deriv_l_ords.astype(np.dtype("long")))
        if nright > 0:
            interpolate._bspl._handle_lhs_derivatives(
                self.knots,
                self.intpol_deg,
                self.origin_grid[-1],
                banded_colloc_matrix, kl, ku,
                deriv_r_ords.astype(np.dtype("long")),
                offset=nt-nright)

        # get full collocation matrix
        return self._banded_colloc_matrix_to_full(banded_colloc_matrix)

# TODO: The below functions could (and probably should be) interpolator class
#       methods. However, we need to figure out how to make this work with JAX.

@partial(jax.jit, static_argnames=('interpolator',))
def get_bspline_coeffs(interpolator, rhs):
    """Return the coefficients of the B-spline interpolant.

    Solves the linear system lhs * coeffs = rhs for coeffs."""
    # TODO: we could do this more efficiently. One easy improvement would be to
    # pre-factorize lhs by splitting .solve() into .lu_factor() and .lu_solve()
    # parts. Only the solve part depends on the right hand side.
    spline_coeffs = interpolator.inv_colloc_matrix @ rhs
    return spline_coeffs

@jax.jit
def not_a_knot_rhs(values):
    values = jnp.asarray(values)
    return values


@partial(jax.jit, static_argnames=('interpolator', 'deriv_order'))
def evaluate_spline(spline_coeffs, interpolator, deriv_order=0):
    """Evaluate the spline using the De Boor coefficients and the B-spline coefficients"""
    # Extract the relevant coefficients for each interval
    lower_indices = interpolator.intervals - interpolator.intpol_deg
    coeff_indices = lower_indices.reshape(-1,1) + jnp.arange(interpolator.intpol_deg+1)
    coeff_subarrays = spline_coeffs[coeff_indices]
    coeff_subarrays = coeff_subarrays.reshape(-1, interpolator.intpol_deg+1) # remove tailing 1 dimension

    # Element-wise multiplication between coefficients and de_boor values, sum over basis functions
    return jnp.einsum('ij,ji->i',
                      coeff_subarrays,
                      interpolator.de_boor_coeffs[deriv_order])


# TODO: implement natural knot interpolator
## Natural knots spline boundary condition – currently unused – 
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
