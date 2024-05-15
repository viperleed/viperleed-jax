"""Module R-factor"""
from functools import partial

import jax
from jax import numpy as jnp

from src import interpolation


def pendry_R(intensity_2,
             interpolator_1, interpolator_2,
             v0_real_steps, v0_imag, energy_step,
             intensity_1):
    """Calculate the R-factor for two beams"""

    # calculate the interpolated intensities and derivatives
    b_spline_coeffs_1 = interpolation.get_bspline_coeffs(
        interpolator_1,
        interpolation.not_a_knot_rhs(intensity_1))
    intens_1 = interpolation.evaluate_spline(
        b_spline_coeffs_1,
        interpolator_1,
        0
    )
    deriv_1 = interpolation.evaluate_spline(
        b_spline_coeffs_1,
        interpolator_1,
        1
    )

    b_spline_coeffs_2 = interpolation.get_bspline_coeffs(
        interpolator_2,
        interpolation.not_a_knot_rhs(intensity_2))
    intens_2 = interpolation.evaluate_spline(
        b_spline_coeffs_2,
        interpolator_2,
        0
    )
    deriv_2 = interpolation.evaluate_spline(
        b_spline_coeffs_2,
        interpolator_2,
        1
    )

    y_1 = pendry_y(intens_1, deriv_1, v0_imag)
    y_2 = pendry_y(intens_2, deriv_2, v0_imag)

    # shift y_1 by v0_real_steps
    y_1 = shift_v0r(y_1, v0_real_steps)

    return pendry_R_from_y(y_1, y_2, energy_step)


def pendry_R_vs_reference(reference_intensity, reference_interpolator,
                          sampling_interpolator, v0_real, v0_imag, energy_step):
    """Return an R-factor function that compares a given intensity to a reference"""
    if not (reference_interpolator.is_compatible(sampling_interpolator)):
        raise ValueError("Interpolators are not compatible")
    jitted_rfactor = jax.jit(pendry_R, static_argnames=(
        'interpolator_2',
        'interpolator_1',
        'v0_real',
        'v0_imag',
        'energy_step',
    ))
    return partial(jitted_rfactor, reference_intensity, reference_interpolator,
                   sampling_interpolator, v0_real, v0_imag, energy_step)

def pendry_R_from_intensity_and_derivative(intens_deriv_1, intens_deriv_2,
                                      v0_real_steps, v0_imag, energy_step):
    intens_1, deriv_1 = intens_deriv_1
    intens_2, deriv_2 = intens_deriv_2

    y_1 = pendry_y(intens_1, deriv_1, v0_imag)
    y_2 = pendry_y(intens_2, deriv_2, v0_imag)

    # shift y_1 by v0_real_steps
    y_1 = shift_v0r(y_1, v0_real_steps)

    return pendry_R_from_y(y_1, y_2, energy_step)


def pendry_R_from_y(y_1, y_2, energy_step):
    #TODO: figure out how to implement v0_real

    # TODO?: potentially, one could do these integrals analytically based on the spline coefficients
    numerators = nansum_trapezoid((y_1 - y_2)**2, dx=energy_step, axis=0)
    denominators = nansum_trapezoid((y_1**2 + y_2**2), dx=energy_step, axis=0)
    # R factor for all beams
    return jnp.sum(numerators) / jnp.sum(denominators)


def pendry_y(intensity, intensity_derivative, v0_imag):
    intens_deriv_ratio = intensity / intensity_derivative
    return intens_deriv_ratio / (intens_deriv_ratio**2 + v0_imag**2)


def nansum_trapezoid(y, dx, axis=-1):
    y_arr = jnp.moveaxis(y, axis, -1)
    # select the axis to integrate over
    return jnp.nansum(y_arr[..., 1:] + y_arr[..., :-1], axis=-1) * dx * 0.5


def shift_v0r(array, n_steps):
    """Applies a v0r shift to the array by shifting the values n_steps up or
    down the first axis (energy) and padding with NaNs."""
    # NB, TODO: This only allows for integer shifts (multiples of the set
    # energy step). This is a limitation of the current implementation.
    # In principle, we could implement a more general shift and allow real
    # numbers by doing this earlier and changing the knot values in the
    # interpolator.
    n_energies, n_beams = array.shape[0], array.shape[1]
    
    rolled_array = jnp.roll(array, n_steps, axis=0)
    row_ids = jnp.arange(n_energies).reshape(-1, 1)
    row_ids_tiled = jnp.tile(row_ids, (1, n_beams))
    mask = jnp.logical_or(row_ids_tiled < n_steps,
                          row_ids >= n_energies+n_steps)
    return jnp.where(mask, jnp.nan, rolled_array)
