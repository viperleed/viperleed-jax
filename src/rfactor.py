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
    y_1 = integer_shift_v0r(y_1, v0_real_steps)

    return pendry_R_from_y(y_1, y_2, energy_step)


def pendry_R_from_intensity_and_derivative(intens_deriv_1, intens_deriv_2,
                                      v0_real_steps, v0_imag, energy_step):
    intens_1, deriv_1 = intens_deriv_1
    intens_2, deriv_2 = intens_deriv_2

    y_1 = pendry_y(intens_1, deriv_1, v0_imag)
    y_2 = pendry_y(intens_2, deriv_2, v0_imag)

    # shift y_1 by v0_real_steps
    y_1 = integer_shift_v0r(y_1, v0_real_steps)

    return pendry_R_from_y(y_1, y_2, energy_step)


def pendry_R_from_y(y_1, y_2, energy_step):

    # mask out NaNs for this calculation
    y_1_mask = jnp.isnan(y_1)
    y_2_mask = jnp.isnan(y_2)
    mask = jnp.logical_or(y_1_mask, y_2_mask)

    y_1 = jnp.where(mask, 0, y_1)
    y_2 = jnp.where(mask, 0, y_2)

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


def integer_shift_v0r(array, n_steps):
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

### R2 ###

def R_2(intensity_2,
        interpolator_1, interpolator_2,
        v0_real_steps, v0_imag, energy_step,
        intensity_1):
    
    # calculate interpolation – no derivatives needed for R2

    b_spline_coeffs_1 = interpolation.get_bspline_coeffs(
        interpolator_1,
        interpolation.not_a_knot_rhs(intensity_1))
    intens_1 = interpolation.evaluate_spline(
        b_spline_coeffs_1,
        interpolator_1,
        0
    )

    b_spline_coeffs_2 = interpolation.get_bspline_coeffs(
        interpolator_2,
        interpolation.not_a_knot_rhs(intensity_2))
    intens_2 = interpolation.evaluate_spline(
        b_spline_coeffs_2,
        interpolator_2,
        0
    )

    # shift intens_1 by v0_real_steps
    intens_1 = integer_shift_v0r(intens_1, v0_real_steps)

    # calculate normalization for each beam
    beam_normalization = (nansum_trapezoid(intens_1, energy_step, axis=0)
              / nansum_trapezoid(intens_2, energy_step, axis=0))

    numerators = nansum_trapezoid((intens_1 - beam_normalization*intens_2)**2,
                                  energy_step, axis=0)
    denominators = nansum_trapezoid(intens_1**2, energy_step, axis=0)
    return jnp.sum(numerators) / jnp.sum(denominators)


def R_1(intensity_2,
        interpolator_1, interpolator_2,
        v0_real_steps, v0_imag, energy_step,
        intensity_1):

    # calculate interpolation – no derivatives needed for R2

    b_spline_coeffs_1 = interpolation.get_bspline_coeffs(
        interpolator_1,
        interpolation.not_a_knot_rhs(intensity_1))
    intens_1 = interpolation.evaluate_spline(
        b_spline_coeffs_1,
        interpolator_1,
        0
    )

    b_spline_coeffs_2 = interpolation.get_bspline_coeffs(
        interpolator_2,
        interpolation.not_a_knot_rhs(intensity_2))
    intens_2 = interpolation.evaluate_spline(
        b_spline_coeffs_2,
        interpolator_2,
        0
    )

    # shift intens_1 by v0_real_steps
    intens_1 = integer_shift_v0r(intens_1, v0_real_steps)

    # calculate normalization for each beam
    beam_normalization = (nansum_trapezoid(intens_1, energy_step, axis=0)
              / nansum_trapezoid(intens_2, energy_step, axis=0))

    numerators = nansum_trapezoid(abs((intens_1 - beam_normalization*intens_2)),
                                  energy_step, axis=0)
    denominators = nansum_trapezoid(intens_1, energy_step, axis=0)
    return jnp.sum(numerators) / jnp.sum(denominators)

### RMS ###

def R_ms(intensity_2,
        interpolator_1, interpolator_2,
        v0_real_steps, v0_imag, energy_step,
        intensity_1):


    b_spline_coeffs_1 = interpolation.get_bspline_coeffs(
        interpolator_1,
        interpolation.not_a_knot_rhs(intensity_1))
    intens_1 = interpolation.evaluate_spline(
        b_spline_coeffs_1,
        interpolator_1,
        0
    )
    first_deriv_1 = interpolation.evaluate_spline(
        b_spline_coeffs_1,
        interpolator_1,
        1
    )
    second_deriv_1 = interpolation.evaluate_spline(
        b_spline_coeffs_1,
        interpolator_1,
        2
    )

    b_spline_coeffs_2 = interpolation.get_bspline_coeffs(
        interpolator_2,
        interpolation.not_a_knot_rhs(intensity_2))
    intens_2 = interpolation.evaluate_spline(
        b_spline_coeffs_2,
        interpolator_2,
        0
    )
    first_deriv_2 = interpolation.evaluate_spline(
        b_spline_coeffs_2,
        interpolator_2,
        1
    )
    second_deriv_2 = interpolation.evaluate_spline(
        b_spline_coeffs_1,
        interpolator_1,
        2
    )

    y_1 = y_ms(intens_1, first_deriv_1, second_deriv_1, v0_imag, energy_step)
    y_2 = y_ms(intens_2, first_deriv_2, second_deriv_2, v0_imag, energy_step)

    # shift intens_1 by v0_real_steps
    y_1 = integer_shift_v0r(y_1, v0_real_steps)

    return pendry_R_from_y(y_1, y_2, energy_step)


def y_ms(intensity, first_derivative, second_derivative, v0_imag, e_step):
    numerator = first_derivative
    condition = second_derivative > 0
    denominator = intensity**2 + 2*(first_derivative*0.5*v0_imag/e_step)**2
    denominator += condition*(second_derivative*jnp.sqrt(0.1)*v0_imag/e_step)**2
    denominator = jnp.sqrt(denominator)
    return numerator / denominator

def R_zj(intensity_calc,
        interpolator_exp, interpolator_calc,
        v0_real_steps, v0_imag, energy_step,
        intensity_exp):

    # calculate interpolation – no derivatives needed for R2

    b_spline_coeffs_exp = interpolation.get_bspline_coeffs(
        interpolator_exp,
        interpolation.not_a_knot_rhs(intensity_exp))
    intens_exp = interpolation.evaluate_spline(
        b_spline_coeffs_exp,
        interpolator_exp,
        0
    )
    first_deriv_exp = interpolation.evaluate_spline(
        b_spline_coeffs_exp,
        interpolator_exp,
        1
    )
    second_deriv_exp = interpolation.evaluate_spline(
        b_spline_coeffs_exp,
        interpolator_exp,
        2
    )

    b_spline_coeffs_calc = interpolation.get_bspline_coeffs(
        interpolator_calc,
        interpolation.not_a_knot_rhs(intensity_calc))
    intens_calc = interpolation.evaluate_spline(
        b_spline_coeffs_calc,
        interpolator_calc,
        0
    )
    first_deriv_calc = interpolation.evaluate_spline(
        b_spline_coeffs_calc,
        interpolator_calc,
        1
    )
    second_deriv_calc = interpolation.evaluate_spline(
        b_spline_coeffs_calc,
        interpolator_calc,
        2
    )

    exp_energy_ranges = jnp.logical_not(jnp.isnan(intens_exp)).sum(axis=0) * energy_step

    # Factor 0.027 for random correlation, Zannazi & Jona 1977
    prefactors = 1/nansum_trapezoid(intens_exp, energy_step, axis=0) /0.027

    # # calculate normalization for each beam
    beam_normalization = (nansum_trapezoid(intens_exp, dx=energy_step, axis=0)
              / nansum_trapezoid(intens_calc, dx=energy_step, axis=0))

    numerators = (abs(beam_normalization*second_deriv_calc-second_deriv_exp)*
                  abs(beam_normalization*first_deriv_calc-first_deriv_exp))
    denominators = abs(first_deriv_exp) + jnp.nanmax(first_deriv_exp, axis=0)

    r_beams = prefactors*nansum_trapezoid(numerators/denominators, axis=0, dx=energy_step)

    return jnp.nansum(r_beams*exp_energy_ranges) / jnp.nansum(exp_energy_ranges)
