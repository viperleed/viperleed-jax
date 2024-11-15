"""Module gaunt_coefficients."""

__authors__ = ('Alexander M. Imre (@amimre)', 'Paul Haidegger (@Paulhai7)')
__created__ = '2024-01-03'

from functools import partial

import numpy as np
from jax import config

config.update('jax_enable_x64', True)
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import jit, vmap

from viperleed_jax.dense_quantum_numbers import (
    DENSE_QUANTUM_NUMBERS,
    MAXIMUM_LMAX,
)

# load precalculated coefficients
_REDUCED_GAUNT_COEFFICIENTS = jnp.load(
    Path(__file__).parent / 'gaunt_coefficients.npy', allow_pickle=False
)

_DENSE_L_2D = DENSE_QUANTUM_NUMBERS[MAXIMUM_LMAX][:, :, 0]
_DENSE_LP_2D = DENSE_QUANTUM_NUMBERS[MAXIMUM_LMAX][:, :, 1]
_DENSE_M_2D = DENSE_QUANTUM_NUMBERS[MAXIMUM_LMAX][:, :, 2]
_DENSE_MP_2D = DENSE_QUANTUM_NUMBERS[MAXIMUM_LMAX][:, :, 3]


# TODO: @Paul add docstring with references to the storage scheme paper
def find_index(
    l1: int, l2: int, l3: int, m1: int, m2: int, m3: int
) -> jnp.array:
    selection_rule_m = (
        (m1 + m2 + m3 == 0)
        & (abs(m3) <= l3)
        & (l3 <= l1 + l2)
        & (l3 >= abs(l1 - l2))
    )
    l_unsorted = jnp.array([l1, l2, l3])
    m_unsorted = jnp.array([m1, m2, m3])
    sorted_indices = jnp.argsort(l_unsorted)[::-1]
    l = l_unsorted[sorted_indices]
    m = m_unsorted[sorted_indices]
    index = (
        l[0] * (6 + l[0] * (11 + l[0] * (6 + l[0]))) / 24
        + l[1] * (2 + l[1] * (3 + l[1])) / 6
        + l[2] * (l[2] + 1) / 2
        + abs(m[2])
        + 1
    )
    index = index.astype(jnp.int64)
    my_m = jax.lax.cond(m[2] >= 0, (lambda m_1: m_1), lambda m_1: -m_1, m[1])
    index2 = (jnp.size(_REDUCED_GAUNT_COEFFICIENTS[index - 1]) - 1) / 2 + my_m
    index2 = index2.astype(jnp.int64)
    return jax.lax.cond(
        selection_rule_m,
        (lambda index, index2: jnp.array([index, index2])),
        lambda index, index2: jnp.array([0, 0]),
        index,
        index2,
    )


def fetch_gaunt(index1: int, index2: int) -> float:
    """
    Fetches the Gaunt coefficient from the loaded file.

    Parameters:
        index1 (int): The first index of the Gaunt coefficient.
        index2 (int): The second index of the Gaunt coefficient.

    Returns:
        float: The value of the Gaunt coefficient.

    """
    return _REDUCED_GAUNT_COEFFICIENTS[index1, index2]


@jit
def fetch_stored_gaunt_coeffs(
    l1: int, l2: int, l3: int, m1: int, m2: int, m3: int
) -> float:
    """Returns stored Gaunt coefficients.

    Enforces the selection rule m1 + m2 + m3 == 0.

    Parameters
    ----------
    l1, l2, l3 : int
        Angular momentum quantum numbers.
    m1, m2, m3 : int
        Magnetic quantum numbers.

    Returns
    -------
    float
        Value of the Gaunt coefficient.

    Notes
    _____
    The Gaunt coefficients are directly related to the Clebsch-Gordan
    coefficients and the Wigner 3-j symbols. The Gaunt coefficients are
    defined as

    .. math::

        \mathrm{Gaunt}(l_1, l_2, l_3, m_1, m_2, m_3) = \int{Y_{l_1, m_1} Y_{l_2, m_2} Y_{l_3, m_3}}
        = \sqrt{\frac{2l_1+1}{4\pi}} \sqrt{\frac{2l_2+1}{4\pi}} \sqrt{\frac{2l_3+1}{4\pi}} \mathrm{Wigner-3j}(l_1, l_2, l_3, m_1, m_2, m_3) \mathrm{Wigner-3j}(l_1, l_2, l_3, 0, 0, 0)

    where :math:`Y_{l, m}` are the spherical harmonics.

    They satisfy the selection rule :math:`m_1 + m_2 + m_3 = 0` and
    :math:`|l_1-l_2| \le l_3 \le l_1+l_2`.
    """
    reduced_indices = find_index(l1, l2, l3, m1, m2, m3)
    return fetch_gaunt(reduced_indices[0], reduced_indices[1])


# vectorize integrate_legendre in all arguments
# Clebsh-Gordon coefficients for computation of temperature-dependent phase shifts
# used in tscatf; can be jitted
@partial(vmap, in_axes=(0, None, None))
@partial(vmap, in_axes=(None, 0, None))
@partial(vmap, in_axes=(None, None, 0))
def integrate_legendre(l1: int, l2: int, l3: int) -> float:
    """Calculates integral over three associated Legendre polynomials.

    Parameters
    ----------
    l1, l2, l3 : int

    Returns
    -------
    float
        Value of the integral.b

    Notes
    -----
    This function returns the value of the integral of three associated
    Legendre polynomials :math:`\int{P_{l1}^{0}P_{l1}^{0}P_{l1}^{0}}`.
    It performs the function of TensErLEED's CPPP subroutine by Pendry.
    Instead of calculating the integral from scratch, we calculate it from
    the stored precomputed Gaunt coefficients using the following relation:
    (see also https://en.wikipedia.org/wiki/3-j_symbol and
    https://docs.sympy.org/latest/modules/physics/wigner.html):

    .. math::

        \mathrm{Gaunt}(l_1, l_2, l_3, m_1, m_2, m_3) = \int{Y_{l_1, m_1} Y_{l_2, m_2} Y_{l_3, m_3}}
        Y_{l, 0} = \sqrt{\frac{2l+1}{4\pi}} P_l^0
    """
    pre_factor = 1 / jnp.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1))
    pre_factor *= jnp.sqrt(4 * jnp.pi)
    reduced_indices = find_index(l1, l2, l3, 0, 0, 0)
    return pre_factor * fetch_gaunt(reduced_indices[0], reduced_indices[1])


_all_pre_calculated_cppp = integrate_legendre(
    np.arange(0, 2 * MAXIMUM_LMAX + 1),
    np.arange(0, MAXIMUM_LMAX + 1),
    np.arange(0, MAXIMUM_LMAX + 1),
)

PRE_CALCULATED_CPPP = {
    l: _all_pre_calculated_cppp[: 2 * l + 1, : l + 1, : l + 1]
    for l in range(MAXIMUM_LMAX + 1)
}

lpp_indices = jax.vmap(
    jax.vmap(find_index, in_axes=(0, 0, None, 0, 0, 0), out_axes=0),
    in_axes=(0, 0, None, 0, 0, 0),
    out_axes=0,
)

lpp_gaunt = jax.vmap(
    jax.vmap(fetch_gaunt, in_axes=(0, 0), out_axes=0),
    in_axes=(0, 0),
    out_axes=0,
)

gaunt_array = jnp.array(
    [
        lpp_indices(
            _DENSE_L_2D,
            _DENSE_LP_2D,
            lpp,
            _DENSE_M_2D,
            -_DENSE_MP_2D,
            -_DENSE_M_2D + _DENSE_MP_2D,
        )
        for lpp in range(MAXIMUM_LMAX * 2 + 1)
    ]
)

CSUM_COEFFS = jnp.array(
    [
        lpp_gaunt(gaunt_array[lpp, :, :, 0], gaunt_array[lpp, :, :, 1])
        * (-1.0) ** (_DENSE_M_2D)
        * 1j
        ** (
            _DENSE_L_2D + _DENSE_LP_2D - lpp
        )  # AI: I found we need this factor, but I still don't understand where it comes from
        for lpp in range(MAXIMUM_LMAX * 2 + 1)
    ]
)
