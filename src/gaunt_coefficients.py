from functools import partial

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
import jax
from pathlib import Path

from src.dense_quantum_numbers import MAXIMUM_LMAX

_REDUCED_GAUNT_COEFFICIENTS = jnp.load(Path(__file__).parent / "gaunt_coefficients.npy",
                                       allow_pickle=False)

@jit
def fetch_stored_gaunt_coeffs(l1: int, l2: int, l3: int,
                              m1: int, m2: int, m3: int) -> float:
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
    selection_rule_m = m1 + m2 + m3 == 0
    return jax.lax.cond(
        selection_rule_m,
        (lambda l1, l2, l3, m1, m2: 
         _REDUCED_GAUNT_COEFFICIENTS[l1, l2, l3, m1, m2]),
        lambda l1, l2, l3, m1, m2: 0.0,
        l1, l2, l3, m1, m2,
    )

# vectorize cppp in all arguments
# Clebsh-Gordon coefficients for computation of temperature-dependent phase shifts
# used in tscatf; can be jitted
# TODO: @ Paul: chose a better name for this function
@partial(vmap, in_axes=(0, None, None))
@partial(vmap, in_axes=(None, 0, None))
@partial(vmap, in_axes=(None, None, 0))
def cppp(l1: int, l2: int , l3: int) -> float:
    """Calculates integral over three associated Legendre polynomials.

    Parameters
    ----------
    l1, l2, l3 : int

    Returns
    -------
    float
        Value of the integral.

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
    pre_factor = 1/jnp.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1))
    pre_factor *= jnp.sqrt(4*jnp.pi)
    return pre_factor * _REDUCED_GAUNT_COEFFICIENTS[l1, l2, l3, 0, 0]


PRE_CALCULATED_CPPP = {
    l: cppp(jnp.arange(0, 2*l+1), jnp.arange(0, l+1), jnp.arange(0, l+1))
    for l in range(MAXIMUM_LMAX+1)
}
