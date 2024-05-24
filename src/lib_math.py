from functools import lru_cache
import numpy as np
from jax.scipy.special import sph_harm
import jax
import jax.numpy as jnp
from functools import partial

from src.dense_quantum_numbers import DENSE_M, DENSE_L
from src.dense_quantum_numbers import MAXIMUM_LMAX

# Spherical Bessel functions from NeuralIL
from src.spherical_bessel import functions

# numerical epsilon to avoid division by zero
EPS = 1e-8
bessel_EPS = 10e-7


def _divide_zero_safe(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
    limit_value: float = 0.0,
) -> jnp.ndarray:
    """Function that forces the result of dividing by 0 to be equal to a limit
    value in a jit- and autodiff-compatible way

    Args:
        numerator: Values in the numerator
        denominator: Values in the denominator, may contain zeros
        limit_value: Value to return where denominator == 0.0
    Returns:
        numerator / denominator with result == 0.0 where denominator == 0.0
    """
    denominator_masked = jnp.where(denominator == 0.0, 1.0, denominator)
    return jnp.where(
        denominator == 0.0,
        limit_value,
        numerator / denominator_masked,
    )

@jax.named_scope("safe_norm")
def safe_norm(vector: jnp.ndarray) -> jnp.ndarray:
    """Safe norm calculation to avoid NaNs in gradients"""
    # avoids nan in gradient for jnp.linalg.norm(C)
    return jnp.sqrt(jnp.sum(vector**2) + EPS**2)


def _generate_bessel_functions(l_max):
    """Generate a list of spherical Bessel functions up to order l_max"""
    bessel_functions = []
    for order in range(l_max+1):
        bessel_functions.append(functions.create_j_l(order))
    return bessel_functions


# generate a list of spherical Bessel functions up to order l_max
BESSEL_FUNCTIONS = _generate_bessel_functions(MAXIMUM_LMAX)


@jax.named_scope("masked_bessel")
def masked_bessel(z, n1):
    z_is_small = abs(z) < bessel_EPS
    _z = jnp.where(z_is_small, bessel_EPS, z)
    return jnp.nan_to_num(
        # Needs a limit of >=10e-7 to avoid numerical noise around z=0 for
        # small values of z with imaginary component
        jnp.where(z_is_small,
                  jnp.zeros(shape=(n1), dtype=jnp.complex128).at[0].set(1.0),
                  bessel(_z, n1))
    )

@jax.named_scope("custom_spherical_jn")
def custom_spherical_jn(n, z):
    return jax.lax.switch(n, BESSEL_FUNCTIONS, z)


# Bessel functions from NeuralIL
@jax.named_scope("bessel")
def bessel(z, n1):
    """Spherical Bessel functions. Evaluated at z, up to degree n1."""
    vmapped_custom_bessel = jax.vmap(custom_spherical_jn, (0, None))
    return vmapped_custom_bessel(jnp.arange(n1), z)


@jax.named_scope("HARMONY")
def HARMONY(C, LMAX):
    """Generates the spherical harmonics for the vector C.

    This is a python implementation of the fortran subroutine HARMONY from
    TensErLEED. It uses the jax.scipy.special.sph_harm function to produce
    equivalent results."""
    _, theta, phi = cart_to_polar(C)
    return sph_harm(DENSE_M[2*LMAX], DENSE_L[2*LMAX], jnp.asarray([phi]), jnp.asarray([theta]), n_max=2*LMAX)

@jax.named_scope("cart_to_polar")
def cart_to_polar(c):
    """Converts cartesian coordinates to polar coordinates."""
    z, x, y = c
    x_y_norm = jnp.sqrt(x**2 + y**2)
    r = jnp.linalg.norm(c)
    theta = jnp.arctan2(jnp.where(x_y_norm > EPS, x_y_norm, EPS),
                        jnp.where(z**2 > EPS**2, z, EPS))
    phi = jnp.arctan2(jnp.where(x_y_norm > EPS, y, EPS),
                      jnp.where(x_y_norm > EPS, x, EPS))
    return r, theta, phi
