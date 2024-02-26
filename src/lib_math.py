from functools import lru_cache
import numpy as np
from jax.scipy.special import sph_harm
import jax
import jax.numpy as jnp
from functools import partial

from src.dense_quantum_numbers import DENSE_M, DENSE_L

# Spherical Bessel functions from NeuralIL
from src.spherical_bessel import functions

# numerical epsilon to avoid division by zero
EPS = 1e-8


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


def safe_norm(vector: jnp.ndarray) -> jnp.ndarray:
    """Safe norm calculation to avoid NaNs in gradients"""
    # avoids nan in gradient for jnp.linalg.norm(C)
    return jnp.sqrt(jnp.sum(vector**2) + EPS)


def _generate_bessel_functions(l_max):
    """Generate a list of spherical Bessel functions up to order l_max"""
    bessel_functions = []
    for order in range(l_max+1):
        bessel_functions.append(functions.create_j_l(order))
    return bessel_functions

# LMAX here for testing; TODO: remove
LMAX = 14

# generate a list of spherical Bessel functions up to order l_max
BESSEL_FUNCTIONS = _generate_bessel_functions(LMAX)


def masked_bessel(z, n1):
    z_is_small = abs(z) < 10e-7
    _z = jnp.where(z_is_small, 10e-7, z)
    return jnp.nan_to_num(
        # Needs a limit of >=10e-7 to avoid numerical noise around z=0 for
        # small values of z with imaginary component
        jnp.where(z_is_small,
                  jnp.zeros(shape=(n1), dtype=jnp.complex128).at[0].set(1.0),
                  bessel(_z, n1))
    )

@jax.jit
def custom_spherical_jn(n, z):
    return jax.lax.switch(n, BESSEL_FUNCTIONS, z)


# need to find a better way to do this; not available in JAX yet
@partial(jax.jit, static_argnames=('n1',))
def bessel(z, n1):
    """Spherical Bessel functions. Evaluated at z, up to degree n1."""
    vmapped_custom_bessel = jax.vmap(custom_spherical_jn, (0, None))
    return vmapped_custom_bessel(jnp.arange(n1), z)


#@partial(jax.jit, static_argnames=('LMAX',))
def HARMONY(C, LMAX):
    """Generates the spherical harmonics for the vector C.

    This is a python implementation of the fortran subroutine HARMONY from
    TensErLEED. It uses the jax.scipy.special.sph_harm function to produce
    equivalent results."""
    r = safe_norm(C)
    eps_sign_z = EPS*jnp.sign(C[0])
    theta = jnp.arccos(_divide_zero_safe(C[0], r, 1.0)-eps_sign_z)
    # Alternative implementation to avoid division by zero:
    # theta = jnp.arccos((C[0]+eps_sign_z)/(r+EPS)-eps_sign_z)
    phi = jnp.arctan2(C[2]+EPS, C[1]+EPS)
    return sph_harm(DENSE_M[2*LMAX], DENSE_L[2*LMAX], jnp.asarray([phi]), jnp.asarray([theta]), n_max=LMAX)
