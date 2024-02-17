from functools import lru_cache
import numpy as np
from jax.scipy.special import sph_harm
import jax
import jax.numpy as jnp
from functools import partial

# Spherical Bessel functions from NeuralIL
from spherical_bessel import functions

# numerical epsilon to avoid division by zero
EPS = 1e-8

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

@jit
def custom_spherical_jn(n, z):
    return jax.lax.switch(n, BESSEL_FUNCTIONS, z)

@lru_cache(maxsize=None)
def fac(n):
    return n * fac(n-1) if n else 1



# need to find a better way to do this; not available in JAX yet
@partial(jax.jit, static_argnames=('n1',))
def bessel(z, n1):
    """Spherical Bessel functions. Evaluated at z, up to degree n1."""
    bj = jnp.empty(shape=(n1,), dtype=jnp.complex128)
    vmapped_custom_bessel = jax.vmap(custom_spherical_jn, (0, None))
    return vmapped_custom_bessel(jnp.arange(n1), z)


# TODO: jit once fixed for neg values by Paul
@partial(jax.jit, static_argnames=('LMAX',))
def HARMONY(C, LMAX, dense_l, dense_m):
    """Generates the spherical harmonics for the vector C.

    This is a python implementation of the fortran subroutine HARMONY from
    TensErLEED. It uses the jax.scipy.special.sph_harm function to produce
    equivalent results."""
    r = jnp.sqrt(C[0] ** 2 + C[1] ** 2 + C[2] ** 2 + EPS)
    theta = jnp.arccos(C[0] / r)
    phi = jnp.arctan2(C[2], C[1])
    return sph_harm(dense_m, dense_l, jnp.array([phi]), jnp.array([theta]), n_max=LMAX)
