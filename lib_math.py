from functools import lru_cache
import numpy as np
from jax.scipy.special import sph_harm
import jax
import jax.numpy as jnp
from functools import partial

# Spherical Bessel functions from NeuralIL
from spherical_bessel import functions

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

@partial(jax.jit, static_argnames=('BESSEL_FUNCTIONS',))
def custom_spherical_jn(n, z):
    return jax.lax.switch(n, BESSEL_FUNCTIONS, z)

@lru_cache(maxsize=None)
def fac(n):
    return n * fac(n-1) if n else 1


def cppp(n1, n2, n3):
    """Tabulates the function PPP(I1,I2,I3), each element containing the integral of the product of three Legendre
    functions P(I1),P(I2),P(I3). The integrals are calculated following Gaunt's summation scheme set out by Slater
    atomic structure.
    PPP is used by function PSTEMP in computing temperature-depending phase shifts.
    Author Pendry"""
    ppp = np.full(shape=(n1, n2, n3), fill_value=np.nan)
    for i1 in range(1, n1+1):
        for i2 in range(1, n2+1):
            for i3 in range(1, n3+1):
                im1, im2, im3 = sorted((i1, i2, i3), reverse=True)
                A = 0.0
                iss = i1 + i2 + i3 - 3
                if (iss % 2 == 1) or (abs(im2-im1)+1 > im3):
                    ppp[i1-1, i2-1, i3-1] = A
                else:
                    ssum = 0.0
                    iss = int(iss/2)
                    sign = 1.0
                    for it in range(1, im3+1):
                        sign = - sign
                        ia1 = im1 + it - 1
                        ia2 = im1 - it + 1
                        ia3 = im3 - it + 1
                        ia4 = im2 + im3 - it
                        ia5 = im2 - im3 + it
                        ssum -= sign*np.math.factorial(ia1-1)*np.math.factorial(ia4-1) / \
                            (np.math.factorial(ia2-1)*np.math.factorial(ia3-1)*np.math.factorial(ia5-1)*np.math.factorial(it-1))
                    ia1 = 2 + iss - im1
                    ia2 = 2 + iss - im2
                    ia3 = 2 + iss - im3
                    ia4 = 3 + 2 * (iss - im3)
                    A = - (-1)**(iss-im2)*fac(ia4-1)*fac(iss)*fac(im3-1) * \
                        ssum/(np.math.factorial(ia1-1)*np.math.factorial(ia2-1)*np.math.factorial(ia3-1)*np.math.factorial(2*iss+1))
                    ppp[i1-1, i2-1, i3-1] = A
    return ppp


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
    r = jnp.sqrt(C[0] ** 2 + C[1] ** 2 + C[2] ** 2)
    theta = jnp.arccos(C[0] / r)
    phi = jnp.arctan2(C[2], C[1])
    return sph_harm(dense_m, dense_l, jnp.array([theta]), jnp.array([phi]), n_max=LMAX)
