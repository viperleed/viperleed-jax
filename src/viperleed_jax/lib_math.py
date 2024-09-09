from jax.scipy.special import sph_harm
import jax
import jax.numpy as jnp

from spbessax import functions

from viperleed_jax.dense_quantum_numbers import DENSE_M, DENSE_L
from viperleed_jax.dense_quantum_numbers import MAXIMUM_LMAX

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

@jax.named_scope("safe_norm")
def safe_norm(vector: jnp.ndarray) -> jnp.ndarray:
    """Safe norm calculation to avoid NaNs in gradients"""
    # avoids nan in gradient for jnp.linalg.norm(C)
    return jnp.sqrt(jnp.sum(vector**2) + (EPS*1e-2)**2)


def _generate_bessel_functions(l_max):
    """Generate a list of spherical Bessel functions up to order l_max"""
    bessel_functions = []
    for order in range(l_max+1):
        bessel_functions.append(functions.create_j_l(order,
                                                     dtype=jnp.complex128,
                                                     output_all=True))
    return bessel_functions


# generate a list of spherical Bessel functions up to order l_max
BESSEL_FUNCTIONS = _generate_bessel_functions(2*MAXIMUM_LMAX)


# Bessel functions from NeuralIL
@jax.named_scope("bessel")
def bessel(z, n1):
    """Spherical Bessel functions. Evaluated at z, up to degree n1."""
    return BESSEL_FUNCTIONS[n1](z+EPS)


@jax.named_scope("HARMONY")
def HARMONY(C, LMAX):
    """Generates the spherical harmonics for the vector C.

    This is a python implementation of the fortran subroutine HARMONY from
    TensErLEED. It uses the jax.scipy.special.sph_harm function to produce
    equivalent results."""
    _, theta, phi = cart_to_polar(C)
    l = DENSE_L[2*LMAX]
    m = DENSE_M[2*LMAX]

    is_on_pole_axis = abs(theta)<=EPS
    _theta = jnp.where(is_on_pole_axis, 0.1, theta)

    # values at the poles(theta = 0) depend on l and m only
    pole_values = (m == 0)*jnp.sqrt((2*l+1)/(4*jnp.pi))
    non_pole_values = sph_harm(m, l,
                               jnp.asarray([phi]), jnp.asarray([_theta]),
                               n_max=2*LMAX)

    return jnp.where(is_on_pole_axis, pole_values, non_pole_values)


def cart_to_polar(c):
    """Converts cartesian coordinates to polar coordinates.

    Note, this function uses safe division to avoid division by zero errors, 
    and gives defined results and gradients for all inputs, EXCEPT for
    c = (0.0, 0.0, 0.0)."""
    z, x, y = c  # LEED coordinates

    x_y_norm = jnp.hypot(x, y)
    r = jnp.linalg.norm(c)
    theta = 2*jnp.arctan(
        _divide_zero_safe(x_y_norm, (jnp.hypot(x_y_norm, z)+z), (1/EPS) * (1 - jnp.sign(z)))
    )

    # forces phi to 0 on theta=0 axis (where phi is undefined)
    phi = 2*jnp.arctan(
        _divide_zero_safe(y, (x_y_norm+x)+EPS, 0.0)
    )

    return r, theta, phi
    
def spherical_to_cart(spherical_coordinates):

    r, theta, phi = spherical_coordinates
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)

    return jnp.array([z, x, y])

