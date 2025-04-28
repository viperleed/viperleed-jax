"""Module lib_math."""

__authors__ = ('Alexander M. Imre (@amimre)', 'Paul Haidegger (@Paulhai7)')
__created__ = '2024-01-02'

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import sph_harm
from spbessax import functions
import numpy as np

from viperleed_jax.dense_quantum_numbers import DENSE_L, DENSE_M, MAXIMUM_LMAX

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
    return jnp.sqrt(jnp.sum(vector**2) + (EPS * 1e-2) ** 2)


def _generate_bessel_functions(l_max):
    """Generate a list of spherical Bessel functions up to order l_max"""
    bessel_functions = []
    for order in range(l_max + 1):
        bessel_functions.append(
            functions.create_j_l(order, dtype=jnp.complex128, output_all=True)
        )
    return bessel_functions


# generate a list of spherical Bessel functions up to order l_max
BESSEL_FUNCTIONS = _generate_bessel_functions(2 * MAXIMUM_LMAX)


# Bessel functions from NeuralIL
def bessel(z, n1):
    """Spherical Bessel functions. Evaluated at z, up to degree n1."""
    return BESSEL_FUNCTIONS[n1](z + EPS)


def spherical_harmonics_components(l_max, vector):
    """Generate the spherical harmonics for a vector.

    This is a python implementation of the fortran subroutine HARMONY from
    TensErLEED. It uses the jax.scipy.special.sph_harm function to produce
    equivalent results.
    """
    _, theta, phi = cart_to_polar(vector)
    l = DENSE_L[2 * l_max]
    m = DENSE_M[2 * l_max]

    is_on_pole_axis = abs(theta) <= EPS
    _theta = jnp.where(is_on_pole_axis, 0.1, theta)

    # values at the poles(theta = 0) depend on l and m only
    pole_values = (m == 0) * jnp.sqrt((2 * l + 1) / (4 * jnp.pi))
    non_pole_values = sph_harm(
        m, l, jnp.asarray([phi]), jnp.asarray([_theta]), n_max=2 * l_max
    )

    return jnp.where(is_on_pole_axis, pole_values, non_pole_values)


def cart_to_polar(c):
    """Convert cartesian coordinates to polar coordinates.

    Note, this function uses safe division to avoid division by zero errors,
    and gives defined results and gradients for all inputs, EXCEPT for
    c = (0.0, 0.0, 0.0).
    """
    z, x, y = c  # LEED coordinates

    x_y_norm = jnp.hypot(x, y)
    r = jnp.linalg.norm(c)
    theta = 2 * jnp.arctan(
        _divide_zero_safe(
            x_y_norm,
            (jnp.hypot(x_y_norm, z) + z),
            (1 / EPS) * (1 - jnp.sign(z)),
        )
    )

    # forces phi to 0 on theta=0 axis (where phi is undefined)
    # phi = 2*jnp.arctan(
    #     _divide_zero_safe(y, (x_y_norm+x)+EPS, 0.0)
    # )
    phi = jnp.sign(y) * jnp.arccos(_divide_zero_safe(x, (x_y_norm) + EPS, 0.0))
    phi = jnp.where(y != 0.0, phi, jnp.heaviside(-x, 0) * jnp.pi)

    return r, theta, phi


@jax.custom_jvp
def cart_to_polar_2(c):
    """Convert cartesian coordinates to polar coordinates.

    Note, this function uses safe division to avoid division by zero errors,
    and gives defined results and gradients for all inputs, EXCEPT for
    c = (0.0, 0.0, 0.0).
    """
    z, x, y = c  # LEED coordinates

    x_y_norm = jnp.hypot(x, y)
    r = jnp.linalg.norm(c)
    theta = 2 * jnp.arctan(
        _divide_zero_safe(
            x_y_norm,
            (jnp.hypot(x_y_norm, z) + z),
            (1 / EPS) * (1 - jnp.sign(z)),
        )
    )

    # forces phi to 0 on theta=0 axis (where phi is undefined)
    # phi = 2*jnp.arctan(
    #     _divide_zero_safe(y, (x_y_norm+x)+EPS, 0.0)
    # )
    phi = jnp.sign(y) * jnp.arccos(_divide_zero_safe(x, (x_y_norm) + EPS, 0.0))
    phi = jnp.where(y != 0.0, phi, jnp.heaviside(-x, 0) * jnp.pi)

    return r, theta, phi


@cart_to_polar_2.defjvp
def cart_to_polar_jacobian(primals, tangents):
    (z, x, y) = primals
    (dz, dx, dy) = tangents
    r, theta, phi = cart_to_polar(primals)
    x_y_norm = jnp.hypot(x, y)
    jacobian = jnp.array(
        [
            [z / r * dz, x / r * dx, y / r * dy],
            [
                -x_y_norm / r**2 * dz,
                x * z / (r**2 * x_y_norm) * dx,
                y * z / (r**2 * x_y_norm) * dy,
            ],
            [0, -y / (x_y_norm**2) * dx, x / (x_y_norm**2) * dy],
        ],
    )


def spherical_to_cart(spherical_coordinates):
    r, theta, phi = spherical_coordinates
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)

    return jnp.array([z, x, y])


def project_onto_plane_sum_1(vector):
    """Project a vector onto the plane defined by sum(x_i) = 1.

    This function orthogonally projects an input vector onto the hyperplane
    in which the sum of all elements equals 1.

    Parameters
    ----------
    vector : array_like
        Input 1D vector of shape (n,).

    Returns
    -------
    projected_vector : jax.Array
        1D array of shape (n,) representing the projection of the input
        vector onto the plane sum(x_i) = 1.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from your_module import project_onto_plane_sum_1
    >>> v = jnp.array([0.6, 0.4, 0.2])
    >>> project_onto_plane_sum_1(v)
    Array([0.5333333, 0.3333333, 0.1333333], dtype=float32)
    >>> jnp.sum(project_onto_plane_sum_1(v))
    Array(1., dtype=float32)
    """
    # transform to jax array if needed
    _vector = jnp.asarray(vector)

    dim = _vector.shape[0]
    project_through_origin = jnp.eye(dim) - jnp.ones((dim, dim)) / dim
    offset_to_sum_one_plane = jnp.ones(dim) / dim
    return project_through_origin @ _vector + offset_to_sum_one_plane


@partial(jax.jit, static_argnames=('index','fun'))
def apply_fun_grouped(in_vec, index, fun):
    """Apply a function separately to groups determined by an index array.

    For each unique index value in `index`, this function collects all elements
    in `in_vec` with the same index, applies `fun` to that group, and writes
    the transformed elements back into their original positions.

    Parameters
    ----------
    in_vec : jax.Array
        Input vector of shape (n,).
    index : tuple
        Integer array of shape (n,) indicating group membership.
    fun : callable
        Function that takes a 1D array of arbitrary length and returns a 1D
        array of the same length. Applied separately to each group.

    Returns
    -------
    out_vec : jax.Array
        Output vector of shape (n,) where each group has been independently
        transformed by `fun`.

    Notes
    -----
    - `fun` must be shape-preserving for each group (input and output lengths
      must match).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> v = jnp.array([1., 2., 3., 4., 5., 6.])
    >>> ind = jnp.array([0, 1, 0, 1, 2, 2])
    >>> def double(x):
    ...     return x * 2
    >>> apply_fun_grouped(v, ind, double)
    Array([ 2.,  4.,  6.,  8., 10., 12.], dtype=float32)
    """
    unique_inds = np.unique(index)

    # prepare output buffer
    out_vec = jnp.zeros_like(in_vec)

    for idx in unique_inds:
        mask = index == idx
        group = in_vec[mask]
        transformed = fun(group)
        # Store the mask and transformed group
        # put into the output vector
        out_vec = out_vec.at[mask].set(transformed)

    return out_vec
