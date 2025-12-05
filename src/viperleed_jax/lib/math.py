"""Module lib_math."""

__authors__ = ('Alexander M. Imre (@amimre)', 'Paul Haidegger (@Paulhai7)')
__created__ = '2024-01-02'

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from spbessax import functions

# in June 2025 jax.scipy.special.sph_harm was deprecated in favor of
# jax.scipy.special.sph_harm_y which has swapped theta & phi and m & n
try:
    from jax.scipy.special import sph_harm_y
except ImportError:
    from jax.scipy.special import sph_harm

    def sph_harm_y(n, m, phi, theta, n_max):
        """Wrap sph_harm_y to maintain compatibility."""
        return sph_harm(m, n, theta, phi, n_max)


from viperleed_jax.dense_quantum_numbers import DENSE_L, DENSE_M, MAXIMUM_LMAX

# numerical epsilon to avoid division by zero
EPS = 1e-8


def divide_zero_safe(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
    limit_value: float = 0.0,
) -> jnp.ndarray:
    """Divide two arrays with a safe handling of division by zero.

    Forces the result of dividing by 0 to be equal to a limit value in a jit-
    and autodiff-compatible way.

    Parameters
    ----------
        numerator: jnp.ndarray
            Values in the numerator.
        denominator: jnp.ndarray
            Values in the denominator, may contain zeros
        limit_value: jnp.ndarray, optional..
            Value to return where denominator == 0.0. Default is 0.0.

    Returns
    -------
        numerator / denominator with result == 0.0 where denominator == 0.0
    """
    denominator_masked = jnp.where(denominator == 0.0, 1.0, denominator)
    return jnp.where(
        denominator == 0.0,
        limit_value,
        numerator / denominator_masked,
    )


def safe_norm(vector: jnp.ndarray) -> jnp.ndarray:
    """Safe norm calculation to avoid NaNs in gradients."""
    # avoids nan in gradient for jnp.linalg.norm(C)
    return jnp.sqrt(jnp.sum(vector**2) + (EPS * 1e-2) ** 2)


def _generate_bessel_functions(l_max):
    """Generate a list of spherical Bessel functions up to order l_max."""
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
    z, *_ = vector
    _, theta, phi = cart_to_polar(vector)
    l = DENSE_L[2 * l_max]
    m = DENSE_M[2 * l_max]

    is_on_pole_axis = (abs(theta) <= EPS) | (abs(theta - jnp.pi) <= EPS)
    _theta = jnp.where(is_on_pole_axis, 0.1, theta)

    # values at the poles(theta = 0) depend on l and m only
    pole_values = (m == 0) * jnp.sqrt((2 * l + 1) / (4 * jnp.pi))
    pole_values = (z < 0) * ((-1) ** l) * pole_values + (z >= 0) * pole_values
    non_pole_values = sph_harm_y(
        l, m, jnp.asarray([_theta]), jnp.asarray([phi]), n_max=2 * l_max
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
        divide_zero_safe(
            x_y_norm,
            (jnp.hypot(x_y_norm, z) + z),
            (1 / EPS) * (1 - jnp.sign(z)),
        )
    )

    # forces phi to 0 on theta=0 axis (where phi is undefined)
    # phi = 2*jnp.arctan(
    #     _divide_zero_safe(y, (x_y_norm+x)+EPS, 0.0)
    # )
    phi = jnp.sign(y) * jnp.arccos(divide_zero_safe(x, (x_y_norm) + EPS, 0.0))
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
        divide_zero_safe(
            x_y_norm,
            (jnp.hypot(x_y_norm, z) + z),
            (1 / EPS) * (1 - jnp.sign(z)),
        )
    )

    # forces phi to 0 on theta=0 axis (where phi is undefined)
    # phi = 2*jnp.arctan(
    #     _divide_zero_safe(y, (x_y_norm+x)+EPS, 0.0)
    # )
    phi = jnp.sign(y) * jnp.arccos(divide_zero_safe(x, (x_y_norm) + EPS, 0.0))
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


def mirror_across_plane_sum_1(vector):
    """Mirror a vector across the plane defined by sum(x_i) = 1.

    This function reflects an input vector across the hyperplane
    sum(x_i) = 1, such that the mirrored vector lies on the opposite
    side of the plane, preserving orthogonal distance.

    Parameters
    ----------
    vector : array_like
        Input 1D vector of shape (n,).

    Returns
    -------
    mirrored_vector : jax.Array
        1D array of shape (n,) representing the mirrored vector.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from your_module import mirror_across_plane_sum_1
    >>> v = jnp.array([0.6, 0.4, 0.2])
    >>> mirror_across_plane_sum_1(v)
    Array([0.4666667, 0.2666667, 0.0666667], dtype=float32)
    >>> jnp.sum(v), jnp.sum(mirror_across_plane_sum_1(v))
    (Array(1.2, dtype=float32), Array(0.8, dtype=float32))
    """
    _vector = jnp.asarray(vector)
    dim = _vector.shape[0]

    # Compute how far the sum deviates from 1
    deviation = jnp.sum(_vector) - 1.0

    # Create a correction vector in the direction of (1,1,...,1)
    correction = (2 * deviation / dim) * jnp.ones_like(_vector)

    return _vector - correction


@partial(jax.jit, static_argnames=('index', 'func'))
def apply_fun_grouped(in_vec, index, func, group_args=None):
    """Apply a function separately to groups determined by an index array.

    For each unique index value in `index`, this function collects all elements
    in `in_vec` with the same index, applies `func` to that group, and writes
    the transformed elements back into their original positions.

    Parameters
    ----------
    in_vec : jax.Array
        Input vector of shape (n,). The elements to be grouped and transformed.
    index : jax.Array
        Integer array of shape (n,) indicating group membership. Elements
        with the same index value belong to the same group.
    func : callable
        Function that takes the group array as its first argument and any
        additional group arguments, and returns a 1D array of the same length
        as the input group.
        Signature: `func(group_elements, group_arg1, group_arg2, ...)`
    group_args : jax.Array or tuple[jax.Array], optional
        Auxiliary arguments for `func`. Each array must be 1D with a length
        equal to the number of unique groups. The element at position `i` in
        each array is passed as an argument to `func` when processing the
        group identified by index `i`. Defaults to **None**, in which case
        `func` should only accept the group array.

    Returns
    -------
    out_vec : jax.Array
        Output vector of shape (n,) where each group has been independently
        transformed by `func`.

    Notes
    -----
    - `func` must be shape-preserving for each group (input and output lengths
      must match).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> v = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> ind = jnp.array([0, 1, 0, 1, 2, 2])
    >>> # Define group-specific arguments: one for each unique group
    >>> # index (0, 1, 2)
    >>> factors = jnp.array([2.0, 3.0, 0.5])  # Factor for group 0, 1, and 2
    >>> def scale(x, factor):
    ...     return x * factor
    >>> apply_fun_grouped(v, ind, scale, factors)
    Array([ 2. ,  6. ,  6. , 12. ,  2.5,  3. ], dtype=float32)

    """
    unique_inds = np.unique(index)

    # handle the default None case
    if group_args is None:
        group_args = ()
    elif not isinstance(group_args, (tuple, list)):
        group_args = (group_args,)
    group_args = tuple(group_args)  # ensure it's a tuple for indexing

    # prepare output buffer
    out_vec = jnp.zeros(in_vec.shape[0])

    for idx in unique_inds:
        mask = index == idx
        group = in_vec[mask]
        # Pass the group-specific argument for the current index 'idx'
        transformed = func(group, *(arg[idx] for arg in group_args))
        # Store the mask and transformed group
        # put into the output vector
        out_vec = out_vec.at[mask].set(transformed)

    return out_vec
