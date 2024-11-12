"""Module propagators."""
__authors__ = ("Alexander M. Imre (@amimre)",
               "Paul Haidegger (@Paulhai7)")
__created__ = "2024-09-03"

import jax
import jax.numpy as jnp
import numpy as np

from viperleed_jax.dense_quantum_numbers import DENSE_QUANTUM_NUMBERS
from viperleed_jax.gaunt_coefficients import CSUM_COEFFS
from viperleed_jax.lib_math import bessel, HARMONY, safe_norm, EPS
from viperleed_jax.atomic_units import kappa

# TODO: replace energy, v_imag with a single arg kappa = 2*energy - 2j*v_imag
def calc_propagator(LMAX, c, energy, v_imag):
    c_norm = safe_norm(c)

    BJ = bessel(kappa(energy, v_imag) * c_norm, 2*LMAX)
    YLM = HARMONY(c, LMAX)  # TODO: move outside since it's not energy dependent

    dense_m_2d = DENSE_QUANTUM_NUMBERS[LMAX][:, :, 2]
    dense_mp_2d =  DENSE_QUANTUM_NUMBERS[LMAX][:, :, 3]

    # AI: I don't fully understand this, technically it should be MPP = -M - MP
    dense_mpp = dense_mp_2d - dense_m_2d

    # pre-computed coeffs, capped to LMAX
    capped_coeffs = CSUM_COEFFS[:2*LMAX+1, :(LMAX+1)**2, :(LMAX+1)**2]

    def propagator_lpp_element(lpp, running_sum):
        bessel_values = BJ[lpp]
        ylm_values = YLM[lpp*lpp+lpp-dense_mpp]
        # Equation (34) from Rous, Pendry 1989
        return running_sum + bessel_values * ylm_values * capped_coeffs[lpp,:,:] #* (abs(dense_mpp) <= lpp)

    # we could skip some computations because some lpp are guaranteed to give
    # zero contributions, but this would need a way around the non-static array
    # sizes

    # This is the propagator from the origin to C
    propagator = jax.lax.fori_loop(0, LMAX*2+1, propagator_lpp_element,
                             jnp.zeros(shape=((LMAX+1)**2, (LMAX+1)**2),
                                       dtype=jnp.complex128))
    propagator *= 4*jnp.pi
    return jnp.where(c_norm >= EPS*100, propagator, jnp.identity((LMAX+1)**2))

# Using equation (34) from Rous, Pendry 1989 it is easy to show that the
# propagator for a vanishing displacement is the identity matrix.
# (The Bessel functions for zero argument are zero for all non-zero orders, thus
# l''=0 is the only non-zero term in the sum. If l'' is 0, m''=0 and l=l' are
# necessary conditions.)


def symmetry_operations(l_max, plane_symmetry_operation):
    """_summary_

    Parameters
    ----------
    l_max : int
        Maximum angular momentum quantum number. Compiled as static argument.
    plane_symmetry_operation : 2x2 array
        The in-plane symmetry operation matrix.

    Returns
    -------
    jax.numpy.ndarray, shape=((l_max+1)**2, (l_max+1)**2)
        Tensor that can be applied element-wise to the propagator to apply the
        symmetry operation.
    bool
        Whether the symmetry operation is a mirror operation. If True, the
        propagator should be transposed.
    """
    dense_m_2d = DENSE_QUANTUM_NUMBERS[l_max][:, :, 2]
    dense_mp_2d =  DENSE_QUANTUM_NUMBERS[l_max][:, :, 3]
    # AI: I don't fully understand this, technically it should be MPP = -M - MP
    dense_mpp = dense_mp_2d - dense_m_2d

    plane_symmetry_det = np.linalg.det(plane_symmetry_operation)
    if abs(plane_symmetry_det) -1 > 1e-8:
        raise ValueError("The determinant of the plane symmetry operation "
                         "matrix must be 1 or -1.")
    contains_mirror = plane_symmetry_det < 0
    mirror_x = np.array([[-1., 0.], [0., 1.]])
    if contains_mirror:
        sym_op = plane_symmetry_operation @ mirror_x
    else:
        sym_op = plane_symmetry_operation

    plane_rotation_angle = get_plane_symmetry_operation_rotation_angle(sym_op)

    symmetry_tensor = jnp.exp(plane_rotation_angle*1j*(dense_mpp)).T
    if contains_mirror:
        symmetry_tensor = (-1.)**(-dense_mpp)* jnp.exp(plane_rotation_angle*1j*(dense_mpp)).T

    return symmetry_tensor, contains_mirror


def get_plane_symmetry_operation_rotation_angle(plane_symmetry_operation):
    """Return the rotation angle for a plane symmetry operation.

    The rotation angle is calculated for the plane symmetry operation by
    embedding it in 3D space. Note that for non-rotation matrices, a rotation
    will still be calculated, but it will be meaningless.

    Parameters
    ----------
    plane_symmetry_operation : ndarray (2,2)
        In plane symmetry operation matrix.

    Returns
    -------
    float
        Rotation angle in radians.
    """
    return (np.log(plane_symmetry_operation[0,0] +
                   1j*plane_symmetry_operation[1, 0])/1j).real
