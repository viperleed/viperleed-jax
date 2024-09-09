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


def symmetry_tensor(l_max, plane_symmetry_operation):
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
    """

    dense_m_2d = DENSE_QUANTUM_NUMBERS[l_max][:, :, 2]
    dense_mp_2d =  DENSE_QUANTUM_NUMBERS[l_max][:, :, 3]

    # AI: I don't fully understand this, technically it should be MPP = -M - MP
    dense_mpp = dense_mp_2d - dense_m_2d

    plane_rotation_angle = get_plane_symmetry_operation_rotation_angle(
        plane_symmetry_operation)

    symmetry_tensor = jnp.exp(plane_rotation_angle*1j*(dense_mpp)).T
    return symmetry_tensor


def get_plane_symmetry_operation_rotation_angle(plane_symmetry_operation):
    """Return the rotation angle for a plane symmetry operation.

    NB: The rotation angle is calculated for the plane symmetry operation by
    embedding it in 3D space. In-plane symmetry operations (even mirror
    operations) can be converted into a rotation operation in 3D space, as the
    z-movement of linked atoms is equal.

    Parameters
    ----------
    plane_symmetry_operation : ndarray (2,2)
        In plane symmetry operation matrix.

    Returns
    -------
    float
        Rotation angle in radians.
    """
    full_rot_mat = np.identity(3)
    full_rot_mat[1:, 1:] = plane_symmetry_operation
    # x vector (NB: vectors are [z,x,y])
    x_vec = np.array([0., 1., 0.])
    # apply rotation
    test_vec = full_rot_mat @ x_vec
    # calculate rotation angle
    return np.arccos(np.dot(test_vec, x_vec)
                     /(np.linalg.norm(test_vec)*np.linalg.norm(x_vec)))
