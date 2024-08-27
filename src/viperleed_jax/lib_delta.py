from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial

from otftleed.lib_math import *
from otftleed.dense_quantum_numbers import DENSE_QUANTUM_NUMBERS
from otftleed.dense_quantum_numbers import  map_l_array_to_compressed_quantum_index

from otftleed.constants import BOHR

from otftleed.gaunt_coefficients import CSUM_COEFFS

@jax.named_scope("apply_geometric_displacements")
def apply_geometric_displacements(t_matrix_ref,t_matrix_new,e_inside,v_imag,
                                  LMAX,tensor_amps_out,tensor_amps_in,
                                  displacements):
    """Evaluates the amplitude change due to displacement for each exit beam.
    
    Using the the tensor produced by the reference calculation, and the adapted
    t-matrix elements produced by tscatf, this function calculates the change in
    amplitude for each of the exit beams for a given displacement vector.

    Parameters
    ----------
    displacements : array
        Displacement vector for all atoms.

    Note
    ----
    This function corresponds to the subroutine MATEL_DWG in TensErLEED.
    """
    # convert to atomic units
    _C = displacements/BOHR

    # The vector C must be expressed W.R.T. a right handed set of axes.
    # CDISP() is input W.R.T. a left handed set of axes.
    _C = _C * jnp.array([1, 1, -1])

    # Evaluate DELTAT matrix for current displacement vector
    DELTAT = TMATRIX_DWG(t_matrix_ref, t_matrix_new, _C, e_inside,v_imag,LMAX)

    # Equation (41) from Rous, Pendry 1989 & sum over atoms (index a)
    AMAT = jnp.einsum('abl,alk,ak->b',
                      tensor_amps_out,
                      DELTAT,
                      tensor_amps_in)

    return AMAT


@jax.named_scope("TMATRIX_DWG")
@partial(vmap, in_axes=(0, 0, 0, None, None, None))  # vmap over atoms
@partial(jax.jit, static_argnames=('v_imag', 'LMAX'))
def TMATRIX_DWG(t_matrix_ref, corrected_t_matrix, C, energy, v_imag, LMAX):
    """The function TMATRIX_DWG generates the TMATRIX(L,L') matrix for given energy & displacement vector.
    E,VPI: Current energy (real, imaginary).
    C(3): Displacement vector;
        C(1)= Component along x axis into the surface
        C(2)= Component along y axis
        C(3)= Component along z axis
    DELTAT(LMMAX,LMMAX): Change in t matrix caused by the displacement.
    AF(LMAX1): exp(i*PHS(L))*sin(PHS(L)). Note that atomic t matrix is i*AF.
    BJ(LMAX1): Bessel functions for each L.
    YLM(LMMAX): Spherical harmonics of vector C."""
    CL = safe_norm(C)

    # mask with dummy value to avoid division by zero
    C_masked = jnp.where(CL <= EPS, jnp.array([100*EPS, 100*EPS, 100*EPS]), C)

    # if the displacement is zero, we branch to the simplified calculation
    without_displacement = TMATRIX_zero_displacement(t_matrix_ref, corrected_t_matrix, C, energy, v_imag, LMAX)
    with_displacement = TMATRIX_non_zero_displacement(t_matrix_ref, corrected_t_matrix, C_masked, energy, v_imag, LMAX)
    DELTAT = jnp.where(CL <= EPS, without_displacement, with_displacement)

    return DELTAT


@jax.named_scope("TMATRIX_non_zero_displacement")
def TMATRIX_non_zero_displacement(t_matrix_ref, corrected_t_matrix, C, energy, v_imag, LMAX):

    propagator = calc_propagator(LMAX, C, energy, v_imag)

    broadcast_New_t_matrix = map_l_array_to_compressed_quantum_index(corrected_t_matrix, LMAX)

    DELTAT = jnp.einsum('ji,j,lj->il',
                        propagator, 1j * broadcast_New_t_matrix, propagator, optimize=True)

    mapped_t_matrix_ref = map_l_array_to_compressed_quantum_index(t_matrix_ref, LMAX)
    DELTAT = DELTAT - jnp.diag(1j*mapped_t_matrix_ref)

    return DELTAT


@jax.named_scope("TMATRIX_zero_displacement")
def TMATRIX_zero_displacement(t_matrix_ref, corrected_t_matrix, C, energy, v_imag, LMAX):
    mapped_t_matrix_new = map_l_array_to_compressed_quantum_index(corrected_t_matrix, LMAX)
    mapped_t_matrix_ref = map_l_array_to_compressed_quantum_index(t_matrix_ref, LMAX)
    DELTAT = jnp.diag(1j*mapped_t_matrix_new) - jnp.diag(1j*mapped_t_matrix_ref)

    return DELTAT

# TODO: move this to a separate file and write tests
def calc_propagator(LMAX, C, energy, v_imag):
    c_norm = safe_norm(C)
    kappa = 2*energy - 2j*v_imag
    Z = jnp.sqrt(kappa) * c_norm
    BJ = bessel(Z,2*LMAX)
    YLM = HARMONY(C, LMAX)  # move outside since it's not energy dependent

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
        return running_sum + bessel_values * ylm_values * capped_coeffs[lpp,:,:]

    # we could skip some computations because some lpp are guaranteed to give
    # zero contributions, but this would need a way around the non-static array
    # sizes

    # This is the propagator from the origin to C
    propagator = jax.lax.fori_loop(0, LMAX*2+1, propagator_lpp_element,
                             jnp.zeros(shape=((LMAX+1)**2, (LMAX+1)**2),
                                       dtype=jnp.complex128))
    propagator *= 4*jnp.pi
    return propagator
