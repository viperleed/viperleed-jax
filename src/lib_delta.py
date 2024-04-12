from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

from src.lib_math import *
from src.dense_quantum_numbers import DENSE_QUANTUM_NUMBERS, DENSE_L, DENSE_M
from src.dense_quantum_numbers import MINUS_ONE_POW_M
from src.dense_quantum_numbers import  map_l_array_to_compressed_quantum_index

# TODO: could we switch the entire calculation to use eV and Angstroms?
from src.constants import BOHR, HARTREE

from src.gaunt_coefficients import fetch_stored_gaunt_coeffs as fetch_gaunt
from src.gaunt_coefficients import PRE_CALCULATED_CPPP, CSUM_COEFFS




@partial(vmap, in_axes=(None, 0, None, 0))  # vmap over atoms
def apply_vibrational_displacements(LMAX, phaseshifts, e_inside, DR):
    """Computes the temperature-dependent t-matrix elements.

    Takes the interpolated phase shifts and computes the atomic t-matrix
    elements. Thermal vibrations are taken into account through a Debye-Waller
    factor, whereby isotropic vibration amplitudes are assumed.

    The entire function comprises equations (22), (23), and (24), page 29 of 
    the Van Hove and Tong, 1979. Vibrational amplitudes are transformed into
    a Debye-Waller factor.  It's important to note that the vibrational
    amplitudes are the sole temperature-dependent component. Therefore, 
    utilizing a temperature-independent vibration amplitude obviates the need 
    to explicitly include temperature, and the phaseshifts in (23) depend on
    vibrations only. Up to and including the calculation of SUM, every
    operation is derived from (23).

    The factor PRE_CALCULATED_CPPP is defined as 
    (4Pi/((2l+1)(2l'+1)(2*l''+1)))^0.5 * Gaunt(l,0,l',0,l'',0). The factor BJ
    is an array of Bessel functions and contains all terms dependent on l'. The
    factor CTAB includes all other terms dependent on l''.

    To compute the t-matrix, the resulting term is divided by 4ik_0 (eq. 22). 
    In the code SUM is only devided by 2i.

    Parameters
    ----------
    LMAX : int
        Maximum angular momentum quantum number.
    phaseshifts : array
        Interpolated phase shifts.
    e_inside : float
        Current energy (real number).
    DR : float
        Isotropic RMS vibration amplitude.

    Returns
    -------
    t_matrix : array
        Temperature-dependent atomic t-matrix (complex number).

    Note
    ----
    This function corresponds loosely to the subroutines TSCATF and PSTEMP in
    TensErLEED. Those subroutines used to take a reference temperature T0 and
    a current temperature TEMP as input. However, this functionality was not
    implemented in TensErLEED and is not implemented here either.

    Similarly, the TensErLEED versions used to take a zero-temperature
    vibrational amplitude DR0, and an anisotropic current vibrational amplitude
    (in the form of DRPER and DRPAR, perpendicular and parallel to the surface,
    respectively) as input. This functionality was also not implemented, with
    DR0 hardcoded to 0 and DRPER and DRPAR hardcoded to be equal. We thus only
    implement the isotropic case here, with a single input parameter DR.

    Finally, the TensErLEED versions used allow a local variation of the of the
    muffin-tin constant, via a parameter VSITE that shifts the used energy in
    the crystal potential as E -> E - VSITE. This functionality was also not
    included as VSITE was also hardcoded to 0 in the TensErLEED code.
    """
    _DR = DR/BOHR
    debye_waller_exponent = -2/3 * _DR**2 * e_inside

    all_l = (2*jnp.arange(2*LMAX+1) + 1)
    bessel_with_prefactor = (
        jnp.exp(debye_waller_exponent)
        * all_l
        * 1j ** jnp.arange(2*LMAX+1)
        * bessel(debye_waller_exponent * 1j, 2*LMAX+1)
    )

    temperature_independent_t_matrix = (
        jnp.exp(2j*phaseshifts)-1)*(2*jnp.arange(LMAX+1) + 1)

    SUM = jnp.einsum('jki,i,j->k',
                     PRE_CALCULATED_CPPP[LMAX],
                     temperature_independent_t_matrix,
                     bessel_with_prefactor)
    t_matrix = (SUM)/(2j) # temperature-dependent t-matrix.
    # SUM is the factor exp(2*i*delta) -1
    # Equation (22), page 29 in Van Hove, Tong book from 1979
    # Unlike TensErLEED, we do not convert it to a phase shift, but keep it as a
    # t-matrix, which we use going forward.
    return t_matrix


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
    AMAT = jnp.einsum('alb,alk,ak->b',
                      tensor_amps_out,
                      DELTAT,
                      tensor_amps_in)

    return AMAT


@partial(vmap, in_axes=(0, 0, 0, None, None, None))  # vmap over atoms
def TMATRIX_DWG(t_matrix_ref, corrected_t_matrix, C, energies, v_imag, LMAX):
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

    CAPPA = 2*energies - 2j*v_imag
    Z = jnp.sqrt(CAPPA)*CL
    BJ = masked_bessel(Z,2*LMAX+1)
    YLM = HARMONY(C, LMAX)

    dense_m_2d = DENSE_QUANTUM_NUMBERS[LMAX][:, :, 2]
    dense_mp_2d =  DENSE_QUANTUM_NUMBERS[LMAX][:, :, 3]

    # AI: I don't fully understand this, technically it should be MPP = -M - MP
    dense_mpp = dense_mp_2d - dense_m_2d

    # pre-computed coeffs, capped to LMAX
    capped_coeffs = CSUM_COEFFS[:2*LMAX+1, :(LMAX+1)**2, :(LMAX+1)**2]

    """lpp = 1
    bessel_values = BJ[lpp]
    ylm_values = YLM[lpp*lpp+lpp-dense_mpp]
    print(bessel_values, ylm_values, capped_coeffs[lpp,:,:])
    print(bessel_values*ylm_values*capped_coeffs[lpp,:,:])"""


    def csum_element(lpp, running_sum):
        bessel_values = BJ[lpp]
        ylm_values = YLM[lpp*lpp+lpp-dense_mpp]
        # Equation (34) from Rous, Pendry 1989
        return running_sum + bessel_values * ylm_values * capped_coeffs[lpp,:,:]

    # we could skip some computations because some lpp are guaranteed to give
    # zero contributions, but this would need a way around the non-static array
    # sizes
    csum = jax.lax.fori_loop(0, LMAX*2+1, csum_element,
                             jnp.zeros(shape=((LMAX+1)**2, (LMAX+1)**2),
                                       dtype=jnp.complex128))
    csum *= 4*jnp.pi
    # csum is the propagator from origin to C

    broadcast_New_t_matrix = map_l_array_to_compressed_quantum_index(corrected_t_matrix, LMAX)

    DELTAT = (csum * 1j * broadcast_New_t_matrix) @ csum

    # alternative einsum version:
    # DELTAT = jnp.einsum('ji,j,lj->il', csum, 1j * broadcast_New_t_matrix, csum, optimize=True)

    mapped_t_matrix_ref = map_l_array_to_compressed_quantum_index(t_matrix_ref, LMAX)
    DELTAT = DELTAT - jnp.diag(1j*mapped_t_matrix_ref)

    return DELTAT
