from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from line_profiler import profile

from src.lib_math import *
from src.dense_quantum_numbers import DENSE_QUANTUM_NUMBERS, DENSE_L, DENSE_M
from src.dense_quantum_numbers import MINUS_ONE_POW_M
from src.dense_quantum_numbers import  map_l_array_to_compressed_quantum_index

from src.gaunt_coefficients import fetch_stored_gaunt_coeffs as fetch_gaunt
from src.gaunt_coefficients import PRE_CALCULATED_CPPP

fetch_lpp_gaunt = jax.vmap(fetch_gaunt,
                            in_axes=(None, None, 0, None, None, None),
                            out_axes=0)

# TODO: could we switch the entire calculation to use eV and Angstroms?
HARTREE = 27.211386245
BOHR = 0.529177211

@partial(jit, static_argnames=('LMAX',))
@partial(vmap, in_axes=(None, 0, None, 0))  # vmap over atoms
def tscatf(LMAX, phaseshifts, e_inside, DR):
    """Computes the temperature-dependent t-matrix elements.

    Takes the interpolated phase shifts and computes the atomic t-matrix
    elements. Thermal vibrations are taken into account through a Debye-Waller
    factor, whereby isotropic vibration amplitudes are assumed.
    
    # TODO @Paul: Finish desciption. Add here, which math is used and reference to the book.

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
    FALFE = -2/3 * DR**2 * e_inside
    Z = FALFE * 1j

    # TODO: @Paul choose better variable names
    FL = (2*jnp.arange(2*LMAX+1) + 1)
    CS = 1j ** jnp.arange(2*LMAX+1)
    BJ = jnp.exp(FALFE) * FL * CS * bessel(Z, 2*LMAX+1)

    CTAB = (jnp.exp(2j*phaseshifts)-1)*(2*jnp.arange(LMAX+1) + 1)

    SUM = jnp.einsum('jki,i,j->k', PRE_CALCULATED_CPPP[LMAX], CTAB, BJ)
    t_matrix = (SUM)/(2j) # temperature-dependent t-matrix.
    # SUM is the factor exp(2*i*delta) -1
    # Equation (22), page 29 in Van Hove, Tong book
    # Unlike TensErLEED, we do not convert it to a phase shift, but keep it as a
    # t-matrix, which we use going forward.
    return t_matrix


def MATEL_DWG(t_matrix_ref,t_matrix_new,e_inside,v_imag,LMAX,tensor_amps_out,tensor_amps_in,out_k_par2,
      out_k_par_3,unit_cell_area,displacements):
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
    e_broadcast = jnp.ones_like(v_imag)*e_inside
    k_inside = jnp.sqrt(2*e_broadcast-2j*v_imag+1j*EPS)

    # EXLM is for outgoing beams, so we need to swap indices m -> -m
    # to do this in the dense representation, we do the following:
    tensor_amps_out = tensor_amps_out[:,(DENSE_L[LMAX]+1)**2 - DENSE_L[LMAX] - DENSE_M[LMAX] -1]

    #   The vector C must be expressed W.R.T. a right handed set of axes.
    #   CDISP() is input W.R.T. a left handed set of axes.
    C = displacements/BOHR
    C = C * jnp.array([1, 1, -1])


    #   Evaluate DELTAT matrix for current displacement vector
    DELTAT = TMATRIX_DWG(t_matrix_ref,t_matrix_new,C, e_inside,v_imag,LMAX)


    AMAT = jnp.einsum('l,alb,alk,ak->ab',
                      MINUS_ONE_POW_M[LMAX], 
                      tensor_amps_out,
                      DELTAT, tensor_amps_in)

    # the propagator is evaluated relative to the muffin tin zero i.e.
    # it uses energy = incident electron energy + inner potential
    out_k_par = out_k_par2**2 + out_k_par_3**2
    energy_broadcast = (2*e_broadcast - 2j*v_imag + 1j*EPS) # @ jnp.ones_like(out_k_par)
    out_k_perp_inside = jnp.sqrt(energy_broadcast@jnp.ones_like(out_k_par) - out_k_par)

    # Equation (41) from Rous, Pendry 1989 & sum over atoms (index a)
    AMAT = jnp.einsum('ab,a,ab->b', AMAT, 1/k_inside, 1/out_k_perp_inside)
    AMAT = AMAT/(2*unit_cell_area)

    return AMAT


#@partial(jit, static_argnames=('LMAX',))
@partial(vmap, in_axes=(0, 0, 0, None, 0, None))  # vmap over atoms
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
    YLM(LMMAX): Spherical harmonics of vector C.
    GTWOC(LMMAX,LMMAX): Propagator from origin to C."""
    CL = safe_norm(C)

    #TODO: I disabled this for now, because I believe the conditional is going 
    #      to be slower than just computing the DELTAT matrix.
    """
    #   If displacement = 0, calculate DELTAT and jump to end
        if CL <= 1.0e-7:
    #       Calcualte DELTAT
            for L in range(LSMAX+1):
                for M in range(-L,L+1):
                    I = L+1
                    I = I*I-L+M
                    DELTAT[I-1][I-1] = 1.0j*(NewAF[L]-AF[L])
            return DELTAT
    """

    CAPPA = 2*energies - 2j*v_imag
    Z = jnp.sqrt(CAPPA)*CL
    BJ = masked_bessel(Z,2*LMAX+1)
    YLM = HARMONY(C, LMAX)
    GTWOC = get_csum(BJ, YLM, LMAX, DENSE_QUANTUM_NUMBERS[LMAX])

    broadcast_New_t_matrix = map_l_array_to_compressed_quantum_index(corrected_t_matrix, LMAX)
    GTEMP = GTWOC.T*1.0j*broadcast_New_t_matrix

    DELTAT = jax.numpy.einsum('il,lj->ij', GTEMP, GTWOC)

    mapped_t_matrix_ref = map_l_array_to_compressed_quantum_index(t_matrix_ref, LMAX)
    DELTAT = DELTAT + jnp.diag(-1.0j*mapped_t_matrix_ref)

    return DELTAT


@partial(jit, static_argnames=('LMAX',), inline=True)
@partial(vmap, in_axes=(None, None, None, 0))
@partial(vmap, in_axes=(None, None, None, 0))
def get_csum(BJ, YLM, LMAX, l_lp_m_mp):
    L, LP, M, MP = l_lp_m_mp
    MPP = MP-M  # I don't fully understand this, technically it should be MPP = -M - MP
    all_lpp = jnp.arange(0, LMAX*2+1)
    # we could skip some computations with non_zero_lpp = jnp.where((all_lpp >= abs(L-LP)) & (all_lpp <= L+LP))
    # but this would need a way around the non-static array size

    # Use the array versions in the vmap call
    gaunt_coeffs = fetch_lpp_gaunt(L, LP, all_lpp, M, -MP, -M+MP)
    gaunt_coeffs = gaunt_coeffs*(-1)**(LP+MP)  #TODO: @Paul: I found we need this factor, but I still don't understand why
    bessel_values = BJ[all_lpp]
    ylm_values = YLM[all_lpp*all_lpp+all_lpp+1-MPP-1]
    # Equation (34) from Rous, Pendry 1989
    csum = jnp.sum(bessel_values*ylm_values*gaunt_coeffs*1j**(L-LP-all_lpp))
    csum = csum*4*jnp.pi
    return csum
