import numpy as np
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit, vmap
import fortranformat as ff
from functools import partial
from line_profiler import profile


from lib_math import *
from dense_quantum_numbers import DENSE_QUANTUM_NUMBERS, DENSE_L, DENSE_M
from dense_quantum_numbers import MINUS_ONE_POW_M
from dense_quantum_numbers import  map_l_array_to_compressed_quantum_index

from gaunt_coefficients import fetch_stored_gaunt_coeffs as fetch_gaunt
from gaunt_coefficients import PRE_CALCULATED_CPPP

fetch_lpp_gaunt = jax.vmap(fetch_gaunt,
                            in_axes=(None, None, 0, None, None, None),
                            out_axes=0)

# TODO: could we switch the entire calculation to use eV and Angstroms?
HARTREE = 27.211386245
BOHR = 0.529177211

def tscatf(IEL,LMAX,phaseshifts,e_inside,V,DR0,DRPER,DRPAR):
    """The function tscatf interpolates tabulated phase shifts and produces the atomic T-matrix elements (output in AF).
    These are also corrected for thermal vibrations (output in CAF). AF and CAF are meant to be stored in array TMAT for
    later use in RSMF, RTINV.

    IEL= chemical element to be treated now, identified by the input
    sequence order of the phase shifts (iel=1,2 or 3).
    ES= list of energies at which phase shifts are tabulated.
    PHSS= tabulated phase shifts.
    NPSI= no. of energies at which phase shifts are given.
    e_inside-V= current energy (V can be used to describe local variations
    of the muffin-tin constant).

    DR0= fourth power of RMS zero-temperature vibration amplitude.
    DRPER= RMS vibration amplitude perpendicular to surface.
    DRPAR= RMS vibration amplitude parallel to surface.
    T0= temperature at which drper and drpar have been computed.
    T= current temperature.
    TSF0, TSF, AF, CAF  see above."""
    E = e_inside - V

#   Average any anisotropy of RMS vibration amplitudes
    DR = jnp.sqrt((DRPER*DRPER+2*DRPAR*DRPAR)/3)
#   Compute temperature-dependent t-matrix elements
    t_matrix = PSTEMP(DR0, DR, E, phaseshifts, LMAX)
    return t_matrix

@partial(jit, static_argnames=('LMAX',))
def PSTEMP(DR0, DR, E, PHS, LMAX):
    """PSTEMP incorporates the thermal vibration effects in the phase shifts, through a Debye-Waller factor. Isotropic
    vibration amplitudes are assumed.

    DR0= 4th power of RMS zero-temperature vibration amplitudes.
    DR= Isotropic RMS vibration amplitude at reference temperature T0.
    T0= Arbitrary reference temperature from DR
    TEMP= Actual temperature.
    E= Current Energy (real number).
    PHS= Input phase shifts.
    DEL= Output (complex) phase shifts."""
    ALFA = jnp.sqrt(DR**4+DR0)/6
    FALFE = -4*ALFA*E
    # TODO: probably we can just skip this conditional
    # if abs(FALFE) < 0.001:
    #     for i in range(LMAX+1):
    #         DEL[i] = PHS[i]
    #     return DEL
    Z = FALFE*1j

    # TODO: @Paul choose better variable names
    BJ = bessel(Z, 2*LMAX+1)
    FL = (2*jnp.arange(2*LMAX+1) + 1)
    CS = 1j ** jnp.arange(2*LMAX+1)
    BJ = jnp.exp(FALFE) * FL * CS * BJ

    CTAB = (jnp.exp(2j*PHS)-1)*(2*jnp.arange(LMAX+1) + 1)

    SUM = jnp.einsum('jki,i,j->k', PRE_CALCULATED_CPPP[LMAX], CTAB, BJ)
    t_matrix = (SUM)/(2j)
    # SUM is the factor exp(2*i*delta) -1, t_matrix is temperature-dependent t-matrix.
    # Equation (22), page 29 in Van Hove, Tong book
    # Unlike TensErLEED, we do not convert it to a phase shift, but keep it as a t-matrix.
    # which we use going forward.
    return t_matrix


def MATEL_DWG(t_matrix_ref,t_matrix_new,e_inside,v_imag,LMAX,tensor_amps_out,tensor_amps_in,out_k_par2,
      out_k_par_3,unit_cell_area,CDISP):
    """The function MATEL_DWG evaluates the change in amplitude delwv for each of the exit beams for each of the
    displacements given the sph wave amplitudes corresponding to the incident wave ALM & for each of the time reversed
    exit beams EXLM.
    DELWV(NCSTEP,NT0): Change in amplitude due to displacement C for each displacement & for each exit beam.
    ALM(LMMAX): Sph wave amplitudes incident at the origin of the top layer due to the incident LEED beam.
    EXLM(NT0,LMMAX): As ALM but for each time reversed exit beam.
    C(3): Current displacement, C(1)= component along x into the surface. C(2),C(3) along ARB1/ARB2.
    CSTEP(3): Increment in displacement.
    NT0: Number of exit beams.
    NRATIO: Ration of area of surface unit cell of reconstructed surface to unit cell area of the unreconstructed
    surface. E.G. for P(2x2) NRATIO=4, for C(2x2) NRATIO=2."""

    k_inside = jnp.sqrt(2*e_inside-2j*v_imag+1j*EPS)

    # EXLM is for outgoing beams, so we need to swap indices m -> -m
    # to do this in the dense representation, we do the following:
    tensor_amps_out = tensor_amps_out[(DENSE_L[LMAX]+1)**2 - DENSE_L[LMAX] - DENSE_M[LMAX] -1]

#   The vector C must be expressed W.R.T. a right handed set of axes.
#   CDISP() is input W.R.T. a left handed set of axes.
    C = CDISP/BOHR
    C = C * jnp.array([1, 1, -1])


#   Evaluate DELTAT matrix for current displacement vector
    DELTAT = TMATRIX_DWG(t_matrix_ref,t_matrix_new,C, e_inside,v_imag,LMAX)


    delwv_per_atom = calcuclate_exit_beam_delta(
            tensor_amps_out, tensor_amps_in, DELTAT, k_inside, out_k_par2, out_k_par_3, unit_cell_area,
            LMAX, e_inside, v_imag
        )
    # sum over atom contributions
    delwv = delwv_per_atom.sum(axis=0)

    return delwv

@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None, None, None))  # vmap over atoms
@partial(vmap, in_axes=(1, None, None, None, 0, 0, None, None, None, None), out_axes=0)  # vmap over exit beams
def calcuclate_exit_beam_delta(tensor_amps_out, tensor_amps_in,
                               DELTAT, k_inside, out_k_par_2, out_k_par_3, unit_cell_area,
                               LMAX, E, v_imag):
    # Equation (41) from Rous, Pendry 1989
    AMAT = jnp.einsum('k,k,km,m->', MINUS_ONE_POW_M[LMAX], tensor_amps_out, DELTAT, tensor_amps_in)
    out_k_par = out_k_par_2*out_k_par_2 + out_k_par_3*out_k_par_3

    # XA is evaluated relative to the muffin tin zero i.e. it uses energy= incident electron energy + inner potential
    out_k_perp_inside = jnp.sqrt(2*E-out_k_par-2j*v_imag+1j*EPS)
    AMAT *= 1/(2*k_inside*unit_cell_area*out_k_perp_inside)
    return AMAT


#@profile
#@partial(jit, static_argnames=('LMAX',))
@partial(vmap, in_axes=(None, None, 0, None, None, None)) # vmap over atoms
def TMATRIX_DWG(t_matrix_ref, t_matrix_new, C, e_inside, v_imag, LMAX):
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
    CL = jnp.linalg.norm(C)

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

    CAPPA = 2*e_inside - 2j*v_imag
    Z = jnp.sqrt(CAPPA)*CL
    BJ = masked_bessel(Z,2*LMAX+1)
    YLM = HARMONY(C, LMAX)
    GTWOC = get_csum(BJ, YLM, LMAX, DENSE_QUANTUM_NUMBERS[LMAX])

    broadcast_New_t_matrix = map_l_array_to_compressed_quantum_index(t_matrix_new, LMAX)
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
    # but I'm not sure the conditional is worth it in terms of performance


    # Use the array versions in the vmap call
    gaunt_coeffs = fetch_lpp_gaunt(L, LP, all_lpp, M, -MP, -M+MP)
    gaunt_coeffs = gaunt_coeffs*(-1)**(LP+MP)  #TODO: @Paul: I found we need this factor, but I still don't understand why
    bessel_values = BJ[all_lpp]
    ylm_values = YLM[all_lpp*all_lpp+all_lpp+1-MPP-1]
    # Equation (34) from Rous, Pendry 1989
    csum = jnp.sum(bessel_values*ylm_values*gaunt_coeffs*1j**(L-LP-all_lpp))
    csum = csum*4*np.pi
    return csum
