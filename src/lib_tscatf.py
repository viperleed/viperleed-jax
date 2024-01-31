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

from gaunt_coefficients import fetch_stored_gaunt_coeffs as fetch_gaunt
from gaunt_coefficients import cppp

fetch_lpp_gaunt = jax.vmap(fetch_gaunt,
                            in_axes=(None, None, 0, None, None, None),
                            out_axes=0)

MEMACH = 1.0E-6
HARTREE = 27.211396
BOHR = 0.529177

# vectorize cppp in all arguments
vectorized_cppp = jax.vmap(jax.vmap(jax.vmap(cppp, (None, None, 0)), (None, 0, None)), (0, None, None))

PRE_CALCULATED_CPPP = vectorized_cppp(jnp.arange(0, 2*LMAX+1), jnp.arange(0, LMAX+1), jnp.arange(0, LMAX+1))

@profile
def tscatf(IEL,LMAX,phaseshifts,EB,V,DR0,DRPER,DRPAR,T0,T):
    """The function tscatf interpolates tabulated phase shifts and produces the atomic T-matrix elements (output in AF).
    These are also corrected for thermal vibrations (output in CAF). AF and CAF are meant to be stored in array TMAT for
    later use in RSMF, RTINV.

    IEL= chemical element to be treated now, identified by the input
    sequence order of the phase shifts (iel=1,2 or 3).
    ES= list of energies at which phase shifts are tabulated.
    PHSS= tabulated phase shifts.
    NPSI= no. of energies at which phase shifts are given.
    EB-V= current energy (V can be used to describe local variations
    of the muffin-tin constant).

    DR0= fourth power of RMS zero-temperature vibration amplitude.
    DRPER= RMS vibration amplitude perpendicular to surface.
    DRPAR= RMS vibration amplitude parallel to surface.
    T0= temperature at which drper and drpar have been computed.
    T= current temperature.
    TSF0, TSF, AF, CAF  see above."""
    E = EB - V
    if E < phaseshifts[0][0]:
        print('TOO LOW ENERGY FOR AVAILABLE PHASE SHIFTS')
#   Find set of phase shifts appropriate to desired chemical element and interpolate linearly to current energy
#   (or extrapolate to energies above the range given for the phase shifts)

    for i in range(len(phaseshifts)-1):
        if (E - phaseshifts[i][0]) * (E - phaseshifts[i+1][0]) <= 0:
            break
    PHS = np.full((LMAX + 1,), dtype=np.float64, fill_value=np.nan)
    AF = np.full((LMAX + 1,), dtype=np.complex128, fill_value=np.nan)
    CAF = np.full((LMAX + 1,), dtype=np.complex128, fill_value=np.nan)
    FAC = (E - phaseshifts[i][0]) / (phaseshifts[i+1][0] - phaseshifts[i][0])
    for l in range(LMAX + 1):
        PHS[l] = phaseshifts[i][1][IEL-1][l] + FAC * (phaseshifts[i+1][1][IEL-1][l] - phaseshifts[i][1][IEL-1][l])
#       Compute temperature-independent t-matrix elements
        AF[l] = np.sin(PHS[l])*np.exp(PHS[l]*1.0j)
#   Average any anisotropy of RMS vibration amplitudes
    DR = np.sqrt((DRPER*DRPER+2*DRPAR*DRPAR)/3)
#   Compute temperature-dependent t-matrix elements
    t_matrix = PSTEMP(DR0, DR, T0, T, E, PHS)
    return t_matrix

#@jit
def PSTEMP(DR0, DR, T0, TEMP, E, PHS):
    """PSTEMP incorporates the thermal vibration effects in the phase shifts, through a Debye-Waller factor. Isotropic
    vibration amplitudes are assumed.

    DR0= 4th power of RMS zero-temperature vibration amplitudes.
    DR= Isotropic RMS vibration amplitude at reference temperature T0.
    T0= Arbitrary reference temperature from DR
    TEMP= Actual temperature.
    E= Current Energy (real number).
    PHS= Input phase shifts.
    DEL= Output (complex) phase shifts."""
    ALFA = DR*DR*TEMP/T0
    ALFA = 0.166667*jnp.sqrt(ALFA*ALFA+DR0)
    FALFE = -4.0*ALFA*E
    # TODO: probably we can just skip this conditional
    if abs(FALFE) < 0.001:
        for i in range(LMAX+1):
            DEL[i] = PHS[i]
        return DEL
    Z = FALFE*1.0j

    # TODO: @Paul choose better variable names
    BJ = bessel(Z, 2*LMAX+1)
    FL = (2*jnp.arange(2*LMAX+1) + 1)
    CS = 1.0j ** jnp.arange(2*LMAX+1)
    BJ = jnp.exp(FALFE) * FL * CS * BJ

    CTAB = (jnp.exp(2.0j*PHS)-1)*(2*jnp.arange(LMAX+1) + 1)

    SUM = jnp.einsum('jki,i,j->k', PRE_CALCULATED_CPPP, CTAB, BJ)
    t_matrix = (SUM)/(2j)
    # SUM is the factor exp(2*i*delta) -1, t_matrix is temperature-dependent t-matrix.
    # Equation (22), page 29 in Van Hove, Tong book
    # Unlike TensErLEED, we do not convert it to a phase shift, but keep it as a t-matrix.
    # which we use going forward.
    return t_matrix


@profile
def MATEL_DWG(NCSTEP,AF,NewAF,E,VV,VPI,LMAX,n_beams,EXLM,ALM,AK2M,
      AK3M,NRATIO,TV,n_atoms,CDISP,PSQ):
    """The function MATEL_DWG evaluates the change in amplitude delwv for each of the exit beams for each of the
    displacements given the sph wave amplitudes corresponding to the incident wave ALM & for each of the time reversed
    exit beams EXLM.
    DELWV(NCSTEP,NT0): Change in amplitude due to displacement C for each displacement & for each exit beam.
    ALM(LMMAX): Sph wave amplitudes incident at the origin of the top layer due to the incident LEED beam.
    EXLM(NT0,LMMAX): As ALM but for each time reversed exit beam.
    C(3): Current displacement, C(1)= component along x into the surface. C(2),C(3) along ARB1/ARB2.
    CSTEP(3): Increment in displacement.
    NCSTEP: Number of displacements. (all displacements in Angstroms)
    NT0: Number of exit beams.
    NRATIO: Ration of area of surface unit cell of reconstructed surface to unit cell area of the unreconstructed
    surface. E.G. for P(2x2) NRATIO=4, for C(2x2) NRATIO=2."""
#   Set teh change in amplitudes to zero for each exit beam.
    DELWV = np.full((NCSTEP, n_beams), dtype=np.complex128, fill_value=0)

    # Dense quantum number indexing
    dense_quantum_numbers_2lmax = DENSE_QUANTUM_NUMBERS_2LMAX
    dense_m_2lmax = dense_quantum_numbers_2lmax[:,0,2]
    dense_l_2lmax = dense_quantum_numbers_2lmax[:,0,0]

    dense_quantum_numbers = DENSE_QUANTUM_NUMBERS
    dense_m = dense_quantum_numbers[:,0,2]
    dense_l = dense_quantum_numbers[:,0,0]
    minus_1_pow_m = jnp.power(-1, dense_m)  # (-1)**M

#   Loop over model structure
    for NC in range(1, NCSTEP+1):
#       Loop over the atoms of the reconstructed unit cell
        for NR in range(1, n_atoms+1):
            C = CDISP[NC-1, NR-1, :]/BOHR
#           The vector C must be expressed W.R.T. a right handed set of axes. CDISP() & CUNDISP() are input W.R.T.
#           a left handed set of axes
            C[2] = -C[2]
#           Evaluate DELTAT matrix for current displacement.
            DELTAT = TMATRIX_DWG(AF,NewAF,C, E,VPI,LMAX, dense_quantum_numbers, dense_l, dense_m, dense_l_2lmax, dense_m_2lmax)

            for NEXIT in range(1,n_beams): #Loop over exit beams
#               Evaluate matrix element
                EMERGE = 2*(E-VV)-AK2M[NEXIT-1]**2-AK3M[NEXIT-1]**2
                # TODO: should we get rid of this conditional?
                # If the beam does not emerge, AMAT is going to be 0 anyway
                if EMERGE >= 0:
                    _EXLM = EXLM[:(LMAX+1)**2,NEXIT-1]                           # TODO: crop EXLM, ALM earlier
                    _ALM = ALM[:(LMAX+1)**2]

                    # EXLM is for outgoing beams, so we need to swap indices m -> -m
                    # to do this in the dense representation, we do the following:
                    _EXLM = _EXLM[(dense_l+1)**2 - dense_l - dense_m -1]

                    # Equation (41) from Rous, Pendry 1989
                    AMAT = jnp.einsum('k,k,km,m->', minus_1_pow_m, _EXLM, DELTAT, _ALM)

#                   Evaluate prefactor                                          # TODO: @Paul: check with Tobis thesis and adjust variable names, and comments
                    D2 = AK2M[NEXIT-1]
                    D3 = AK3M[NEXIT-1]
                    D = D2*D2 + D3*D3
                    CAK = 2*E-2j*VPI+0.0000001j
                    CAK = np.sqrt(CAK)
                    if D >= 2*E:
                        print('Something went wrong in MATEL_DWG') # @Paul: can we remove this?
                        return 0
#                   XA is evaluated relative to the muffin tin zero i.e. it uses energy= incident electron energy +
#                   inner potential
                    XA = 2*E-D-2j*VPI+0.0000001j
                    XA = np.sqrt(XA)
                    AMAT *= 1/(2*CAK*TV*XA*NRATIO)
                    DELWV[NC-1][NEXIT-1] += AMAT
    return DELWV

#@profile
#@partial(jit, static_argnames=('LMAX',))
def TMATRIX_DWG(AF, NewAF, C, E, VPI, LMAX, dense_quantum_numbers, dense_l, dense_m, dense_l_2lmax, dense_m_2lmax):
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
    CL = jnp.sqrt(C[0]*C[0] + C[1]*C[1] + C[2]*C[2])

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

    CAPPA = 2*E - 2j*VPI
    Z = jnp.sqrt(CAPPA)*CL
    BJ = jnp.nan_to_num(bessel(Z,2*LMAX+1))
    YLM = HARMONY(C, LMAX, dense_l_2lmax, dense_m_2lmax)
    GTWOC = sum_quantum_numbers(LMAX, BJ, YLM, dense_quantum_numbers)

    broadcast_New_AF = map_l_array_to_compressed_quantum_index(NewAF, LMAX, dense_l)
    GTEMP = GTWOC.T*1.0j*broadcast_New_AF

    DELTAT = jax.numpy.einsum('il,lj->ij', GTEMP, GTWOC)

    diagonal_indices = jnp.diag_indices_from(DELTAT)
    mapped_AF = map_l_array_to_compressed_quantum_index(AF, LMAX, dense_l)
    DELTAT = DELTAT.at[diagonal_indices].add(-1.0j*mapped_AF)

    return DELTAT


# TODO: better name
@partial(jit, static_argnames=('LMAX',))
def sum_quantum_numbers(LMAX, BJ, YLM, dense_quantum_numbers):
    propagator_origin_to_c = jax.vmap(jax.vmap(
        get_csum,
        in_axes=(None, None, None, 0),
    ), in_axes=(None, None, None, 0),
    )(BJ, YLM, LMAX, dense_quantum_numbers)
    return propagator_origin_to_c  # named GTWOC in fortran code


@partial(jit, static_argnames=('LMAX',))
def get_csum(BJ, YLM, LMAX, l_lp_m_mp):
    L, LP, M, MP = l_lp_m_mp
    MPP = MP-M  # I don't fully understand this, technically it should be MPP = -M - MP
    all_lpp = jnp.arange(0, LMAX*2+1)
    # we could skip some computations with non_zero_lpp = jnp.where((all_lpp >= abs(L-LP)) & (all_lpp <= L+LP))
    # but I'm not sure the conditional is worth it in terms of performance
    # Convert scalar values to arrays
    L_array = jnp.array(L)
    LP_array = jnp.array(LP)
    M_array = jnp.array(M)
    MP_array = jnp.array(MP)

    # Use the array versions in the vmap call
    gaunt_coeffs = fetch_lpp_gaunt(L_array, LP_array, all_lpp, M_array, -MP_array, -M_array+MP_array)
    gaunt_coeffs = gaunt_coeffs*(-1)**(LP+MP)  #TODO: @Paul: I found we need this factor, but I still don't understand why
    bessel_values = BJ[all_lpp]
    ylm_values = YLM[all_lpp*all_lpp+all_lpp+1-MPP-1]
    # Equation (34) from Rous, Pendry 1989
    csum = jnp.sum(bessel_values*ylm_values*gaunt_coeffs*1.0j**(L-LP-all_lpp))
    csum = csum*4*np.pi
    return csum


# TODO: come up with a jittable version of this
def get_valid_quantum_numbers(LMAX):
    valid_quantum_numbers = np.empty(((LMAX+1)*(LMAX+1), (LMAX+1)*(LMAX+1), 4), dtype=int)
    for L in range(LMAX+1):
        for LP in range(LMAX+1):
            for M in range(-L, L+1):
                for MP in range(-LP, LP+1):
                    valid_quantum_numbers[(L+1)*(L+1)-L+M-1][(LP+1)*(LP+1)-LP+MP-1] = [L, LP, M, MP]
    return jnp.array(valid_quantum_numbers)


def map_l_array_to_compressed_quantum_index(array, LMAX, broadcast_l_index):
    """Takes an array of shape (LMAX+1) with values for each L and maps it to a
    dense form of shape (LMAX+1)*(LMAX+1) with the values for each L replicated
    (2L+1) times. I.e. an array
    [val(l=0), val(l=1), val(l=2), ...] is mapped to
    [val(l=0), val(l=1), val(l=1), val(l=1), val(l=2), ...].
    """
    mapped_array = jnp.asarray(array)[broadcast_l_index]
    return mapped_array

LMAX = 14
DENSE_QUANTUM_NUMBERS = get_valid_quantum_numbers(LMAX)
DENSE_QUANTUM_NUMBERS_2LMAX = get_valid_quantum_numbers(2*LMAX)
