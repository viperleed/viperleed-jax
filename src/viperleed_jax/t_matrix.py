from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp

from viperleed_jax.lib_math import bessel
from viperleed_jax.constants import BOHR

from viperleed_jax.gaunt_coefficients import PRE_CALCULATED_CPPP


@jax.named_scope("vib_dependent_tmatrix")
# vmap over sites for which to calculate the t-matrix
def vib_dependent_tmatrix(LMAX, phaseshifts, e_inside, vib_amp):
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
    vibrations only. Up to and including the calculation of tmatrix_2j, every
    operation is derived from (23).

    The factor PRE_CALCULATED_CPPP is defined as 
    (4Pi/((2l+1)(2l'+1)(2*l''+1)))^0.5 * Gaunt(l,0,l',0,l'',0). The factor BJ
    is an array of Bessel functions and contains all terms dependent on l'. The
    factor CTAB includes all other terms dependent on l''.

    To compute the t-matrix, the resulting term is divided by 4ik_0 (eq. 22).
    In the code tmatrix_2j is only devided by 2i.

    Parameters
    ----------
    LMAX : int
        Maximum angular momentum quantum number.
    phaseshifts : array
        Interpolated phase shifts.
    e_inside : float
        Current energy (real number).
    vib_amp : float
        Isotropic RMS vibration amplitude (in Bohr radii).

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
    implement the isotropic case here, with a single input parameter vib_amp.

    Finally, the TensErLEED versions used allow a local variation of the of the
    muffin-tin constant, via a parameter VSITE that shifts the used energy in
    the crystal potential as E -> E - VSITE. This functionality was also not
    included as VSITE was again hardcoded to 0 in the TensErLEED code.
    """
    debye_waller_exponent = -2/3 * (vib_amp/BOHR)**2 * e_inside

    all_l = (2*jnp.arange(2*LMAX+1) + 1)
    bessel_with_prefactor = (
        jnp.exp(debye_waller_exponent)
        * all_l
        * 1j ** jnp.arange(2*LMAX+1)
        * bessel(debye_waller_exponent * 1j, 2*LMAX)
    )

    temperature_independent_t_matrix = (
        jnp.exp(2j*phaseshifts)-1)*(2*jnp.arange(LMAX+1) + 1)

    t_matrix_2j = jnp.einsum(
        'jki,i,j->k',
        PRE_CALCULATED_CPPP[LMAX],  # about 3/4 of these are zero. We could skip them
        temperature_independent_t_matrix,
        bessel_with_prefactor
    )
    t_matrix = (t_matrix_2j)/(2j) # temperature-dependent t-matrix.
    # t_matrix_2j is the factor exp(2*i*delta) - 1
    # Equation (22), page 29 in Van Hove, Tong book from 1979
    # Unlike TensErLEED, we do not convert it to a phase shift, but keep it as a
    # t-matrix, which we use going forward.
    return t_matrix


# vmap over sites for which to calculate the t-matrix
vmap_vib_dependent_tmatrix = jax.vmap(vib_dependent_tmatrix,
                                      in_axes=(None, 1, None, 0))
