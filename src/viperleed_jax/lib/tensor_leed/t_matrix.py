"""Module t_matrix."""

__authors__ = ('Alexander M. Imre (@amimre)', 'Paul Haidegger (@Paulhai7)')
__created__ = '2024-08-14'

from functools import partial

import jax
import jax.numpy as jnp

from viperleed_jax.constants import BOHR
from viperleed_jax.gaunt_coefficients import PRE_CALCULATED_CPPP
from viperleed_jax.lib.math import bessel


# vmap over sites for which to calculate the t-matrix
# @partial(jax.profiler.annotate_function, name="vib_dependent_tmatrix")
def vib_dependent_tmatrix(l_max, phaseshifts, e_inside, vib_amp):
    """Compute the temperature-dependent t-matrix elements.

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
    l_max : int
        Maximum angular momentum quantum number.
    phaseshifts : array, shape (l_max+1,)
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
    debye_waller_exponent = -2 / 3 * (vib_amp / BOHR) ** 2 * e_inside

    all_l = 2 * jnp.arange(2 * l_max + 1) + 1
    bessel_with_prefactor = (
        jnp.exp(debye_waller_exponent)
        * all_l
        * 1j ** jnp.arange(2 * l_max + 1)
        * bessel(debye_waller_exponent * 1j, 2 * l_max)
    )

    temperature_independent_t_matrix = (jnp.exp(2j * phaseshifts) - 1) * (
        2 * jnp.arange(l_max + 1) + 1
    )

    t_matrix_2j = jnp.einsum(
        'jki,i,j->k',
        PRE_CALCULATED_CPPP[
            l_max
        ],  # about 3/4 of these are zero. We could skip them
        temperature_independent_t_matrix,
        bessel_with_prefactor,
    )
    t_matrix = (t_matrix_2j) / (2j)  # temperature-dependent t-matrix.
    # t_matrix_2j is the factor exp(2*i*delta) - 1
    # Equation (22), page 29 in Van Hove, Tong book from 1979
    # Unlike TensErLEED, we do not convert it to a phase shift, but keep it as a
    # t-matrix, which we use going forward.
    return t_matrix


# vmap over sites for which to calculate the t-matrix
vmap_vib_dependent_tmatrix = jax.vmap(
    vib_dependent_tmatrix, in_axes=(None, 1, None, 0)
)

# vmap over energies
vmap_energy_vib_dependent_tmatrix = jax.vmap(
    vib_dependent_tmatrix, in_axes=(None, 0, 0, None), out_axes=0
)


def _calculate_dynamic_t_matrices(
    l_max,
    batch_energies,
    dynamic_t_matrix_site_elements,
    phaseshifts,
    energies,
    vib_amps,
    energy_indices,
):
    # Convert energy_indices to a JAX array for the outer mapping.
    energy_indices = jnp.array(energy_indices)
    # Pre-build the static list of (vib_amp, site_element) pairs.
    pairs = list(zip(vib_amps, dynamic_t_matrix_site_elements))

    def energy_map_fn(e_idx):
        # For each energy index, loop over the static pairs.
        results = []
        for vib_amp, site_el in pairs:
            result = vib_dependent_tmatrix(
                l_max,
                phaseshifts[site_el][e_idx, : l_max + 1],
                energies[e_idx],
                vib_amp.reshape(),  # reshape from (1,) to scalar for grad compatibility
            )
            results.append(result)
        return jnp.stack(results)

    dynamic_t_matrices = jax.lax.map(
        energy_map_fn, energy_indices, batch_size=batch_energies
    )
    return jnp.asarray(dynamic_t_matrices)


@partial(
    jax.jit,
    static_argnames=[
        'l_max',
        'batch_energies',
        'phaseshifts',
    ],
)
def calculate_t_matrices(
    t_matrix_context,
    l_max,
    batch_energies,
    phaseshifts,
    dynamic_vib_amps,
    energy_indices,
):
    # Process one energy at a time to reduce memory usage.
    energy_indices = jnp.array(energy_indices)

    def energy_fn(e_idx):
        # Compute the dynamic t-matrix for a single energy.
        # _calculate_dynamic_t_matrices expects a sequence of energies; here we pass a list of one index.
        if len(t_matrix_context.dynamic_t_matrices) > 0:
            dyn_t = _calculate_dynamic_t_matrices(
                l_max,
                batch_energies,
                t_matrix_context.dynamic_site_elements,
                phaseshifts,
                t_matrix_context.energies,
                dynamic_vib_amps,
                [e_idx],
            )[0]
            # Map the dynamic t-matrix to the atom-site-element basis.
            dyn_mapped = dyn_t[t_matrix_context.t_matrix_id]
        else:
            # If no dynamic t-matrices are available, return an empty array.
            dyn_mapped = jnp.zeros_like(t_matrix_context.static_t_matrices)
        # Get the corresponding static t-matrix, or zeros if none exist.
        if len(t_matrix_context.static_t_matrices) == 0:
            stat_t = jnp.zeros_like(dyn_t)
        else:
            stat_t = t_matrix_context.static_t_matrices[e_idx, :, :]
        stat_mapped = stat_t[t_matrix_context.t_matrix_id, :]
        # Select between dynamic and static for this energy.
        # The condition is broadcasted to shape (num_selected, lm)
        return jnp.where(
            t_matrix_context.is_dynamic_mask[:, jnp.newaxis],
            dyn_mapped,
            stat_mapped,
        )

    # Process each energy one by one.
    return jax.lax.map(energy_fn, energy_indices, batch_size=batch_energies)
