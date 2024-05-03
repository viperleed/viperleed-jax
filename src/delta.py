# -*- coding: utf-8 -*-
"""

@author: Paul Haidegger, Alexander M. Imre
"""
import jax
from jax import jit
from jax import numpy as jnp
from functools import partial

from src.constants import BOHR
from src.lib_delta import apply_vibrational_displacements
from src.lib_delta import apply_geometric_displacements
from src.lib_math import EPS


@partial(jit, static_argnames=('ref_data', 'unit_cell_area', 'phaseshifts',
                               'batch_lmax'))
def delta_amplitude(vib_amps, displacements, ref_data, unit_cell_area, phaseshifts,
                    batch_lmax=False):

    # unpack hashable arrays
    energies = ref_data.energies
    # unpack tensor data
    v_imag = ref_data.v0i

    # energy dependent quantities
    out_k_par2 = ref_data.kx_in
    out_k_par3 = ref_data.ky_in

    k_inside = jnp.sqrt(2*energies-2j*v_imag+1j*EPS)

    # Propagator evaluated relative to the muffin tin zero i.e.
    # it uses energy = incident electron energy + inner potential
    out_k_par = out_k_par2**2 + out_k_par3**2
    out_k_perp_inside = jnp.sqrt(
        ((2*energies-2j*v_imag)[:, jnp.newaxis] - out_k_par)
        + 1j*EPS
    )

    # Prefactors from Equation (41) from Rous, Pendry 1989
    prefactors = jnp.einsum('e,eb,->eb',
        1/k_inside,
        1/out_k_perp_inside,
        1/(2*(unit_cell_area/BOHR**2))
    )

    delta_amps_by_lmax = []

    for lmax in ref_data.needed_lmax:
        print(f'compiling for lmax {lmax}')
        #energy_ids = ref_data.energy_ids_for_lmax(lmax)  # TODO: reuse this one ref_data is a proper pytree
        energy_ids = ref_data.energy_ids_for_lmax[lmax]

        # select the relevant data for the current lmax
        l_energies = energies[energy_ids]
        l_phaseshifts = phaseshifts.phaseshifts[energy_ids, :, :lmax+1]
        l_t_matrix_ref = ref_data.ref_t_matrix[lmax]
        l_tensor_amps_in = ref_data.tensor_amps_in[lmax]
        l_tensor_amps_out = ref_data.tensor_amps_out[lmax]

        sequential_ids = jnp.arange(len(energy_ids))

        if batch_lmax:
            t_matrix_new = jax.vmap(
                apply_vibrational_displacements,
                in_axes=(None, 0, 0, None))(lmax, l_phaseshifts,
                                            l_energies, jnp.asarray(vib_amps))
        else:
            def _vib_disp_by_energy(id):
                return apply_vibrational_displacements(
                    lmax,
                    l_phaseshifts[id, ...],
                    l_energies[id],
                    jnp.asarray(vib_amps))

            # Calculate the t-matrix with the vibrational displacements
            t_matrix_new = jax.lax.map(_vib_disp_by_energy, sequential_ids)

        if batch_lmax:
            delta_amps = jax.vmap(
                apply_geometric_displacements,
                in_axes=(0, 0, 0, None, None, 0, 0, None),
                )(l_t_matrix_ref,
                  t_matrix_new,
                  l_energies,
                  v_imag,
                  lmax,
                  l_tensor_amps_out,
                  l_tensor_amps_in,
                  jnp.asarray(displacements)
            )
        else:

            # Applying the geometric displacements is most efficiently handled
            # by a map operation as a loop over the energies
            # A for loop is much slower to jit-compile and a vmap
            # results in too large intermediate arrays

            def _geo_disp_by_energy(id):
                return apply_geometric_displacements(
                    l_t_matrix_ref[id],
                    t_matrix_new[id],
                    l_energies[id],
                    v_imag,
                    lmax,
                    l_tensor_amps_out[id],
                    l_tensor_amps_in[id],
                    jnp.asarray(displacements))

            # Perform the map
            delta_amps = jax.lax.map(_geo_disp_by_energy, sequential_ids)

        # add to list for later concatenation
        delta_amps_by_lmax.append(delta_amps)

    # now re-sort the delta_amps to the original order
    delta_amps = jnp.concatenate(delta_amps_by_lmax, axis=0)
    delta_amps = delta_amps[ref_data.energy_sorting]

    # Finally apply the prefactors calculated earlier to the result
    delta_amps = prefactors * delta_amps


    # The result is already a JAX array, so there's no need to call jnp.array on delta_amps.
    return delta_amps
