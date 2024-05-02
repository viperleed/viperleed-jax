# -*- coding: utf-8 -*-
"""

@author: Paul Haidegger, Alexander M. Imre
"""
import numpy as np

from src.constants import HARTREE, BOHR
from src.lib_phaseshifts import *
from src.lib_tensors import *
from src.lib_delta import *

DEBUG = True

BATCH = False # TODO: try later (also variable batch sizes)

from time import time


#@partial(jit, static_argnames=('LMAX','energies', 'tensors', 'unit_cell_area', 'phaseshifts',))
def delta_amplitude(LMAX, ref_data, DR, energies, tensors, unit_cell_area, phaseshifts, displacements):
    if DEBUG:
        jax.debug.callback(init_time, ordered=True)
    # unpack hashable arrays
    _energies = jnp.asarray(ref_data.energies)
    _phaseshifts = jnp.asarray(phaseshifts.val)
    # unpack tensor data
    v_imag = ref_data.v0i

    # energy dependent quantities
    out_k_par2 = ref_data.kx_in
    out_k_par3 = ref_data.ky_in

    k_inside = jnp.sqrt(2*_energies-2j*v_imag+1j*EPS)

    # Propagator evaluated relative to the muffin tin zero i.e.
    # it uses energy = incident electron energy + inner potential
    out_k_par = out_k_par2**2 + out_k_par3**2
    out_k_perp_inside = jnp.sqrt(
        ((2*_energies-2j*v_imag)[:, jnp.newaxis] - out_k_par)
        + 1j*EPS
    )

    # Prefactors from Equation (41) from Rous, Pendry 1989
    prefactors = jnp.einsum('e,eb,->eb',
        1/k_inside,
        1/out_k_perp_inside,
        1/(2*(unit_cell_area/BOHR**2))
    )

    if DEBUG:
        jax.debug.print('Setup:', ordered=True)
        jax.debug.callback(show_time, ordered=True)


    delta_amps_by_lmax = []
    energy_ids_by_lmax = []

    for lmax in ref_data.needed_lmax:
        print(f'compiling for lmax {lmax}')
        #energy_ids = ref_data.energy_ids_for_lmax(lmax)  # TODO: reuse this one ref_data is a proper pytree
        energy_ids = ref_data.energy_ids_for_lmax[lmax]

        # select the relevant data for the current lmax
        l_energies = _energies[energy_ids]
        l_phaseshifts = _phaseshifts[energy_ids, :, :lmax+1]
        l_t_matrix_ref = ref_data.ref_t_matrix[lmax]
        l_tensor_amps_in = ref_data.tensor_amps_in[lmax]
        l_tensor_amps_out = ref_data.tensor_amps_out[lmax]

        sequential_ids = jnp.arange(len(energy_ids))

        if BATCH:
            t_matrix_new = jax.vmap(
                apply_vibrational_displacements,
                in_axes=(None, 0, 0, None))(lmax, l_phaseshifts, l_energies, DR)
        else:
            def _vib_disp_by_energy(id):
                return apply_vibrational_displacements(
                    lmax,
                    l_phaseshifts[id, ...],
                    l_energies[id],
                    jnp.asarray(DR))

            # Calculate the t-matrix with the vibrational displacements
            t_matrix_new = jax.lax.map(_vib_disp_by_energy, sequential_ids)

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
        delta_amps_by_lmax.append(delta_amps)
        energy_ids_by_lmax.append(energy_ids)

    # now re-sort the delta_amps to the original order
    delta_amps = jnp.concatenate(delta_amps_by_lmax, axis=0)
    delta_amps = delta_amps[ref_data.energy_sorting]

    # Finally apply the prefactors calculated earlier to the result (element)
    delta_amps = prefactors * delta_amps

    if DEBUG:
        jax.debug.print('Displacements:', ordered=True)
        jax.debug.callback(show_time, ordered=True)
        jax.debug.print("hello {bar}", bar=jnp.sum(abs(delta_amps)), ordered=True)
        jax.debug.callback(show_time, ordered=True)

    # The result is already a JAX array, so there's no need to call jnp.array on delta_amps.
    return delta_amps

last_time = time()

def init_time():
    global last_time
    last_time = time()
    print(f'Start: {last_time}')

def show_time():
    global last_time
    now = time()
    print(f'{(now - last_time):3f} s')
    last_time = now
