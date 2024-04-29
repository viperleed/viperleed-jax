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

from time import time


@partial(jit, static_argnames=('LMAX','energies', 'tensors', 'unit_cell_area', 'phaseshifts',))
def delta_amplitude(LMAX, DR, energies, tensors, unit_cell_area, phaseshifts, displacements):
    if DEBUG:
        jax.debug.callback(init_time, ordered=True)
    # unpack hashable arrays
    _energies = jnp.asarray(energies.val)
    _phaseshifts = jnp.asarray(phaseshifts.val)
    # unpack tensor data
    t_matrix_ref = jnp.array([t.t_matrix for t in tensors])
    t_matrix_ref = t_matrix_ref.swapaxes(0, 1)  # swap energy and atom indices
    v_imag = tensors[0].v0i_substrate[0]

    # atom dependent quantities
    tensor_amps_out = jnp.array([t.tensor_amps_out for t in tensors])
    tensor_amps_in = jnp.array([t.tensor_amps_in for t in tensors])
    tensor_amps_out = tensor_amps_out.swapaxes(0, 1)  # swap energy and atom indices
    tensor_amps_in = tensor_amps_in.swapaxes(0, 1)  # swap energy and atom indices

    # tensor_amps_out is for outgoing beams, so we need to swap indices m -> -m
    # to do this in the dense representation, we do the following:
    tensor_amps_out = tensor_amps_out[:, :, (DENSE_L[LMAX]+1)**2 - DENSE_L[LMAX] - DENSE_M[LMAX] -1, :]

    # apply (-1)^m to tensor_amps_out - this factor is needed
    # in the calculation of the amplitude differences
    tensor_amps_out = jnp.einsum('l,ealb->ealb', MINUS_ONE_POW_M[LMAX], tensor_amps_out)

    # energy dependent quantities
    out_k_par2 = tensors[0].kx_in # same for all atoms
    out_k_par3 = tensors[0].ky_in # same for all atoms

    if DEBUG:
        jax.debug.print('Setup:', ordered=True)
        jax.debug.callback(show_time, ordered=True)

    # Calculate the t-matrix with the vibrational displacements
    tscatf_vmap = jax.vmap(apply_vibrational_displacements, in_axes=(None, 0, 0, None), out_axes=1)  # vmap over energy
    t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
    t_matrix_new = t_matrix_new.swapaxes(0, 1)  # swap energy and atom indices

    if DEBUG:
        jax.debug.print('Vib disp:', ordered=True)
        jax.debug.callback(show_time, ordered=True)

    k_inside = jnp.sqrt(2*_energies-2j*v_imag+1j*EPS)

    # Propagator evaluated relative to the muffin tin zero i.e.
    # it uses energy = incident electron energy + inner potential
    out_k_par = out_k_par2**2 + out_k_par3**2
    out_k_perp_inside = jnp.sqrt(
        ((2*_energies-2j*v_imag)[:, jnp.newaxis] - out_k_par)
        + 1j*EPS
    )

    # TODO: this could be done earlier already (even in tensor data readin)
    # Prefactors from Equation (41) from Rous, Pendry 1989
    # pulled into the tensor_amps_out array
    tensor_amps_out_with_prefactors = jnp.einsum('ealb,e,eb,->ealb',
        tensor_amps_out,
        1/k_inside,
        1/out_k_perp_inside,
        1/(2*(unit_cell_area/BOHR**2))
    )

    # Applying the geometric displacements is most efficiently handled
    # by a scan operation as a loop over the energies
    # A for loop is much slower to jit-compile and a vmap
    # results in too large intermediate arrays

    def _geo_disp_by_energy(carry, en_id):
        delta_amp = apply_geometric_displacements(
            t_matrix_ref[en_id], t_matrix_new[en_id],
            _energies[en_id],
            v_imag, LMAX,
            tensor_amps_out_with_prefactors[en_id], tensor_amps_in[en_id],
            displacements)
        return carry, delta_amp

    # Prepare the sequence to scan over.
    energy_seq = jnp.arange(len(_energies))

    if DEBUG:
        jax.debug.print('Geo setup:', ordered=True)
        jax.debug.callback(show_time, ordered=True)

    # Perform the scan 
    _, delta_amps = jax.lax.scan(_geo_disp_by_energy, None, energy_seq)

    if DEBUG:
        jax.debug.print('Geo disp:', ordered=True)
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
