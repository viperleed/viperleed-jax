# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:31:49 2023

@author: haide
"""
import numpy as np

from src.lib_phaseshifts import *
from src.lib_tensors import *
from src.lib_delta import *


@partial(jit, static_argnames=('LMAX','energies', 'tensors', 'unit_cell_area', 'phaseshifts',))
BOHR = 0.529177211

def delta_amplitude(LMAX, DR, energies, tensors, unit_cell_area, phaseshifts, displacements):
    # unpack hashable arrays
    _energies = energies.val
    _phaseshifts = phaseshifts.val
    # unpack tensor data
    t_matrix_ref = jnp.array([t.t_matrix for t in tensors])
    t_matrix_ref = t_matrix_ref.swapaxes(0, 1)  # swap energy and atom indices
    v_imag = tensors[0].v0i_substrate[0]

    # atom dependent quantities
    tensor_amps_out = jnp.array([t.tensor_amps_out for t in tensors])
    tensor_amps_in = jnp.array([t.tensor_amps_in for t in tensors])
    tensor_amps_out = tensor_amps_out.swapaxes(0, 1)  # swap energy and beam indices
    tensor_amps_in = tensor_amps_in.swapaxes(0, 1)  # swap energy and beam indices

    # tensor_amps_out is for outgoing beams, so we need to swap indices m -> -m
    # to do this in the dense representation, we do the following:
    tensor_amps_out = tensor_amps_out[:, :, (DENSE_L[LMAX]+1)**2 - DENSE_L[LMAX] - DENSE_M[LMAX] -1, :]

    # apply (-1)^m to tensor_amps_out - this factor is needed
    # in the calculation of the amplitude differences
    tensor_amps_out = jnp.einsum('l,ealb->ealb', MINUS_ONE_POW_M[LMAX], tensor_amps_out)

    # energy dependent quantities
    out_k_par2 = tensors[0].kx_in # same for all atoms
    out_k_par3 = tensors[0].ky_in # same for all atoms

    # Calculate the t-matrix with the vibrational displacements
    tscatf_vmap = jax.vmap(apply_vibrational_displacements, in_axes=(None, 0, 0, None), out_axes=1)  # vmap over energy
    t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
    t_matrix_new = t_matrix_new.swapaxes(0, 1)  # swap energy and atom indices

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

    # amplitude differences
    matel_dwg_vmap_energy = jax.vmap(apply_geometric_displacements, in_axes=(0, 0, 0, None, None, 0, 0, None))
    d_amplitude = matel_dwg_vmap_energy(t_matrix_ref, t_matrix_new, _energies, v_imag,
                        LMAX, tensor_amps_out_with_prefactors, tensor_amps_in,
                        displacements)

    return d_amplitude
