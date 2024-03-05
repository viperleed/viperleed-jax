# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:31:49 2023

@author: haide
"""
import numpy as np

from src.lib_phaseshifts import *
from src.lib_tensors import *
from src.lib_delta import *


def delta_amplitude(LMAX, DR, energies, tensors, unit_cell_area, phaseshifts, displacements):
    # unpack tensor data
    t_matrix_ref = jnp.array([t.t_matrix for t in tensors])
    v_imag = jnp.array([t.v0i_substrate for t in tensors])
    tensor_amps_out = jnp.array([t.tensor_amps_out for t in tensors])
    tensor_amps_in = jnp.array([t.tensor_amps_in for t in tensors])
    out_k_par2 = jnp.array([t.kx_in for t in tensors])
    out_k_par3 = jnp.array([t.ky_in for t in tensors])

    tscatf_vmap = jax.vmap(apply_vibrational_displacements, in_axes=(None, 0, 0, None), out_axes=1)  # vmap over energy
    t_matrix_new = tscatf_vmap(LMAX, phaseshifts, energies, DR)

    # amplitude differences
    matel_dwg_vmap_energy = jax.vmap(apply_geometric_displacements, in_axes=(1, 1, 0, 1, None, 1, 1, 1, 1, None, None))
    d_amplitude = matel_dwg_vmap_energy(t_matrix_ref, t_matrix_new, energies, v_imag,
                        LMAX, tensor_amps_out, tensor_amps_in, out_k_par2, out_k_par3,
                        unit_cell_area, displacements)

    return d_amplitude
