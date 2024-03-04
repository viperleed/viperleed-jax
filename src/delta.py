# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:31:49 2023

@author: haide
"""
import numpy as np

from src.lib_phaseshifts import *
from src.lib_tensors import *
from src.lib_delta import *


def delta_amplitude(LMAX, DR, tensor_data, unit_cell_area, phaseshifts, displacements):
    t_matrix_ref = tensor_data.t_matrix  # atomic t-matrix of current site as used in reference calculation
    v_imag = tensor_data.v0i_substrate # imaginary part of the inner potential, substrate

    tensor_amps_out = tensor_data.tensor_amps_out  # spherical wave amplitudes incident from exit beam NEXIT in "time-reversed"
    #                                       LEED experiment (or rather, all terms of Born series immediately after
    #                                       scattering on current atom)
    tensor_amps_in = tensor_data.tensor_amps_in  # spherical wave amplitudes incident on current atomic site in reference calculation

    #                                     (i.e., scattering path ends before scattering on that atom)
    out_k_par2, out_k_par3 = tensor_data.kx_in, tensor_data.ky_in  # (negative) absolute lateral momentum of Tensor LEED beams
    #                                                        (for use as incident beams in time-reversed LEED calculation)

    # NewCAF: working array in which current (displaced) atomic t-matrix is stored
    # TODO: we could also either append empty phaseshifts to the phaseshifts array or move the conditional around tscatf
    apply_vibrational_displacements_vmap = jax.vmap(apply_vibrational_displacements, in_axes=(None, 0, 0, None))  # vmap over energy
    t_matrix_new = apply_vibrational_displacements_vmap(LMAX, phaseshifts, tensor_data.e_kin, DR)

    # amplitude differences
    apply_geometric_displacements_vmap_energy = jax.vmap(apply_geometric_displacements, in_axes=(0, 0, 0, 0, None, 0, 0, 0, 0, None, None))
    d_amplitude = apply_geometric_displacements_vmap_energy(t_matrix_ref, t_matrix_new, tensor_data.e_kin, v_imag,
                        LMAX, tensor_amps_out, tensor_amps_in, out_k_par2, out_k_par3,
                        unit_cell_area, displacements)

    return d_amplitude
