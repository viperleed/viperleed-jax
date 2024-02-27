# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:31:49 2023

@author: haide
"""
import numpy as np

from src.lib_phaseshifts import *
from src.lib_tensors import *
from src.lib_tscatf import *


def _select_phaseshifts(IEL, phaseshifts):
    """Selects the phaseshifts for the given element number IEL"""
    return jax.lax.select(IEL == 0,
                          jnp.zeros_like(phaseshifts[:, 0, :]),
                          phaseshifts[:, IEL-1, :])


def delta_amplitude(IEL, LMAX, DR, tensor_dict, unit_cell_area, interpolated_phaseshifts, displacement):
    e_inside = tensor_dict['e_kin']  # computational energy inside crystal
    t_matrix_ref = tensor_dict['t_matrix']  # atomic t-matrix of current site as used in reference calculation
    v_imag = tensor_dict['v0i_substrate']# imaginary part of the inner potential, substrate

    tensor_amps_out = tensor_dict['tensor_amps_out']  # spherical wave amplitudes incident from exit beam NEXIT in "time-reversed"
    #                                       LEED experiment (or rather, all terms of Born series immediately after
    #                                       scattering on current atom)
    tensor_amps_in = tensor_dict['tensor_amps_in']  # spherical wave amplitudes incident on current atomic site in reference calculation
    # crop tensors to LMAX
    tensor_amps_out = tensor_amps_out[:, :(LMAX+1)**2, :]
    tensor_amps_in = tensor_amps_in[:, :(LMAX+1)**2] 
    #                                     (i.e., scattering path ends before scattering on that atom)
    out_k_par2, out_k_par3 = tensor_dict['kx_in'], tensor_dict['ky_in']  # (negative) absolute lateral momentum of Tensor LEED beams
    #                                                        (for use as incident beams in time-reversed LEED calculation)

    # NewCAF: working array in which current (displaced) atomic t-matrix is stored
    # TODO: we could also either append empty phaseshifts to the phaseshifts array or move the conditional around tscatf
    selected_phaseshifts = _select_phaseshifts(IEL, interpolated_phaseshifts)
    tscatf_vmap = jax.vmap(tscatf, in_axes=(None, 0, 0, None))
    t_matrix_new = tscatf_vmap(LMAX,
                              selected_phaseshifts,
                              e_inside, DR)

    # amplitude differences
    matel_dwg_vmap_energy = jax.vmap(MATEL_DWG, in_axes=(0, 0, 0, 0, None, 0, 0, 0, 0, None, None))
    d_amplitude = matel_dwg_vmap_energy(t_matrix_ref, t_matrix_new, e_inside, v_imag,
                        LMAX, tensor_amps_out, tensor_amps_in, out_k_par2, out_k_par3,
                        unit_cell_area, displacement)

    return d_amplitude
