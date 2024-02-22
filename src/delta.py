# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:31:49 2023

@author: haide
"""
import numpy as np

from lib_phaseshifts import *
from lib_tensors import *
from lib_tscatf import *

#From "PARAM"
LMAX = 14  # maximum angular momentum to be used in calculation
n_beams = 9  # no. of TLEED output beams
n_atoms = 1  # currently 1 is the only possible choice
n_geo = 1  # number of geometric variations ('displacements') to be considered

# From Stdin
# DR0,DRPER,DRPAR: thermal vibration amplitude steps to be included in
# current variation - DR0 is always 0., DRPER = DRPAR forced
DR0 = 0
DRPER = 0.1908624
DRPAR = DRPER

CDISP = np.full((n_geo, n_atoms, 3),dtype=np.float64,fill_value=np.nan)  # displaced positions of current atomic site for variation
for i in range(n_geo):
    CDISP[i][0][0] = -0.01*i + 0.05
    CDISP[i][0][1] = 0
    CDISP[i][0][2] = 0
print(CDISP)

# unit vectors in Angstrom
u_vec1 = np.array([1.2722, -2.2036])
u_vec2 = np.array([1.2722,  2.2036])

# area of (overlayer) lateral unit cell - in case TLEED wrt smaller unit cell is used, TVA from reference computation must be set.
unit_cell_area = np.linalg.norm(np.cross(u_vec1, u_vec2))


IEL = 1  # element no. (in phase shifts supplied with input) that delta amplitudes
#          will be calculated for (not necessarily the same element as the one
#          used in the reference calculation!) - IEL = 0 means a vacancy will be assumed

VSITE = 0  # possible energy shift in phase shift computations - can be used to describe
#            local variations of the muffin-tin-constant

@profile
def main():

    _, phaseshifts, _, _ = readPHASESHIFTS(None, None, readfile='PHASESHIFTS', check=False, ignoreEnRange=False)

    n_energies = 0
    with open('T_1', 'r') as datei:
        for zeile in datei:
            if '-1' in zeile:
                n_energies += 1

    my_dict = read_tensor('T_1', n_beams=9, n_energies= n_energies, l_max=LMAX+1)

    # TODO: raise Error if requested energies are out of range respective to
    # phaseshift energies (can't interpolate if out of range)

    n_energies = 51
    energies = np.array([my_dict['e_kin'][i] for i in range(n_energies)])
    interpolated_phaseshifts = interpolate_phaseshifts(phaseshifts, LMAX, energies)

    all_delwv = np.full((n_energies, n_geo, n_beams), dtype=np.complex128, fill_value=np.nan)
    for i in range(n_energies):
        e_inside = my_dict['e_kin'][i]  # computational energy inside crystal
        t_matrix_ref = my_dict['t_matrix'][i]  # atomic t-matrix of current site as used in reference calculation
        VV = my_dict['v0r'][i]  # real part of the inner potential
        v_imag = my_dict['v0i_substrate'][i]  # imaginary part of the inner potential, substrate, resp.
        tensor_amps_out = my_dict['tensor_amps_out'][i]  # spherical wave amplitudes incident from exit beam NEXIT in "time-reversed"
        #                                       LEED experiment (or rather, all terms of Born series immediately after
        #                                       scattering on current atom)
        tensor_amps_in = my_dict['tensor_amps_in'][i]  # spherical wave amplitudes incident on current atomic site in reference calculation
        # crop to LMAX
        tensor_amps_out = tensor_amps_out[:(LMAX+1)**2, :]
        tensor_amps_in = tensor_amps_in[:(LMAX+1)**2] 
        #                                     (i.e., scattering path ends before scattering on that atom)
        out_k_par2, out_k_par3 = my_dict['kx_in'][i], my_dict['ky_in'][i]  # (negative) absolute lateral momentum of Tensor LEED beams
        #                                                        (for use as incident beams in time-reversed LEED calculation)
        PSQ = my_dict['k_delta'][i]  # lateral momentum of Tensor LEED beams relative to incident beam (0,0)

        EEV = (e_inside-VV)*HARTREE  # current energy in eV
        print("Current energy", EEV, "eV")

        # NewCAF: working array in which current (displaced) atomic t-matrix is stored
        if (IEL != 0):
            # TODO: when treating multiple atoms, choose the correct site for phaseshifts (IEL)
            t_matrix_new = tscatf(IEL, LMAX, interpolated_phaseshifts[i, IEL-1, :],
                            e_inside, VSITE, DR0, DRPER, DRPAR)
        else:
            t_matrix_new = np.full((LMAX+1,), dtype=np.complex128, fill_value=0.0)

        # DELWV : working space for computation and storage of amplitude differences
        for nc in range(n_geo):
            C = CDISP[nc,...]
            DELWV = MATEL_DWG(t_matrix_ref, t_matrix_new, e_inside, v_imag,
                            LMAX, tensor_amps_out, tensor_amps_in, out_k_par2, out_k_par3,
                            unit_cell_area, C)

            all_delwv[i, nc, :] = DELWV
        print(DELWV)
    with open('delta.npy','wb') as f:
        np.save(f, all_delwv[:, :, :])

if __name__ == '__main__':
    main()
