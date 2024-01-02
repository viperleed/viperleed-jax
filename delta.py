# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:31:49 2023

@author: haide
"""

import numpy as np
import fortranformat as ff

from lib_phaseshifts import *
from lib_tensors import *
from lib_tscatf import *

#From "GLOBAL"
MEMACH = 1.0E-6
HARTREE = 27.211396
BOHR = 0.529177

#From "PARAM"
LMAX = 14  # maximum angular momentum to be used in calculation
NT0 = 9  # no. of TLEED output beams
NATOMS = 1  # currently 1 is the only possible choice
NCSTEP = 201  # number of geometric variations ('displacements') to be considered

LMMAX = (LMAX + 1)*(LMAX + 1)
L1 = LMAX + 1
LMAX21 = 2*LMAX + 1
LMMAX2 = LMAX21*LMAX21

# From Stdin
# DR0,DRPER,DRPAR: thermal vibration amplitude steps to be included in
# current variation - DR0 is always 0., DRPER = DRPAR forced
DR0 = 0
DRPER = 0.1908624
DRPAR = DRPER
CUNDISP = np.full((NATOMS, 3),dtype=np.float64,fill_value=0.0)  # position of current atomic site in reference calculation
CUNDISP[0][0] = 0
CUNDISP[0][1] = 0
CUNDISP[0][2] = 0
CDISP = np.full((NCSTEP, NATOMS, 3),dtype=np.float64,fill_value=np.nan)  # displaced positions of current atomic site for variation
for i in range(NCSTEP):
    CDISP[i][0][0] = -0.0005*i + 0.05
    CDISP[i][0][1] = 0
    CDISP[i][0][2] = 0
ARB1 = [1.2722, -2.2036]  # einheits vektoren
ARB2 = [1.2722, 2.2036]
for i in range(2):
    ARB1[i] /= BOHR
    ARB2[i] /= BOHR
TV = abs(ARB1[0]*ARB2[1]-ARB1[1]*ARB2[0])  # area of (overlayer) lateral unit cell - in case TLEED wrt smaller unit cell
#                                            is used, TVA from reference computation must be set.



IEL = 1  # element no. (in phase shifts supplied with input) that delta amplitudes
#          will be calculated for (not necessarily the same element as the one
#          used in the reference calculation!) - IEL = 0 means a vacancy will be assumed

VSITE = 0  # possible energy shift in phase shift computations - can be used to describe
#            local variations of the muffin-tin-constant

NN3 = LMAX+1
NN2 = LMAX+1
NN1 = NN2+NN3-1
PPP = cppp(NN1, NN2, NN3)  # Clebsh-Gordon coefficients for computation of temperature-dependent phase shifts

T0 = 100  # must equal T if input vib. amplitudes are to be used properly - not 0. !!
T = 100  # must equal T0 if input vib. amplitudes are to be used properly - not 0. !!
#          if T0 is set to the temperature that the vibs. were computed for, T could
#          in principle be used to simulate the temperature behaviour of
#          a Debye-like phonon spectrum. Yet, this simply alters the vib. amplitude used
#          for the DW factor, thus it only makes sense to either vary DRPER or T.
NRATIO = 1  # originally, ratio between substrate and overlayer unit cell area. However,
#             currently all TLEED parts must be performed with overlayer symmetry only,
#             thus NRATIO = 1 and TV = TVB are the only safe choice


BELM = get_clebsh_gordon(LMAX)  # Clebsh-Gordon coefficients for tmatrix()
firstline, phaseshifts, newpsGen, newpsWrite = readPHASESHIFTS(None, None, readfile='PHASESHIFTS', check=False, ignoreEnRange=False)

n_energies = 0
with open('T_1', 'r') as datei:
    for zeile in datei:
        if '-1' in zeile:
            n_energies += 1

my_dict = read_tensor('T_1', n_beams=9, n_energies= n_energies, l_max=LMAX+1)

all_delwv = np.full((1, NCSTEP, NT0), dtype=np.complex128, fill_value=np.nan)
for i in range(1):
    E = my_dict['e_kin'][i]  # computational energy inside crystal
    CAF = my_dict['t_matrix'][i]  # atomic t-matrix of current site as used in reference calculation
    VV = my_dict['v0r'][i]  # real part of the inner potential
    VPIS = my_dict['v0i_substrate'][i]  # imaginary part of the inner potential, substrate, resp.
    EXLM = my_dict['tensor_amps_out'][i]  # spherical wave amplitudes incident from exit beam NEXIT in "time-reversed"
    #                                       LEED experiment (or rather, all terms of Born series immediately after
    #                                       scattering on current atom)
    ALM = my_dict['tensor_amps_in'][i]  # spherical wave amplitudes incident on current atomic site in reference calculation
    #                                     (i.e., scattering path ends before scattering on that atom)
    AK2M, AK3M = my_dict['kx_in'][i], my_dict['ky_in'][i]  # (negative) absolute lateral momentum of Tensor LEED beams
    #                                                        (for use as incident beams in time-reversed LEED calculation)
    PSQ = my_dict['k_delta'][i]  # lateral momentum of Tensor LEED beams relative to incident beam (0,0)

    EEV = (E-VV)*HARTREE  # current energy in eV
    print("Current energy", EEV, "eV")

    # NewCAF: working array in which current (displaced) atomic t-matrix is stored
    if (IEL != 0):
        NewCAF = tscatf(IEL, L1, phaseshifts, E, VSITE, PPP, NN1, NN2, NN3, DR0, DRPER, DRPAR, T0, T)
    else:
        NewCAF = np.full((LMAX+1,), dtype=np.complex128, fill_value=0.0)

    # DELWV : working space for computation and storage of amplitude differences
    DELWV = MATEL_DWG(NCSTEP, CAF, NewCAF, BELM, E, VV, VPIS, LMAX, LMMAX, NT0, EXLM, ALM, AK2M, AK3M,
                      NRATIO, TV, LMAX, LMMAX, NATOMS, CDISP, CUNDISP, PSQ, LMAX21, LMMAX2)

    all_delwv[i, :, :] = DELWV
    print(DELWV)
with open('test.npy','wb') as f:
    np.save(f, all_delwv[:, :, 0])
