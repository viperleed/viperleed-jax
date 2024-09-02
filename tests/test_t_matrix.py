from pathlib import Path
import pytest
import numpy as np

from viperleed_jax import t_matrix
from viperleed_jax.constants import BOHR


@pytest.fixture(scope='session')
def t_matrix_cu111_surf_vib_amp_0_3A():
    file = Path(__file__).parent / 'test_data' / 'Cu_111_new/Premade_cases/test_vibration_positive_change.npy'
    with open(file, 'rb') as f:
        t_matrix = np.load(f)
    return t_matrix

@pytest.fixture(scope='session')
def t_matrix_cu111_surf_vib_amp_0_1A():
    file = Path(__file__).parent / 'test_data' / 'Cu_111_new/Premade_cases/test_vibration_negative_change.npy'
    with open(file, 'rb') as f:
        t_matrix = np.load(f)
    return t_matrix

class TestTMatrix:

    def test_t_matrix_cu111_surf_vib_amp_0_3A(self,
                                              cu111_phaseshifts,
                                              t_matrix_cu111_surf_vib_amp_0_3A):
        energy = 2.28680897
        result = t_matrix.vib_dependent_tmatrix(
            l_max=14,
            phaseshifts=cu111_phaseshifts._phaseshifts[0,0,:],
            e_inside=energy,
            vib_amp=0.3*BOHR)
        assert result == pytest.approx(t_matrix_cu111_surf_vib_amp_0_3A[0,0], rel=1e-5)

    def test_t_matrix_cu111_surf_vib_amp_0_1A(self,
                                              cu111_phaseshifts,
                                              t_matrix_cu111_surf_vib_amp_0_1A):
        energy = 2.28680897
        result = t_matrix.vib_dependent_tmatrix(
            l_max=14,
            phaseshifts=cu111_phaseshifts._phaseshifts[0,0,:],
            e_inside=energy,
            vib_amp=0.1*BOHR)
        assert result == pytest.approx(t_matrix_cu111_surf_vib_amp_0_1A[0,0], rel=1e-5)

