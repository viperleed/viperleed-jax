import jax
import numpy as np
import pytest
from pytest_cases import case, fixture, parametrize_with_cases

from tests.fixtures.base import LARGE_FILE_PATH
from viperleed_jax.lib.tensor_leed import t_matrix

_REF_DATA_PATH = LARGE_FILE_PATH / 't_matrices'

_CU_111_REFERENCE_FILE_NAME = 'ref_t_matrices_cu_111.npz'
_CU_111_REFERENCE_FILE = np.load(_REF_DATA_PATH / _CU_111_REFERENCE_FILE_NAME)
_CU_111_REFERENCE_VIB_AMPS = _CU_111_REFERENCE_FILE['vib_amps']

JIT_T_MATRIX_FUNC = jax.jit(
    t_matrix.vib_dependent_tmatrix, static_argnames='l_max'
)


@fixture(scope='session')
@pytest.mark.parametrize(
    'n,vib_amp',
    list(enumerate(_CU_111_REFERENCE_VIB_AMPS)),
    ids=list(_CU_111_REFERENCE_VIB_AMPS),
)
def store_t_matrix_cu_111(n, vib_amp):
    return vib_amp, _CU_111_REFERENCE_FILE['ref_t_matrices'][n, ...]


class TMatrixInputs:
    """Cases for testing the T-matrix calculation."""

    @case(tags='cu111')
    def case_cu111(
        self,
        cu_111_dynamic_l_max_info,
        cu_111_dynamic_l_max_phaseshifts,
        store_t_matrix_cu_111,
    ):
        # load stored data
        vib_amp, ref_t_matrices = store_t_matrix_cu_111
        energies = cu_111_dynamic_l_max_info.energies
        l_max = cu_111_dynamic_l_max_info.max_l_max
        phaseshifts = cu_111_dynamic_l_max_phaseshifts
        abs = 1e-8
        return l_max, phaseshifts, energies, vib_amp, ref_t_matrices, abs


@parametrize_with_cases(
    'l_max, phaseshifts, energies, vib_amp, t_matrices, abs',
    cases=TMatrixInputs,
)
def test_single_calculation(
    l_max, phaseshifts, energies, vib_amp, t_matrices, abs
):
    """Check that the t-matrix calculation yields the expected result."""
    energy_id = 0
    energy = energies[energy_id]
    site_el_ids = list(range(phaseshifts._phaseshifts.shape[0]))
    for s in site_el_ids:
        calculated = t_matrix.vib_dependent_tmatrix(
            l_max=l_max,
            phaseshifts=phaseshifts._phaseshifts[s, energy_id, :],
            e_inside=energy,
            vib_amp=vib_amp,
        )
        assert calculated == pytest.approx(t_matrices[s, energy_id], abs=abs)


@parametrize_with_cases(
    'l_max, phaseshifts, energies, vib_amp, t_matrices, abs',
    cases=TMatrixInputs,
)
def test_single_calculation_jit(
    l_max, phaseshifts, energies, vib_amp, t_matrices, abs
):
    """Check that the jit compiled function yields the same result as above"""
    energy_id = 0
    energy = energies[energy_id]
    site_el_ids = list(range(phaseshifts._phaseshifts.shape[0]))
    for s in site_el_ids:
        calculated = JIT_T_MATRIX_FUNC(
            l_max=l_max,
            phaseshifts=phaseshifts._phaseshifts[s, energy_id, :],
            e_inside=energy,
            vib_amp=vib_amp,
        )
        assert calculated == pytest.approx(t_matrices[s, energy_id], abs=abs)


@parametrize_with_cases(
    'l_max, phaseshifts, energies, vib_amp, t_matrices, abs',
    cases=TMatrixInputs,
)
def test_vmap_energy(l_max, phaseshifts, energies, vib_amp, t_matrices, abs):
    site_el_ids = list(range(phaseshifts._phaseshifts.shape[0]))
    for s in site_el_ids:
        calculated = t_matrix.vmap_energy_vib_dependent_tmatrix(
            l_max, phaseshifts._phaseshifts[s, :], energies, vib_amp
        )
        assert calculated == pytest.approx(t_matrices[s, ...], abs=abs)
