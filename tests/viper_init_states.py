from pathlib import Path

from pytest_cases import fixture

from viperleed_jax.from_state import run_viperleed_initialization

DATA_PATH = Path(__file__).parent / 'test_data'

INPUTS_CU_111_DYNAMIC_LMAX = DATA_PATH / 'Cu_111' / 'dynamic_l_max'
INPUTS_CU_111_FIXED_LMAX = DATA_PATH / 'Cu_111' / 'fixed_l_max'
INPUTS_FE2O3_012_CONVERGED = DATA_PATH / 'Fe2O3_012' / 'converged'
INPUTS_FE2O3_012_UNRELAXED = DATA_PATH / 'Fe2O3_012' / 'unrelaxed'
INPUTS_FE3O4_111 = DATA_PATH / 'Fe3O4_111'
INPUTS_PT_111_10x10_TE = DATA_PATH / 'Pt_111_10x10_Te' / 'converged'


@fixture
def state_cu_111_dynamic_l_max():
    return run_viperleed_initialization(INPUTS_CU_111_DYNAMIC_LMAX)


@fixture
def state_cu_111_fixed_l_max():
    return run_viperleed_initialization(INPUTS_CU_111_FIXED_LMAX)


@fixture
def state_fe2o3_012_converged():
    return run_viperleed_initialization(INPUTS_FE2O3_012_CONVERGED)


@fixture
def state_fe2o3_012_unrelaxed():
    return run_viperleed_initialization(INPUTS_FE2O3_012_UNRELAXED)


@fixture
def state_fe3o4_111():
    return run_viperleed_initialization(INPUTS_FE3O4_111)


@fixture
def state_pt_111_10x10_te():
    return run_viperleed_initialization(INPUTS_PT_111_10x10_TE)
