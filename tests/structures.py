"""Module structures of tests."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-26'

from pathlib import Path

from pytest_cases import parametrize

from viperleed_jax.from_state import run_viperleed_initialization

DATA_PATH = Path(__file__).parent / 'test_data'

INPUTS_CU_111_DYNAMIC_LMAX = DATA_PATH / 'Cu_111' / 'dynamic_l_max'
INPUTS_CU_111_FIXED_LMAX = DATA_PATH / 'Cu_111' / 'fixed_l_max'
INPUTS_FE2O3_012_CONVERGED = DATA_PATH / 'Fe2O3_012' / 'converged'
INPUTS_FE2O3_012_UNRELAXED = DATA_PATH / 'Fe2O3_012' / 'unrelaxed'
INPUTS_FE3O4_111 = DATA_PATH / 'Fe3O4_111'
INPUTS_PT_111_10x10_TE = DATA_PATH / 'Pt_111_10x10_Te' / 'converged'

INPUT_PATHS = (
    INPUTS_CU_111_DYNAMIC_LMAX,
    INPUTS_CU_111_FIXED_LMAX,
    INPUTS_FE2O3_012_CONVERGED,
    INPUTS_FE2O3_012_UNRELAXED,
    INPUTS_FE3O4_111,
    INPUTS_PT_111_10x10_TE,
)


class CaseStatesAfterInit:
    """Collection of cases with structures after viperleed.calc init."""

    @parametrize(calc_path=INPUT_PATHS)
    def case_state_after_init(self, calc_path):
        # returns states after initialization
        # slab and rpars can be accessed as state.slab, state.rpars
        return run_viperleed_initialization(calc_path)
