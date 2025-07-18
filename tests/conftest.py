import pickle
from pathlib import Path

import pytest
import jax
from pytest_cases import fixture, parametrize_with_cases

from tests.fixtures.base import LARGE_FILE_PATH
from tests.fixtures.cu_111_dynamic_l_max import *
from tests.fixtures.cu_111_fixed_l_max import *
from tests.fixtures.fe2o3_012_converged import *
from tests.viper_init_states import (
    state_cu_111_dynamic_l_max,
    state_cu_111_fixed_l_max,
    state_fe2o3_012_converged,
    state_fe2o3_012_unrelaxed,
    state_fe3o4_111,
    state_pt_111_10x10_te,
    init_state_pt25rh75_o_3x1,
)
from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.ref_calc_data import process_tensors
from viperleed_jax.files import phaseshifts as ps
from viperleed_jax.files.tensors import read_tensor_zip
from viperleed_jax.from_state import run_viperleed_initialization

from .structures import CaseStatesAfterInit

FE2O3_UNRELAXED_INPUT_PATH = (
    Path(__file__).parent / 'test_data' / 'Fe2O3_012' / 'unrelaxed'
)

# Make sure to use double precision for testing
jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='session')
def large_file_path():
    return LARGE_FILE_PATH


@pytest.fixture(scope='session')
def fe2o3_unrelaxed_input_path():
    return FE2O3_UNRELAXED_INPUT_PATH


@pytest.fixture(scope='session')
def fe2o3_unrelaxed_tensor_path():
    return LARGE_FILE_PATH / 'Fe2O3_012' / 'unrelaxed' / 'Tensors_001.zip'


@pytest.fixture(scope='session')
def fe2o3_unrelaxed_pickle_path():
    return LARGE_FILE_PATH / 'Fe2O3_012' / 'unrelaxed' / 'tensor.pckl'


# Reading the zipped tensors takes 5+ minutes
@pytest.fixture(scope='session')
def fe2o3_read_tensor_zip(fe2o3_unrelaxed_tensor_path):
    return read_tensor_zip(
        fe2o3_unrelaxed_tensor_path, lmax=14, n_beams=38, n_energies=208
    )


# Therefore, we have pickeled tensors for faster testing
@pytest.fixture(scope='session')
def fe2o3_pickled_tensor(fe2o3_unrelaxed_pickle_path):
    with open(fe2o3_unrelaxed_pickle_path, 'rb') as f:
        tensors = pickle.load(f)
    return tensors


# State after initialization (for slab and rpars)
@pytest.fixture(scope='session')
def fe2o3_unrelaxed_state_after_init():
    state_after_init = run_viperleed_initialization(FE2O3_UNRELAXED_INPUT_PATH)
    slab, rpars = state_after_init.slab, state_after_init.rpars
    return slab, rpars


@pytest.fixture(scope='session')
def fe2o3_ref_data_fixed_lmax_12(fe2o3_pickled_tensor):
    fixed_lmax = 12
    tensor_tuple = tuple(fe2o3_pickled_tensor.values())
    return process_tensors(tensor_tuple, fix_lmax=fixed_lmax)


# Atom basis for various structures
@fixture
@parametrize_with_cases('test_case', cases=CaseStatesAfterInit)
def atom_basis(test_case):
    """Fixture for creating an AtomBasis."""
    state, _ = test_case
    return AtomBasis(state.slab)
