import os
from pathlib import Path
import pickle

import pytest


from viperleed_jax.from_state import run_viperleed_initialization
from viperleed_jax.files.tensors import read_tensor_zip
from viperleed_jax.data_structures import ReferenceData

if 'VIPERLEED_ON_THE_FLY_TESTS_LARGE_FILE_PATH' not in os.environ:
    raise ValueError('VIPERLEED_ON_THE_FLY_TESTS_LARGE_FILE_PATH not set')

LARGE_FILE_PATH = Path(os.environ['VIPERLEED_ON_THE_FLY_TESTS_LARGE_FILE_PATH'])

FE2O3_UNRELAXED_INPUT_PATH = Path(__file__).parent / 'test_data' / 'Fe2O3_012' / 'unrelaxed'

if not LARGE_FILE_PATH.exists():
    raise ValueError(f'Large file path {LARGE_FILE_PATH} does not exist')

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
    tensors = read_tensor_zip(fe2o3_unrelaxed_tensor_path, lmax=14, n_beams=38, n_energies=208)
    return tensors

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
    return ReferenceData(tensor_tuple, fix_lmax=fixed_lmax)
