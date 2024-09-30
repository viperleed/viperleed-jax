import os
from pathlib import Path
import pickle

import numpy as np

import pytest
from pytest_cases import fixture, case

from viperleed_jax.from_state import run_viperleed_initialization
from viperleed_jax.files.tensors import read_tensor_zip
from viperleed_jax.data_structures import ReferenceData
from viperleed_jax.files import phaseshifts as ps
from viperleed_jax.tensor_calculator import TensorLEEDCalculator

from viperleed.calc.files.phaseshifts import readPHASESHIFTS

from tests.fixtures.cu_111_dynamic_l_max import *
from tests.fixtures.cu_111_fixed_l_max import *

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

@pytest.fixture(scope='session')
def fe2o3_tensor_calculator(fe2o3_ref_data_fixed_lmax_12):
    ref_data = fe2o3_ref_data_fixed_lmax_12

########
# Cu111#
########

@pytest.fixture(scope='session')
def cu111_input_path():
    return Path(__file__).parent / 'test_data' / 'Cu_111_new'

@pytest.fixture(scope='session')
def cu111_tensor_path():
    return Path(__file__).parent / 'test_data' / 'Cu_111_new' / 'Tensors' / 'Tensors_001.zip'

@pytest.fixture(scope='session')
def cu111_energies():
    return np.array([
        2.28680897,  2.4583211 ,  2.63103557,  2.80476761,  2.97936988,
        3.15472341,  3.33073044,  3.50731039,  3.68439579,  3.86193037,
        4.03986502,  4.2181592 ,  4.39677715,  4.57568788,  4.75486517,
        4.93428421,  5.11392546,  5.29376984,  5.47380161,  5.65400648,
        5.83437061,  6.01488304,  6.1955328 ,  6.37631083,  6.55720854,
        6.73821783,  6.91933107,  7.10054302,  7.281847  ,  7.46323729,
        7.64471006,  7.82625914,  8.00788212,  8.18957233,  8.37132931,
        8.55314732,  8.7350235 ,  8.91695595,  9.09894276,  9.28097916,
        9.46306324,  9.6451931 ,  9.82736778, 10.00958443, 10.19184017,
        10.37413406, 10.55646515, 10.73883247, 10.92123318, 11.10366535,
        11.286129  ], dtype=np.float64)

@pytest.fixture(scope='session')
def cu111_state_after_init(cu111_input_path):
    state_after_init = run_viperleed_initialization(cu111_input_path)
    slab, rpars = state_after_init.slab, state_after_init.rpars
    return slab, rpars

@pytest.fixture(scope='session')
def cu111_raw_phaseshifts(cu111_state_after_init, cu111_input_path):
    slab, rpars = cu111_state_after_init
    phaseshifts_path = cu111_input_path / 'PHASESHIFTS'
    _, raw_phaseshifts, _, _ = readPHASESHIFTS(
        slab, rpars, readfile=phaseshifts_path, check=True, ignoreEnRange=False)
    return raw_phaseshifts

@pytest.fixture(scope='session')
def cu111_phaseshifts(cu111_state_after_init, cu111_raw_phaseshifts, cu111_energies):
    slab, rpars = cu111_state_after_init
    phaseshift_map = ps.phaseshift_site_el_order(slab, rpars)
    return ps.Phaseshifts(cu111_raw_phaseshifts,
                                cu111_energies,
                                l_max=14,
                                phaseshift_map=phaseshift_map)
