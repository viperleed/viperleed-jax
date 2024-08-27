import os
from pathlib import Path
import pickle

import pytest

from otftleed.files.tensors import read_tensor_zip

if 'VIPERLEED_ON_THE_FLY_TESTS_LARGE_FILE_PATH' not in os.environ:
    raise ValueError('VIPERLEED_ON_THE_FLY_TESTS_LARGE_FILE_PATH not set')

LARGE_FILE_PATH = Path(os.environ['VIPERLEED_ON_THE_FLY_TESTS_LARGE_FILE_PATH'])

if not LARGE_FILE_PATH.exists():
    raise ValueError(f'Large file path {LARGE_FILE_PATH} does not exist')

@pytest.fixture(scope='session')
def large_file_path():
    return LARGE_FILE_PATH

@pytest.fixture(scope='session')
def fe2o3_unrelaxed_tensor_path():
    return LARGE_FILE_PATH / 'unrelaxed_frozen_for_testing' / 'Tensors_001.zip'

@pytest.fixture(scope='session')
def fe2o3_unrelaxed_pickle_path():
    return LARGE_FILE_PATH / 'unrelaxed_frozen_for_testing' / 'tensor.pckl'


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