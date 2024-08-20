import pickle

import pytest
import numpy as np

from src.files.tensors import read_tensor_zip

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

# We can test that the read tensors are consistent with the pickled tensors
# and then do the other tests (also in the other files) from the pickled version

class TestReadTensors:
    # Test read_tensor_zip and check consistency vs. pickled tensors

    def test_read_tensor_length(self, fe2o3_read_tensor_zip, fe2o3_pickled_tensor):
        assert len(fe2o3_read_tensor_zip) == 30
        assert len(fe2o3_read_tensor_zip) == len(fe2o3_pickled_tensor)

    def test_read_tensor_consistency(self, fe2o3_read_tensor_zip, fe2o3_pickled_tensor):
        for file_name in fe2o3_read_tensor_zip.keys():
            assert file_name in fe2o3_pickled_tensor.keys()
            assert fe2o3_read_tensor_zip[file_name].is_consistent(fe2o3_pickled_tensor[file_name])


class TestPickledTensors:
    def test_tensors_self_consistent(self, fe2o3_pickled_tensor):
        for file_name in fe2o3_pickled_tensor.keys():
            assert fe2o3_pickled_tensor[file_name].is_consistent(fe2o3_pickled_tensor[file_name])

    def test_tensors_inter_consistent(self, fe2o3_pickled_tensor):
        for file_name in fe2o3_pickled_tensor.keys():
            for other_file in fe2o3_pickled_tensor.keys():
                if file_name != other_file:
                    assert fe2o3_pickled_tensor[file_name].is_consistent(fe2o3_pickled_tensor[other_file])

    def test_expected_attribs(self, fe2o3_pickled_tensor):
        for tensor in fe2o3_pickled_tensor.values():
            assert tensor.n_beams == 38
            assert tensor.n_energies == 208
            assert len(tensor.e_kin) == 208
