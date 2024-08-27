import pickle

import pytest

# We can test that the read tensors are consistent with the pickled tensors
# and then do the other tests (also in the other files) from the pickled version
from viperleed_jax.lib_tensors import read_tensor

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
