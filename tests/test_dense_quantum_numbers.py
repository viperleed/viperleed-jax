import pytest
import numpy as np
from viperleed_jax.data_structures import ReferenceData
from viperleed_jax import dense_quantum_numbers


def test_map_l_array_to_compressed_quantum_index_mapping():
    # Define your test case here
    input = np.array([1., 2., 3.])
    mapped = dense_quantum_numbers.map_l_array_to_compressed_quantum_index(input, 2)
    assert mapped == pytest.approx(
        np.array([1.,
                  2., 2., 2.,
                  3., 3., 3., 3., 3.])
    )

def test_map_l_array_to_compressed_quantum_index_raise():
    # input array with wrong shape
    input = np.array([1., 2., 3., 4., 5.])
    with pytest.raises(ValueError):
        dense_quantum_numbers.map_l_array_to_compressed_quantum_index(input, 10)
