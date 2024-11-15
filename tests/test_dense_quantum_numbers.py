import numpy as np
import pytest

from viperleed_jax import dense_quantum_numbers


def test_correct_stored_quantum_numbers():
    """"""
    expected_qns = dense_quantum_numbers._dense_quantum_numbers(
        2 * dense_quantum_numbers.MAXIMUM_LMAX
    )
    assert (
        dense_quantum_numbers._FULL_DENSE_QUANTUM_NUMBERS == expected_qns
    ).all()  # int comparison


def test_map_l_array_to_compressed_quantum_index_mapping():
    # Define your test case here
    input = np.array([1.0, 2.0, 3.0])
    mapped = dense_quantum_numbers.map_l_array_to_compressed_quantum_index(
        input, 2
    )
    assert mapped == pytest.approx(
        np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    )


def test_map_l_array_to_compressed_quantum_index_raise():
    # input array with wrong shape
    input = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        dense_quantum_numbers.map_l_array_to_compressed_quantum_index(input, 10)
