import pytest
import numpy as np

from viperleed_jax.batching import Batch, Batching

class TestBatch:
    def test_batch_initialization(self):
        batch = Batch(l_max=2, energies=[1, 2, 3], energy_indices=[0, 1, 2])
        assert batch.l_max == 2
        assert all(batch.energies == np.array([1, 2, 3]))
        assert all(batch.energy_indices == np.array([0, 1, 2]))

    def test_batch_length(self):
        batch = Batch(l_max=2, energies=[1, 2, 3], energy_indices=[0, 1, 2])
        assert len(batch) == 3

class TestBatching:
    def test_batching_initialization_ascending_l_max(self):
        batching = Batching(energies=[1, 2, 3], l_max_per_energy=[1, 2, 3], max_energies_per_batch=2)
        assert batching.l_max_per_energy == [1, 2, 3]

    def test_batching_initialization_non_ascending_l_max(self):
        with pytest.raises(ValueError):
            Batching(energies=[1, 2, 3], l_max_per_energy=[3, 2, 1], max_energies_per_batch=2)

    def test_batching_initialization_invalid_max_energies_per_batch(self):
        with pytest.raises(ValueError):
            Batching(energies=[1, 2, 3], l_max_per_energy=[1, 2, 3], max_energies_per_batch=-1)
