import numpy as np
import pytest
from viperleed_jax.batching import Batch, Batching

class TestBatch:
    def test_batch_initialization(self):
        batch = Batch(
            l_max=2, energies=[1, 2, 3], energy_indices=[0, 1, 2], batch_id=0
        )
        assert batch.l_max == 2
        np.testing.assert_array_equal(batch.energies, np.array([1, 2, 3]))
        np.testing.assert_array_equal(batch.energy_indices, np.array([0, 1, 2]))

    def test_batch_length(self):
        batch = Batch(
            l_max=2, energies=[1, 2, 3], energy_indices=[0, 1, 2], batch_id=0
        )
        assert len(batch) == 3


class TestBatching:
    def test_batching_initialization_ascending_l_max(self):
        # l_max_per_energy in ascending order.
        energies = np.array([10, 20, 30, 40])
        l_max_per_energy = [1, 2, 2, 3]
        batching = Batching(
            energies=energies, l_max_per_energy=l_max_per_energy
        )

        # Check that the provided l_max_per_energy is stored as given.
        assert batching.l_max_per_energy == l_max_per_energy

        # Unique l_max values sorted: [1, 2, 3] -> should create 3 batches.
        assert len(batching.batches) == 3

        # Validate first batch (l_max=1)
        batch0 = batching.batches[0]
        np.testing.assert_array_equal(batch0.energy_indices, np.array([0]))
        np.testing.assert_array_equal(batch0.energies, energies[[0]])
        assert batch0.l_max == 1

        # Validate second batch (l_max=2)
        batch1 = batching.batches[1]
        np.testing.assert_array_equal(batch1.energy_indices, np.array([1, 2]))
        np.testing.assert_array_equal(batch1.energies, energies[[1, 2]])
        assert batch1.l_max == 2

        # Validate third batch (l_max=3)
        batch2 = batching.batches[2]
        np.testing.assert_array_equal(batch2.energy_indices, np.array([3]))
        np.testing.assert_array_equal(batch2.energies, energies[[3]])
        assert batch2.l_max == 3

    def test_batch_ids(self):
        # Verify that batch IDs are assigned sequentially based on sorted unique l_max values.
        energies = np.array([10, 20, 30, 40])
        l_max_per_energy = [2, 2, 3, 3]
        batching = Batching(
            energies=energies, l_max_per_energy=l_max_per_energy
        )
        batch_ids = [batch.batch_id for batch in batching.batches]
        # Unique l_max values sorted: [2, 3] -> expected IDs: [0, 1]
        assert batch_ids == [0, 1]

    def test_batching_initialization_non_ascending_l_max(self):
        # l_max_per_energy not in ascending order should raise a ValueError.
        energies = np.array([10, 20, 30])
        l_max_per_energy = [3, 2, 1]
        with pytest.raises(ValueError):
            Batching(energies=energies, l_max_per_energy=l_max_per_energy)

    def test_restore_sorting_property(self):
        energies = np.array([10, 20, 30, 40, 50])
        l_max_per_energy = [1, 2, 2, 3, 3]
        batching = Batching(
            energies=energies, l_max_per_energy=l_max_per_energy
        )
        restored = batching.restore_sorting

        # Expected order: indices for l_max==1, then l_max==2, then l_max==3.
        expected = np.concatenate(
            [
                np.where(np.array(l_max_per_energy) == 1)[0],
                np.where(np.array(l_max_per_energy) == 2)[0],
                np.where(np.array(l_max_per_energy) == 3)[0],
            ]
        )
        np.testing.assert_array_equal(restored, expected)

    def test_max_batch_size_property(self):
        energies = np.array([10, 20, 30, 40, 50, 60])
        l_max_per_energy = [1, 2, 2, 3, 3, 3]
        batching = Batching(
            energies=energies, l_max_per_energy=l_max_per_energy
        )
        # Batch for l_max=3 has 3 elements, for l_max=2 has 2, and for l_max=1 has 1.
        assert batching.max_batch_size == 3

    def test_single_batch(self):
        # All energies have the same l_max, so there should be only one batch.
        energies = np.array([10, 20, 30])
        l_max_per_energy = [2, 2, 2]
        batching = Batching(
            energies=energies, l_max_per_energy=l_max_per_energy
        )
        assert len(batching.batches) == 1

        batch0 = batching.batches[0]
        np.testing.assert_array_equal(
            batch0.energy_indices, np.array([0, 1, 2])
        )
        np.testing.assert_array_equal(batch0.energies, energies)
        assert batch0.l_max == 2

        assert batching.max_batch_size == 3
        np.testing.assert_array_equal(
            batching.restore_sorting, np.array([0, 1, 2])
        )
