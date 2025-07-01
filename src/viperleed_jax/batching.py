"""Module parameter_space."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-08-30'

import numpy as np

# TODO: make into pytree

# The calculation of delta-amplitudes is a computationally expensive operation
# that can be well parallelized. However, the computation involves operations
# with large input-, output- and intermediate arrays which can lead to memory
# issues. To avoid this, we can split the computation into smaller chunks and
# process them sequentially.
#
# There are two main ways in which we produce these batches:
# 1) The calculation may use an energy-dependent L-max value. The largest array
#    size scale with (L-max+1)^4, so we want to use the smallest L-max value
#    possible. Due to JAXs limitation on static array sizes, this gives us a
#    natural way to split the computation into smaller chunks that are processed
#    sequentially.
# 2) If the first point is not enough, or we want to calculate with a fixed
#    L-max value we can further split the computation into smaller chunks by
#    splitting the energy dimension.


class Batch:  # TODO: could be a dataclass
    """Represents a batch of energies with the same l_max.

    Batches are used to split the computation of delta-amplitudes into smaller
    chunks that can be processed in parallel.

    Parameters
    ----------
    l_max : int
        The maximum value of l.
    energies : list or array-like
        The energies associated with the batch.
    energy_indices : list or array-like
        The indices of the energies in the original dataset.

    Methods
    -------
    __len__():
        Returns the length of the batch.
    """

    def __init__(self, l_max, energies, energy_indices, batch_id):
        self.l_max = l_max
        self.energies = np.array(energies)
        self.energy_indices = np.array(energy_indices)
        self.batch_id = batch_id

    def __len__(self):
        """Returns the length of the batch."""
        return len(self.energies)


class Batching:
    def __init__(self, energies, l_max_per_energy):
        # l_max_per_energy must be in ascending order
        if not all(
            l_max_per_energy[i] <= l_max_per_energy[i + 1]
            for i in range(len(l_max_per_energy) - 1)
        ):
            raise ValueError('l_max_per_energy must be in ascending order')
        self.l_max_per_energy = l_max_per_energy


        self.max_l_max = max(l_max_per_energy)
        self.batches = self.create_batches(energies)

    def create_batches(self, energies):
        batches = []
        # iterate over the energies and create batches
        needed_l_max = set(self.l_max_per_energy)
        for batch_id, l_max in enumerate(sorted(needed_l_max)):
            batch_indices = np.where(np.array(self.l_max_per_energy) == l_max)[0]
            batch_energies = energies[batch_indices]
            batches.append(
                Batch(
                    l_max,
                    batch_energies,
                    batch_indices,
                    batch_id,
                )
            )

        # return as a tuple
        return tuple(batches)

    @property
    def max_batch_size(self):
        """Returns the maximum size of the batches"""
        return max(len(batch) for batch in self.batches)

    @property
    def restore_sorting(self):
        """Returns the indices to restore the original order of the energies."""
        return np.concatenate([batch.energy_indices for batch in self.batches])
