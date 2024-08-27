# Batching module

import numpy as np

#TODO: make into pytree

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

class Batch:
    def __init__(self, l_max, energies, energy_indices):
        self.l_max = l_max
        self.energies = np.array(energies)
        self.energy_indices = np.array(energy_indices)

    def __len__(self):
        return len(self.energies)


class Batching:
    def __init__(self, energies, l_max_per_energy, max_energies_per_batch=None):
        # l_max_per_energy must be in ascending order
        if not all(l_max_per_energy[i] <= l_max_per_energy[i+1]
                   for i in range(len(l_max_per_energy)-1)):
            raise ValueError("l_max_per_energy must be in ascending order")
        self.l_max_per_energy = l_max_per_energy
        
        if (not isinstance(max_energies_per_batch, int )
            or max_energies_per_batch <= 0):
            raise ValueError("max_energies_per_batch must be a positive integer")

        self.max_l_max = max(l_max_per_energy)
        self.batches = self.create_batches(energies, max_energies_per_batch)

    def create_batches(self, energies, max_energies_per_batch):
        batches = []
        # iterate over the energies and create batches
        current_l_max = self.l_max_per_energy[0]
        batch_energies = []
        batch_indices = []

        for en_id, energy in enumerate(energies):
            if (self.l_max_per_energy[en_id] != current_l_max
                or len(batch_energies) == max_energies_per_batch):
                # finish the batch and start a new one
                batches.append(
                    Batch(current_l_max, batch_energies, batch_indices)
                )
                batch_energies = [energy,]
                batch_indices = [en_id]
            else:
                batch_energies.append(energy)
                batch_indices.append(en_id)
            current_l_max = self.l_max_per_energy[en_id]

        # finish the last batch if there are any energies left
        if batch_energies:
            batches.append(
                Batch(current_l_max, batch_energies, batch_indices)
            )

        # return as a tuple
        return tuple(batches)

    @property
    def max_batch_size(self):
        return max(len(batch) for batch in self.batches)

    @property
    def restore_sorting(self):
        # create a mapping to restore the original order
        return np.concatenate([batch.energy_indices for batch in self.batches])
