"""Module base."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-06-24'


from abc import ABC, abstractmethod

from ..calculator import normalize_occ_vector
from ...atomic_units import to_internal_displacement_vector

class DerivedQuantity(ABC):
    def __init__(self, tree):
        self.tree = tree

    @abstractmethod
    def __call__(self, params):
        pass



class NormalizedOccupations(DerivedQuantity):
    """Derived quantity for normalized occupations."""

    def __init__(self, tree, atom_ids):
        super().__init__(tree)
        self.name = 'normalized_occupations'
        self.atom_ids = tuple(atom_ids)

    def __call__(self, params):
        """Calculate normalized occupations."""
        non_normalized_occupations = self.tree(params)
        return normalize_occ_vector(
            non_normalized_occupations,
            self.atom_ids
        )

class AtomicUnitsDisplacements(DerivedQuantity):
    """Derived quantity for atomic units displacements."""

    def __init__(self, tree):
        super().__init__(tree)
        self.name = 'atomic_units_displacements'

    def __call__(self, params):
        """Calculate displacements in atomic units."""
        displacements = self.tree(params)
        return to_internal_displacement_vector(displacements)
