"""Module base."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-06-24'

from functools import partial

import jax

from viperleed_jax.atomic_units import to_internal_displacement_vector
from viperleed_jax.lib.calculator import normalize_occ_vector
from viperleed_jax.transformation_tree.derived_quantities import (
    DerivedQuantitySingleTree,
)


class NormalizedOccupations(DerivedQuantitySingleTree):
    """Derived quantity for normalized occupations."""

    def __init__(self, parameter_space, atom_ids):
        super().__init__(parameter_space)
        self.name = 'normalized_occupations'
        self.atom_ids = tuple(atom_ids)

    def _set_tree(self):
        """Set the tree for the derived quantity."""
        self.tree = self.parameter_space.occ_tree

    @partial(jax.jit, static_argnames=('self',))
    def __call__(self, params):
        """Calculate normalized occupations."""
        non_normalized_occupations = self.tree(params)
        return normalize_occ_vector(non_normalized_occupations, self.atom_ids)


class AtomicUnitsDisplacements(DerivedQuantitySingleTree):
    """Derived quantity for atomic units displacements."""

    def __init__(self, parameter_space):
        super().__init__(parameter_space)
        self.name = 'atomic_units_displacements'

    def _set_tree(self):
        """Set the tree for the derived quantity."""
        self.tree = self.parameter_space.geo_tree

    def __call__(self, params):
        """Calculate displacements in atomic units."""
        displacements = self.tree(params)
        return to_internal_displacement_vector(displacements)
