"""Module onset_height_change."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-06-24'

from functools import partial

import jax
from jax import numpy as jnp

from viperleed_jax.constants import ATOM_Z_DIR_ID, DISP_Z_DIR_ID
from viperleed_jax.transformation_tree.derived_quantities import (
    DerivedQuantitySingleTree,
)


class OnsetHeightChange(DerivedQuantitySingleTree):
    """Derived quantity for onset height change.

    Used to be called CXDisp in TensErLEED
    """

    def __init__(self, parameter_space):
        super().__init__(parameter_space)
        self.name = 'onset_height_change'

        # z positions of atoms in reference calculation
        self.atoms_ref_z_position = jnp.array(
            [leaf.atom.atom.cartpos[ATOM_Z_DIR_ID] for leaf in self.tree.leaves]
        )

    def _set_tree(self):
        """Set the tree for the derived quantity."""
        self.tree = self.parameter_space.geo_tree

    @partial(jax.jit, static_argnames=('self',))
    def __call__(self, params):
        """Calculate the change in onset height."""
        displacements = self.tree(params)
        z_changes = displacements[:, DISP_Z_DIR_ID]
        new_z_pos = self.atoms_ref_z_position + z_changes
        return jnp.max(new_z_pos) - jnp.max(self.atoms_ref_z_position)
