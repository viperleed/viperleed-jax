"""Module derived_quantities."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-06-24'


from dataclasses import dataclass, field
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp

from viperleed_jax import atomic_units, lib_math
from viperleed_jax.propagator import (
    calc_propagator,
    calculate_propagators,
    symmetry_operations,
)
from viperleed_jax.transformation_tree.derived_quantities import (
    LinearPropagatedQuantity,
)
from viperleed_jax.transformation_tree.linear_transformer import LinearMap


@partial(
    jax.tree_util.register_dataclass,   
    data_fields=[
        'kappa',
        'static_propagators',
        'propagator_transpose_int',
        'symmetry_operations',
        'propagator_id',
        'is_dynamic_propagator',
    ],
    meta_fields=[],
)
@dataclass
class PropagatorContext:
    kappa: jnp.ndarray  # shape (n_energies,)
    static_propagators: jnp.ndarray  # shape (atom_basis, n_energies, lm, m)
    propagator_transpose_int: jnp.ndarray  # shape (atom_basis,)
    symmetry_operations: jnp.ndarray  # shape (atom_basis, lm, m)
    propagator_id: jnp.ndarray
    is_dynamic_propagator: jnp.ndarray

class Propagators(LinearPropagatedQuantity):
    def __init__(
        self,
        geo_tree,
        kappa,
        energies,
        batch_energies,
        batch_atoms,
        max_l_max,
    ):
        super().__init__(geo_tree, name='propagators', transformer_class=LinearMap)
        self.geo_tree = geo_tree
        self.kappa = jnp.asarray(kappa)
        self.energies = jnp.asarray(energies)
        self.batch_energies = batch_energies
        self.batch_atoms = batch_atoms
        self.max_l_max = max_l_max

        # TODO: logging
        static_propagators = self._calculate_static_propagators()

        # rotation angles
        propagator_symmetry_operations, propagator_transpose = (
            self._propagator_rotation_factors()
        )
        self.propagator_symmetry_operations = jnp.asarray(
            propagator_symmetry_operations
        )
        # NB: Using an integer array here because there seems so be some kind of
        # bug where jax.jit would flip on of the boolean values for some
        # cases.
        propagator_transpose_int = propagator_transpose.astype(jnp.int32)

        self.context = PropagatorContext(
            is_dynamic_propagator=self.is_dynamic_propagator,
            propagator_id=self.propagator_id,
            kappa=self.kappa,
            static_propagators=static_propagators,
            propagator_transpose_int=propagator_transpose_int,
            symmetry_operations=self.propagator_symmetry_operations,
        )

    @property
    def n_dynamic_propagators(self):
        return self.n_dynamic_values

    @property
    def n_static_propagators(self):
        return self.n_static_values

    @property
    def propagator_map(self):
        return self.static_dynamic_map

    @property
    def is_dynamic_propagator(self):
        return np.array([val == 'dynamic' for (val, id) in self.propagator_map])

    @property
    def propagator_id(self):
        return np.array([id for (val, id) in self.propagator_map])

    @property
    def static_propagator_inputs(self):
        return (
            self.static_reference_nodes_values
        )

    @property
    def propagator_plane_symmetry_operations(self):
        return self.leaf_plane_symmetry_operations

    @property
    def leaf_plane_symmetry_operations(self):
        """Return the in-plane symmetry operations for each leaf in respect to the
        reference displacement (the one for which the propagator is calculated).
        """
        return tuple(
            sym_op.weights[1:, 1:]
            for sym_op in self._arg_transformers
        )

    def _propagator_rotation_factors(self):
        ops = [
            symmetry_operations(self.max_l_max, plane_sym_op)
            for plane_sym_op in self.propagator_plane_symmetry_operations
        ]
        symmetry_tensors = np.array([op[0] for op in ops])
        mirror_propagators = np.array([op[1] for op in ops])

        return symmetry_tensors, mirror_propagators


    def __call__(self, displacements_au, energy_ids):
        return calculate_propagators(
            self.context,
            displacements_au,
            energy_ids,
            self.batch_energies,
            self.batch_atoms,
            self.max_l_max,
        )

    def _calculate_static_propagators(self):
        # Convert static propagator inputs to an array.
        static_inputs = self.static_propagator_inputs
        if len(static_inputs) == 0:
            # If there are no static inputs, store an empty array.
            self._static_propagators = jnp.array([])
            return

        displacements_ang = jnp.asarray(static_inputs)
        displacements_au = atomic_units.to_internal_displacement_vector(
            displacements_ang
        )
        spherical_harmonics_components = jnp.array(
            [
                lib_math.spherical_harmonics_components(self.max_l_max, disp)
                for disp in displacements_au
            ]
        )

        # Outer loop: iterate over energy indices.
        def energy_fn(e_idx):
            # For each energy, iterate over all displacements.
            def displacement_fn(i):
                disp = displacements_au[i]
                comps = spherical_harmonics_components[i]
                return calc_propagator(
                    self.max_l_max,
                    disp,
                    comps,
                    self.kappa[e_idx],
                )

            return jax.lax.map(
                displacement_fn,
                jnp.arange(displacements_au.shape[0]),
                batch_size=self.batch_atoms,
            )

        # Map over energies with the specified batch size.
        static_propagators = jax.lax.map(
            energy_fn,
            jnp.arange(len(self.energies)),
            batch_size=self.batch_energies,
        )
        # The result has shape (num_energies, num_displacements, ...).
        # Use einsum to swap axes so that the final shape is
        # (num_displacements, num_energies, ...), matching the original ordering.
        return jnp.einsum(
            'ed...->de...', static_propagators
        )
