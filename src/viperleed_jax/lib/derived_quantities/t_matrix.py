"""Module t-matrix."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-06-24'


from dataclasses import dataclass, field
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from viperleed.calc import LOGGER as logger

from viperleed_jax.lib.tensor_leed.t_matrix import (
    calculate_t_matrices,
    vib_dependent_tmatrix,
)
from viperleed_jax.transformation_tree.derived_quantities import (
    LinearPropagatedQuantity,
)
from viperleed_jax.transformation_tree.linear_transformer import LinearMap


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        'energies',
        'static_t_matrices',
        't_matrix_id',
        'is_dynamic_mask',
    ],
    meta_fields=[
        'dynamic_site_elements',
    ],
)
@dataclass
class TMatrixContext:
    energies: jnp.ndarray
    static_t_matrices: jnp.ndarray
    dynamic_site_elements: jnp.ndarray = field(metadata=dict(static=True))
    t_matrix_id: jnp.ndarray
    is_dynamic_mask: jnp.ndarray


class TMatrix(LinearPropagatedQuantity):
    def __init__(
        self,
        vib_tree,
        energies,
        phaseshifts,
        batch_energies,
        max_l_max,
    ):
        super().__init__(vib_tree, name='t-matrix', transformer_class=LinearMap)
        self.phaseshifts = phaseshifts
        self.batch_energies = batch_energies
        logger.debug(
            f'Pre-calculating {self.n_static_values} static t-matrices.'
        )
        static_t_matrices = self._calculate_static_t_matrices(energies, batch_energies, max_l_max)

        self.context = TMatrixContext(
            energies=energies,
            static_t_matrices=static_t_matrices,
            dynamic_site_elements=self.dynamic_t_matrix_site_elements,
            t_matrix_id=self.t_matrix_id.copy(),
            is_dynamic_mask=self.is_dynamic_t_matrix.copy(),
        )

    @property
    def n_dynamic_t_matrices(self):
        return self.n_dynamic_values

    @property
    def n_static_t_matrices(self):
        return self.n_static_values

    @property
    def static_t_matrix_inputs(self):
        return [
            (node.site_element, node.ref_vib_amp)
            for node in self.static_reference_nodes
        ]

    @property
    def dynamic_t_matrix_site_elements(self):
        return tuple(node.site_element for node in self.dynamic_reference_nodes)

    @property
    def t_matrix_map(self):
        return self.static_dynamic_map

    @property
    def is_dynamic_t_matrix(self):
        return np.array([val == 'dynamic' for (val, id) in self.t_matrix_map])

    @property
    def t_matrix_id(self):
        return np.array([id for (val, id) in self.t_matrix_map])

    def __call__(
        self, vib_amps_au, l_max, energy_ids
    ):
        return calculate_t_matrices(
            self.context,
            l_max,
            self.batch_energies,
            self.phaseshifts,
            vib_amps_au,
            energy_ids,
        )

    def _calculate_static_t_matrices(self, energies, batch_energies, max_l_max):
        # This is only done once – perform for maximum lmax and crop later
        energy_indices = jnp.arange(len(energies))

        # Outer loop: iterate over energy indices with batching
        def energy_fn(e_idx):
            # For each energy, compute t-matrices for all static input pairs.
            # self._parameter_space.static_t_matrix_inputs is assumed to be a list
            # of (site_el, vib_amp) pairs.
            def compute_t(pair):
                site_el, vib_amp = pair
                return vib_dependent_tmatrix(
                    max_l_max,
                    self.phaseshifts[site_el][e_idx, : max_l_max + 1],
                    energies[e_idx],
                    vib_amp,
                )

            # Use a Python loop to compute for each pair and stack the results.
            # This loop is over a typically small list so it shouldn't be a bottleneck.
            return jnp.stack(
                [
                    compute_t(pair)
                    for pair in self.static_t_matrix_inputs
                ]
            )

        # Map over energies with the given batch size.
        static_t_matrices = jax.lax.map(
            energy_fn, energy_indices, batch_size=batch_energies
        )
        # static_t_matrices has shape (num_energies, num_static_inputs, lm, ...),
        # which is equivalent to the original einsum('ael->eal') result.
        return static_t_matrices
