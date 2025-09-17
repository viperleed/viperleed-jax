"""Module normalized_occupations."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-06-25'

from functools import partial

import jax
from jax import numpy as jnp
import numpy as np

from viperleed_jax.lib.math import (
    apply_fun_grouped,
    mirror_across_plane_sum_1,
    project_onto_plane_sum_1,
)
from viperleed_jax.transformation_tree.derived_quantities import (
    DerivedQuantitySingleTree,
)
from viperleed_jax.lib.bounded_simplex import bounded_softmax_from_unit
from viperleed_jax.lib.math import EPS
from viperleed_jax.transformation_tree.occ_parameters import (
    OccTotalOccupationConstraint,
)

class NormalizedOccupations(DerivedQuantitySingleTree):
    """Derived quantity for normalized occupations."""

    def __init__(self, parameter_space, op_type='mirror'):
        super().__init__(parameter_space)
        self.name = 'normalized_occupations'
        self.op_type = op_type

        # get the min and max possible concentrations for each site
        min_results = self.tree(np.zeros((self.tree.root.dof,)))
        max_results = self.tree(np.ones((self.tree.root.dof,)))
        self.lows = np.minimum(min_results, max_results)
        self.highs = np.maximum(min_results, max_results)
        self.fixed = abs(self.highs - self.lows) < EPS

        # check for total occupation constraints
        sum_group_min_occs, sum_group_max_occs = [], []
        for i in range(max(self.atom_ids)):
            atom_indices = np.where(np.array(self.atom_ids) == i)[0]
            sum_group_min_occs.append(np.sum(self.lows[atom_indices]))
            sum_group_max_occs.append(np.sum(self.highs[atom_indices]))
        max_vacancies = jnp.clip(1 - jnp.array(sum_group_min_occs), max=1.0)
        min_vacancies = jnp.clip(1 - jnp.array(sum_group_max_occs), min=0.0)
        for i, leaf in enumerate(self.tree.leaves):
            for ancestor in leaf.ancestors:
                if isinstance(ancestor, OccTotalOccupationConstraint):
                    max_vacancies = max_vacancies.at[i].set(
                        1 - ancestor.total_occupation
                    )
                    min_vacancies = min_vacancies.at[i].set(
                        1 - ancestor.total_occupation
                    )
                    break

        self.max_vacancies = max_vacancies
        self.min_vacancies = min_vacancies

    def _set_tree(self):
        """Set the tree for the derived quantity."""
        self.tree = self.parameter_space.occ_tree
        self.atom_ids = tuple(self.tree.atom_basis.atom_ids)

    # @partial(jax.jit, static_argnames=('self',))
    def __call__(self, params):
        """Calculate normalized occupations."""
        non_normalized_occupations = self.tree(params)
        # scale back to [0, 1] from [lows, highs]
        scaled_occupations = (non_normalized_occupations - self.lows) / (
            self.highs - self.lows
        )
        scaled_occupations = jnp.where(self.fixed, 0.5, scaled_occupations)

        bounded_simplex_inputs = jnp.array(
            jnp.asarray([scaled_occupations, self.lows, self.highs]).T
        )
        return apply_fun_grouped(
            bounded_simplex_inputs,
            index=self.atom_ids,
            func=apply_bounded_simplex,
            group_args=(self.max_vacancies, self.min_vacancies),
        )

        # return normalize_occ_vector(
        #     non_normalized_occupations, self.atom_ids, op_type=self.op_type
        # )


def apply_bounded_simplex(in_vecs, max_vacancy, min_vacancy):
    """Apply bounded simplex to a vector."""
    x, lower, upper = in_vecs[:, 0], in_vecs[:, 1], in_vecs[:, 2]
    # max_vacancy, min_vacancy = group_args
    n = x.shape[0]

    vac_parameter = (n - x.sum()) / n
    c_bar = jnp.concatenate([x, vac_parameter.reshape(-1)])
    c_bar_lower = jnp.concatenate([lower, min_vacancy.reshape(-1)])
    c_bar_upper = jnp.concatenate([upper, max_vacancy.reshape(-1)])
    bounded = bounded_softmax_from_unit(
        c_bar,
        c_bar_lower,
        c_bar_upper,
        temperature=1.0,
        tol=EPS,
        max_iter=100,
    )
    # return only the first n elements (the concentrations)
    return bounded[:-1]


def normalize_occ_vector(non_norm_occ_vector, atom_ids, op_type='mirror'):
    r"""Normalize the occupation vector to <=1 for each atom.

    Each scatterer site that can be occupied by multiple chemical species
    whose sum of occupations must be less than or equal to 1.
    This function normalizes the occupation vector for any scatterers that
    would otherwise exceed this limit. Occupations for scatterers that do not
    exceed the limit are left unchanged.
    The normalization is done by projecting the occupation vector onto the
    plane defined by the constraint \sum{c_i}=1, where c_i are the
    concentrations of the elements occupying the scatterer site.

    non_norm_occ_vector and atom_ids must be of the same length. atom_ids must
    be a hashable collection (e.g., a tuple) of integers that represent which
    atoms in the non_norm_occ_vector should be normalized together. This should
    be taken from the atom_ids property of the calculator.

    Parameters
    ----------
    non_norm_occ_vector : array_like
        The occupation vector to be normalized.
    atom_ids : tuple
        The atom IDs for which the occupation vector should be normalized.
        This must be hashable (e.g., a tuple).

    Returns
    -------
    jax.Array
        The normalized occupation vector, where the sum of the elements
        corresponding to the specified atom IDs is equal to 1.
    """
    # raise error is atom_ids is not hashable (most likely a tuple)
    try:
        hash(atom_ids)
    except TypeError as err:
        msg = 'atom_ids must be hashable (e.g., a tuple)'
        raise TypeError(msg) from err

    # select the operation to be used for normalization
    if op_type == 'mirror':
        normalize_func = _normalize_atom_occ_vector_mirror
    elif op_type == 'projection':
        normalize_func = _normalize_atom_occ_vector_projection
    else:
        msg = (
            f'Invalid operation type: {op_type}. '
            'Valid options are "mirror" or "projection".'
        )
        raise ValueError(msg)

    return apply_fun_grouped(
        in_vec=non_norm_occ_vector,
        index=atom_ids,
        func=normalize_func,
    )


def _normalize_atom_occ_vector_mirror(occ_vector):
    """Normalize the occupation vector to sum to 1.

    This function is used to normalize the occupation vector for a single
    atom. It is called by the `normalize_occ_vector` function.
    """
    _occ_vector = jnp.asarray(occ_vector)

    return jax.lax.cond(
        jnp.sum(_occ_vector) <= 1.0,
        lambda vec: vec,
        mirror_across_plane_sum_1,
        operand=_occ_vector,
    )


def _normalize_atom_occ_vector_projection(occ_vector):
    """Normalize the occupation vector to sum to 1.

    This function is used to normalize the occupation vector for a single
    atom. It is called by the `normalize_occ_vector` function.
    """
    _occ_vector = jnp.asarray(occ_vector)

    return jax.lax.cond(
        jnp.sum(_occ_vector) <= 1.0,
        lambda vec: vec,
        project_onto_plane_sum_1,
        operand=_occ_vector,
    )
