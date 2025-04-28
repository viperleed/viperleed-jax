"""Library for tensor_calculator."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-04-28'

import jax
from jax import numpy as jnp

from ..lib_math import apply_fun_grouped, project_onto_plane_sum_1


def normalize_occ_vector(non_norm_occ_vector, atom_ids):
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

    return apply_fun_grouped(
        in_vec=non_norm_occ_vector,
        index=atom_ids,
        func=_normalize_atom_occ_vector,
    )


def _normalize_atom_occ_vector(occ_vector):
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
