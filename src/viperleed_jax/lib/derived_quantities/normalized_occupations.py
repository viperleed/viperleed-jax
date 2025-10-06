"""Module normalized_occupations."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-06-25'


import numpy as np
from jax import numpy as jnp

from viperleed_jax.lib.bounded_simplex import bounded_softmax_from_unit
from viperleed_jax.lib.math import (
    EPS,
    apply_fun_grouped,
)
from viperleed_jax.transformation_tree.derived_quantities import (
    DerivedQuantitySingleTree,
)
from viperleed_jax.transformation_tree.occ_parameters import (
    OccTotalOccupationConstraint,
)

_MAX_ITERATIONS = 100


class NormalizedOccupations(DerivedQuantitySingleTree):
    """Derived quantity for normalized occupations."""

    def __init__(self, parameter_space):
        super().__init__(parameter_space)
        self.name = 'normalized_occupations'

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
        max_vacancies = 1 - jnp.array(sum_group_min_occs)
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


def apply_bounded_simplex(in_vecs, max_vacancy, min_vacancy):
    """
    Project a group's occupation vector onto a bounded simplex.

    Uses a "slack-variable" augmentation and a bounded softmax map.

    Given per-atom lower/upper bounds and a per-group min/max vacancy
    allowance, this function appends one vacancy parameter to the
    concentration vector and maps the (n+1)-vector through
    ``bounded_softmax_from_unit`` such that the returned first n entries
    (the concentrations) satisfy both the box constraints and the
    simplex-like sum constraint implied by the vacancy.

    Parameters
    ----------
    in_vecs : jax.Array
        2D array of shape (n, 3) holding, **per row**, the triplet
        ``[x_i, lower_i, upper_i]``:
        - ``x``: unconstrained (pre-normalized) concentrations (shape (n,))
        - ``lower``: per-element lower bounds (shape (n,))
        - ``upper``: per-element upper bounds (shape (n,))
        Internally this function reads columns via
        ``x = in_vecs[:, 0]``, ``lower = in_vecs[:, 1]``,
        ``upper = in_vecs[:, 2]``.
        If any `lower_i == upper_i`, that concentration is fixed to that value.
    max_vacancy : float or jax.Array
        Upper bound for the group's vacancy parameter (scalar or 0-D array).
        The effective vacancy after mapping will not exceed this value.
        If equal `min_vacancy`, the vacancy is fixed.
    min_vacancy : float or jax.Array
        Lower bound for the group's vacancy parameter (scalar or 0-D array).
        The effective vacancy after mapping will be at least this value.

    Returns
    -------
    bounded_x : jax.Array
        1D array of shape (n,) with concentrations that:
        - respect per-element bounds ``lower <= bounded_x <= upper``
        - together with the implied vacancy sum to 1 (within numerical tolerance).

    Notes
    -----
    - The augmented vector is constructed as:
      ``c_bar = concat([x, v])`` with ``v = (n - sum(x)) / n`` (a slack vacancy).
      Bounds are augmented analogously with ``[lower, min_vacancy]`` and
      ``[upper, max_vacancy]``.
    - The core mapping is delegated to ``bounded_softmax_from_unit``, using
      globals ``EPS`` and ``_MAX_ITERATIONS`` for numerical stability and
      iteration control.
    - ``max_vacancy`` / ``min_vacancy`` are assumed to be scalars for the
      **group** (not per element). If they arrive as 0-D arrays, they are
      reshaped to length-1 vectors for concatenation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.array([0.6, 0.6, 0.1])
    >>> lo = jnp.array([0.0, 0.0, 0.0])
    >>> hi = jnp.array([1.0, 1.0, 1.0])
    >>> in_vecs = jnp.stack([x, lo, hi], axis=1)  # shape (3,3)
    >>> apply_bounded_simplex(in_vecs, max_vacancy=0.5, min_vacancy=0.0).shape
    (3,)
    """
    x, lower, upper = in_vecs[:, 0], in_vecs[:, 1], in_vecs[:, 2]
    n = x.shape[0]

    # slack "vacancy" before mapping
    vac_parameter = (n - x.sum()) / n

    # augment variables and bounds with the vacancy slot
    c_bar = jnp.concatenate([x, vac_parameter.reshape(-1)])
    c_bar_lower = jnp.concatenate([lower, jnp.asarray(min_vacancy).reshape(-1)])
    c_bar_upper = jnp.concatenate([upper, jnp.asarray(max_vacancy).reshape(-1)])

    bounded = bounded_softmax_from_unit(
        c_bar,
        c_bar_lower,
        c_bar_upper,
        temperature=1.0,
        tol=EPS,
        max_iter=_MAX_ITERATIONS,
    )
    # return the concentrations (c_i) only, drop the vacancy slot
    return bounded[:-1]
