"""Tests for fixed values in DISPLACEMENTS, which are not supported yet."""

__authors__ = ('Florian Kraushofer (@fkraushofer)',)
__created__ = '2026-11-26'

import pytest
from pytest_cases import fixture, parametrize_with_cases
from viperleed.calc.files.new_displacements.lines import (
    ConstraintLine,
    GeoDeltaLine,
    OccDeltaLine,
    VibDeltaLine,
)

from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.transformation_tree.geo_parameters import GeoTree
from viperleed_jax.transformation_tree.occ_parameters import OccTree
from viperleed_jax.transformation_tree.vib_parameters import VibTree

from ..structures import CaseStatesAfterInit


@fixture
@parametrize_with_cases('case', cases=CaseStatesAfterInit.case_pt25rh75_o_3x1)
def atom_basis_ptrh(case):
    state, _ = case
    return AtomBasis(state.slab)


@pytest.mark.parametrize(
    'tree_cls, line',
    (
        (GeoTree, 'geo O 1 = 0.05'),
        (VibTree, 'vib O 1 = 0.05'),
        (OccTree, 'occ O 1 = O 0.8'),
        (GeoTree, 'geo O 1 = ind(2)'),
    ),
)
def test_fixed_value_constraint_not_supported(atom_basis_ptrh, tree_cls, line):
    """Fixed values/indices give a clear error rather than crashing."""
    tree = tree_cls(atom_basis_ptrh)
    with pytest.raises(NotImplementedError, match='Fixed-value constraints'):
        tree.apply_explicit_constraint(ConstraintLine(line))


@pytest.mark.parametrize(
    'tree_cls, line_cls, line',
    (
        (GeoTree, GeoDeltaLine, 'O 1 z = 0.05 0.05'),
        (VibTree, VibDeltaLine, 'O 1 = 0.05'),
        (VibTree, VibDeltaLine, 'O 1 = 0.05 0.05'),
        (OccTree, OccDeltaLine, 'Me* L(1-3) = Rh 0.5, Pt 0.5'),
        (OccTree, OccDeltaLine, 'Me* L(1-3) = Rh 0.5 0.5, Pt 0.5 0.5'),
        (OccTree, OccDeltaLine, 'Me* L(1-3) = Rh 0.2 0.8, Pt 0.5'),
    ),
)
def test_single_value_delta_not_supported(
    atom_basis_ptrh, tree_cls, line_cls, line
):
    """A single value (or start == stop) gives a clear error.

    Without the error, the zero-width range would be dropped and the
    line silently ignored.
    """
    tree = tree_cls(atom_basis_ptrh)
    with pytest.raises(NotImplementedError, match='Fixed values are not'):
        tree.apply_bounds_line(line_cls(line))
