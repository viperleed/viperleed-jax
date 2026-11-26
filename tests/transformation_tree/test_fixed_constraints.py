"""Tests for fixed-value CONSTRAIN lines, which are not supported yet."""

__authors__ = ('Florian Kraushofer (@fkraushofer)',)
__created__ = '2026-11-26'

import pytest
from pytest_cases import fixture, parametrize_with_cases
from viperleed.calc.files.new_displacements.lines import ConstraintLine

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
