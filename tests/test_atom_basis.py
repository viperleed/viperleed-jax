"""Tests for the base scatterers."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-25'

import pytest
from tests.calc import poscar_slabs
from tests.calc.tags import CaseTag as Tag
from pytest_cases import parametrize_with_cases, filters

from viperleed_jax.atom_basis import Atom, AtomBasis

test_poscars = parametrize_with_cases(
    'test_slab',
    cases=poscar_slabs,
    filter=(filters.has_tag(Tag.NO_INFO) & (~filters.has_tag(Tag.BULK))),
)

@test_poscars
def test_atom_basis_from_slab(test_slab):

    slab, *_ = test_slab
    if all(at.is_bulk for at in slab.atlist):
        pytest.skip('Bulk slab, skipping test.')

    # check if the atom basis can be created
    atom_basis = AtomBasis(slab)

    # check if the atom basis contains the correct number of atoms
    assert len(atom_basis) > 0
