"""Tests for the base scatterers."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-25'

import pytest
from tests.calc import poscar_slabs
from tests.calc.tags import CaseTag as Tag
from pytest_cases import parametrize_with_cases

from viperleed_jax.atom_basis import Atom, AtomBasis

infoless = parametrize_with_cases(
    'test_slab', cases=poscar_slabs, has_tag=Tag.NO_INFO
)


@infoless
def test_atom_basis_from_slab(test_slab):
    slab, *_ = test_slab
    atom_basis = AtomBasis(slab)
    assert len(atom_basis) > 0
