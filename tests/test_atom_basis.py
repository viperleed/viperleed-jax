"""Tests for the base scatterers."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-25'

import pytest
from tests.calc import poscar_slabs
from tests.calc.tags import CaseTag as Tag
from pytest_cases import parametrize_with_cases, filters

from viperleed_jax.atom_basis import AtomBasis, TargetSelectionError


from viperleed_jax.files.displacements.tokens.target import TargetToken



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


# Fake Atom and SiteElement structure
class FakeAtom:
    def __init__(self, num, label, layer):
        self.num = num
        self.layer = type("Layer", (), {"num": layer})()
        self.site = type("Site", (), {"label": label, "el": label, "mixedEls": []})()
        self.symrefm = None
        self.is_bulk = False

@pytest.fixture
def mock_slab():
    slab = type("FakeSlab", (), {})()
    slab.sitelist = [FakeAtom(0, label, 0).site for label in ['A', 'A_extra', 'B', 'C']]
    slab.atlist = [
        FakeAtom(1, "A", 0),
        FakeAtom(2, "A", 1),
        FakeAtom(3, "A_extra", 2),
        FakeAtom(4, "B", 0),
        FakeAtom(5, "C", 1),
    ]
    slab.linklists = []
    return slab

@pytest.fixture
def atom_basis(mock_slab):
    return AtomBasis(mock_slab)


class TestSelectionMask:
    """Tests for the selection mask creation."""
    @pytest.mark.parametrize(
        'targets, exp_mask',
        [
            # simple label matching
            ([TargetToken('A')], [True, True, True, False, False]),
            ([TargetToken('B')], [False, False, False, True, False]),
            # wildcard postfix
            ([TargetToken('A_*')], [False, False, True, False, False]),
            ([TargetToken('*')], [True, True, True, True, True]),
            # numeric subtarget list
            ([TargetToken('A 1 3')], [True, False, True, False, False]),
            ([TargetToken('A 1-3')], [True, True, True, False, False]),
            # layer selection                                                   # Note that layer labeling is 1-based
            ([TargetToken('A L(1)')], [True, False, False, False, False]),
            ([TargetToken('A L(1-2)')], [True, True, False, False, False]),
            # multiple targets
            (
                [TargetToken('A 1'), TargetToken('B')],
                [True, False, False, True, False],
            ),
            (
                [TargetToken('A L(2)'), TargetToken('C 5')],
                [False, True, False, False, True],
            ),
        ],
        ids=lambda x: str(x),
    )
    def test_target_mask_creation(self, atom_basis, targets, exp_mask):
        mask = atom_basis.selection_mask(targets)
        assert all(mask == exp_mask)


    @pytest.mark.parametrize(
        'targets',
        [
            # simple label matching
            ([TargetToken('A 10')]),
            ([TargetToken('A 3-1')]),
            ([TargetToken('Z')]),
            ([TargetToken('B 2')]),
            ([TargetToken('A_sub')]),
        ],
        ids=lambda x: str(x),
    )
    def test_target_selection_errors(self, atom_basis, targets):
        with pytest.raises(TargetSelectionError):
            atom_basis.selection_mask(targets)
