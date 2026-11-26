"""Tests for fractional directions (a, ab, ab[i j], [i j]) in the geo tree.

The old DISPLACEMENTS reader is the oracle: for each atom, the vectors
the geo tree reaches must be the ones the old reader assigns, including
atoms that are only symmetry linked to a target.
"""

__authors__ = ('Florian Kraushofer (@fkraushofer)',)
__created__ = '2026-11-26'

import copy
import itertools

import numpy as np
import pytest
from viperleed.calc.files.displacements import readDISPLACEMENTS_block
from viperleed.calc.files.new_displacements.lines import (
    GeoDeltaLine,
    OffsetsLine,
)

from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.from_state import run_viperleed_initialization
from viperleed_jax.transformation_tree.geo_parameters import GeoTree

from ..viper_init_states import INPUTS_PT_111_10x10_TE

RANGE = '-0.05 0.15'  # asymmetric: fixes the sign of each vector
OFFSET = 0.1
ATOL = 1e-6
# Atoms of Pt_111_10x10_Te, at most one per symmetry-linked group, by
# the free direction (freedir) of the group.
FREE_ATOMS = (1, 2, 8)  # free in-plane movement
ONE_D_ATOMS = {'[2 1]': 7, '[1 -1]': 10, '[1 2]': 24}  # 1D in-plane

# Ranges are mapped between the frames of symmetry-linked atoms with
# a wrong scale and sign when the direction is not along a Cartesian
# axis of the reference atom, and offsets of atoms with 1D in-plane
# freedir are not supported.
_WRONG_RANGE_MAPPING = pytest.mark.xfail(
    reason='geo ranges are mapped incorrectly between linked atoms',
    strict=True,
)
_NO_1D_OFFSETS = pytest.mark.xfail(
    reason='offsets are not supported for atoms with 1D in-plane freedir',
    raises=ValueError,
    strict=True,
)


@pytest.fixture(scope='module')
def state():
    """Return the initialized Pt(111) slab, with hexagonal ab cell."""
    return run_viperleed_initialization(INPUTS_PT_111_10x10_TE)


def _targets(slab, nums):
    """Return a DISPLACEMENTS target for atoms `nums`, per element."""
    els = {slab.atlist[num - 1].el for num in nums}
    assert len(els) == 1
    return f'{els.pop()} {" ".join(str(num) for num in nums)}'


def _old_reader_points(state, line, attribute):
    """Return {atom number: (n, 3) array in zxy} read by the old reader."""
    slab, rpars = copy.deepcopy((state.slab, state.rpars))
    readDISPLACEMENTS_block(rpars, slab, (['= GEO_DELTA', line], 'test'))
    points = {}
    for atom in slab.atlist:
        if atom.is_bulk:
            continue
        vectors = np.array(getattr(atom, attribute).get('all', [[0, 0, 0]]))
        points[atom.num] = vectors[:, [2, 0, 1]]  # xyz -> zxy
    return points


def _tree_corners(state, offset_line=None, delta_lines=()):
    """Return {atom number: (n, 3) corners of the reachable box} (zxy)."""
    tree = GeoTree(AtomBasis(state.slab))
    if offset_line:
        tree.apply_offsets(OffsetsLine(offset_line))
    for line in delta_lines:
        tree.apply_bounds_line(GeoDeltaLine(line))
    tree.apply_bounds()
    tree.apply_implicit_constraints()
    tree.finalize_tree()
    corners = list(itertools.product((0, 1), repeat=tree.root.dof))
    points = {}
    for leaf in tree.leaves:
        transformer = tree.root.transformer_to_descendent(leaf)
        points[leaf.num] = np.array(
            [np.asarray(transformer(np.array(c))) for c in corners]
        )
    return points


def _assert_same_points(tree_points, old_points):
    """Check that two sets of points per atom agree (as sets)."""
    assert tree_points.keys() == old_points.keys()
    for num, tree_pts in tree_points.items():
        dist = np.linalg.norm(
            tree_pts[:, None, :] - old_points[num][None, :, :], axis=-1
        )
        assert dist.min(axis=1).max() < ATOL, f'atom {num}'
        assert dist.min(axis=0).max() < ATOL, f'atom {num}'


def _cases(slab):
    """Yield (id, targets, direction in new syntax, old-reader direction)."""
    free = _targets(slab, FREE_ATOMS)
    yield 'a', free, 'a', 'ab[1 0]'
    yield 'ab[1 2]', free, 'ab[1 2]', 'ab[1 2]'
    yield '[2 1]', free, '[2 1]', 'ab[2 1]'
    for freedir, num in ONE_D_ATOMS.items():
        direction = f'ab{freedir}'
        yield f'1D {freedir}', _targets(slab, [num]), direction, direction


# Case names are fixed by FREE_ATOMS and ONE_D_ATOMS: they must be known
# at collection, without initializing a slab.
_CASE_NAMES = ['a', 'ab[1 2]', '[2 1]'] + [f'1D {f}' for f in ONE_D_ATOMS]
_GEO_DELTA_CASE_NAMES = [
    pytest.param(
        name,
        marks=(
            ()
            if name in {'1D [1 -1]', '1D [1 2]'}  # not hit by the mapping
            else _WRONG_RANGE_MAPPING
        ),
    )
    for name in _CASE_NAMES
]
_OFFSET_CASE_NAMES = [
    pytest.param(name, marks=_NO_1D_OFFSETS if name.startswith('1D') else ())
    for name in _CASE_NAMES
]


def _case(slab, case_name):
    """Return targets, direction and old-reader direction of a case."""
    return next(c[1:] for c in _cases(slab) if c[0] == case_name)


@pytest.mark.parametrize('case_name', _GEO_DELTA_CASE_NAMES)
def test_geo_delta_matches_old_reader(state, case_name):
    """Fractional GEO_DELTA directions give the old reader's vectors."""
    targets, direction, old_direction = _case(state.slab, case_name)
    tree_points = _tree_corners(
        state, delta_lines=[f'{targets} {direction} = {RANGE}']
    )
    old_points = _old_reader_points(
        state, f'{targets} {old_direction} = {RANGE} 0.05', 'disp_geo'
    )
    # the old reader lists steps along the range: keep its end points
    old_points = {k: v[[0, -1]] for k, v in old_points.items()}
    _assert_same_points(tree_points, old_points)


@pytest.mark.parametrize('case_name', _OFFSET_CASE_NAMES)
def test_offset_matches_old_reader(state, case_name):
    """Fractional OFFSETS directions give the old reader's vectors."""
    targets, direction, old_direction = _case(state.slab, case_name)
    tree_points = _tree_corners(
        state, offset_line=f'geo {targets} {direction} = {OFFSET}'
    )
    old_points = _old_reader_points(
        state,
        f'{targets} {old_direction} offset = {OFFSET}',
        'disp_geo_offset',
    )
    _assert_same_points(tree_points, old_points)


@_WRONG_RANGE_MAPPING
def test_geo_delta_ab_plane_matches_old_reader(state):
    """The ab plane is spanned by the old reader's a and perpendicular."""
    targets = _targets(state.slab, FREE_ATOMS)
    tree_points = _tree_corners(
        state, delta_lines=[f'{targets} ab = {RANGE}']
    )
    # The old reader has no 2D fractional direction: combine one 1D line
    # along a with one along the in-plane perpendicular to a (= a + 2 b
    # for a hexagonal cell).
    along_a, perp_a = (
        _old_reader_points(
            state, f'{targets} ab[{i} {j}] = {RANGE} 0.05', 'disp_geo'
        )
        for i, j in ((1, 0), (1, 2))
    )
    old_points = {
        num: (
            along_a[num][[0, -1]][:, None] + perp_a[num][[0, -1]][None]
        ).reshape(-1, 3)
        for num in along_a
    }
    _assert_same_points(tree_points, old_points)

