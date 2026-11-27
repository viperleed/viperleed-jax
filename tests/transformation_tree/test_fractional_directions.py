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
from viperleed.calc.files.displacements import (
    readDISPLACEMENTS,
    readDISPLACEMENTS_block,
)
from viperleed.calc.files.new_displacements.lines import (
    GeoDeltaLine,
    OffsetsLine,
)
from viperleed.calc.files.new_displacements.file import DisplacementsFile

from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.from_state import run_viperleed_initialization
from viperleed_jax.transformation_tree.geo_parameters import GeoTree

from ..viper_init_states import DATA_PATH, INPUTS_PT_111_10x10_TE

# A file of #428's test data: fractional GEO_DELTA directions, one per
# line, for free (ab[1 0]) and for 1D in-plane ([2 1]) atoms.
LATERAL_FILE = (
    DATA_PATH / 'displacements' / 'Pt_111_10x10_Te' / 'DISPLACEMENTS_lateral_1'
)

RANGE = '-0.05 0.15'  # asymmetric: fixes the sign of each vector
OFFSET = 0.1
ATOL = 1e-6
# Atoms of Pt_111_10x10_Te, at most one per symmetry-linked group, by
# the free direction (freedir) of the group.
FREE_ATOMS = (1, 2, 8)  # free in-plane movement
ONE_D_ATOMS = {'[2 1]': 7, '[1 -1]': 10, '[1 2]': 24}  # 1D in-plane

# Offsets of atoms with 1D in-plane freedir are not supported yet.
_NO_1D_OFFSETS = pytest.mark.xfail(
    reason='offsets are not supported for atoms with 1D in-plane freedir',
    raises=ValueError,
    strict=True,
)


# A module-scoped copy of state_pt_111_10x10_te: initialization is the
# slow part, and the tests only read the state (the old reader works on
# deep copies).
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
    return _old_reader_block_points(
        state, (['= GEO_DELTA', line], 'test'), attribute
    )


def _old_reader_block_points(state, block, attribute, rpars=None):
    """Return {atom number: (n, 3) array in zxy} for an old-reader block."""
    rpars = state.rpars if rpars is None else rpars
    slab, rpars = copy.deepcopy((state.slab, rpars))
    readDISPLACEMENTS_block(rpars, slab, block)
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
        tree.apply_bounds_line(
            line if isinstance(line, GeoDeltaLine) else GeoDeltaLine(line)
        )
    tree.apply_bounds()
    tree.apply_implicit_constraints()
    tree.finalize_tree()
    points = {}
    for leaf in tree.leaves:
        transformer = tree.root.transformer_to_descendent(leaf)
        weights = np.asarray(transformer.weights)
        # Only the parameters that move this atom span its corners
        weights = weights[:, np.abs(weights).max(axis=0) > ATOL]
        corners = np.array(
            list(itertools.product((0, 1), repeat=weights.shape[1])),
            dtype=float,
        )
        points[leaf.num] = np.asarray(transformer.biases) + corners @ weights.T
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


# (id, atom numbers, direction in new syntax, old-reader direction)
_CASES = (
    ('a', FREE_ATOMS, 'a', 'ab[1 0]'),
    ('ab[1 2]', FREE_ATOMS, 'ab[1 2]', 'ab[1 2]'),
    ('[2 1]', FREE_ATOMS, '[2 1]', 'ab[2 1]'),
    *(
        (f'1D {freedir}', (num,), f'ab{freedir}', f'ab{freedir}')
        for freedir, num in ONE_D_ATOMS.items()
    ),
)
_GEO_DELTA_CASES = [pytest.param(*case[1:], id=case[0]) for case in _CASES]
_OFFSET_CASES = [
    pytest.param(
        *case[1:],
        id=case[0],
        marks=_NO_1D_OFFSETS if case[0].startswith('1D') else (),
    )
    for case in _CASES
]
_CASE_ARGS = 'nums, direction, old_direction'


@pytest.mark.parametrize(_CASE_ARGS, _GEO_DELTA_CASES)
def test_geo_delta_matches_old_reader(
    state, nums, direction, old_direction
):
    """Fractional GEO_DELTA directions give the old reader's vectors."""
    targets = _targets(state.slab, nums)
    tree_points = _tree_corners(
        state, delta_lines=[f'{targets} {direction} = {RANGE}']
    )
    old_points = _old_reader_points(
        state, f'{targets} {old_direction} = {RANGE} 0.05', 'disp_geo'
    )
    # the old reader lists steps along the range: keep its end points
    old_points = {k: v[[0, -1]] for k, v in old_points.items()}
    _assert_same_points(tree_points, old_points)


@pytest.mark.parametrize(_CASE_ARGS, _OFFSET_CASES)
def test_offset_matches_old_reader(
    state, nums, direction, old_direction
):
    """Fractional OFFSETS directions give the old reader's vectors."""
    targets = _targets(state.slab, nums)
    tree_points = _tree_corners(
        state, offset_line=f'geo {targets} {direction} = {OFFSET}'
    )
    old_points = _old_reader_points(
        state,
        f'{targets} {old_direction} offset = {OFFSET}',
        'disp_geo_offset',
    )
    _assert_same_points(tree_points, old_points)


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


# Cartesian in-plane directions that are not along an axis. Mapped to
# the frames of the symmetry-linked atoms, their components get
# opposite signs, and the ranges must follow the vectors, not the
# extent of the image along the Cartesian axes.
@pytest.mark.parametrize('direction', ('xy[1 1]', 'xy[1 -1]', 'xy[2 -1]'))
def test_cartesian_geo_delta_matches_old_reader(state, direction):
    """Non-axis Cartesian GEO_DELTA directions give the old reader's."""
    targets = _targets(state.slab, FREE_ATOMS)
    line = f'{targets} {direction} = {RANGE}'
    tree_points = _tree_corners(state, delta_lines=[line])
    old_points = _old_reader_points(state, f'{line} 0.05', 'disp_geo')
    old_points = {k: v[[0, -1]] for k, v in old_points.items()}
    _assert_same_points(tree_points, old_points)


def test_geo_delta_file_matches_old_reader(state):
    """A #428 DISPLACEMENTS file gives the old reader's vectors."""
    disp_file = DisplacementsFile()
    disp_file.read(LATERAL_FILE)
    search_block = disp_file.next(2.0)
    tree_points = _tree_corners(
        state, delta_lines=search_block.geo_delta_lines
    )
    rpars = copy.deepcopy(state.rpars)
    readDISPLACEMENTS(rpars, str(LATERAL_FILE))
    old_points = _old_reader_block_points(
        state, rpars.disp_blocks[0], 'disp_geo', rpars=rpars
    )
    old_points = {k: v[[0, -1]] for k, v in old_points.items()}
    _assert_same_points(tree_points, old_points)
