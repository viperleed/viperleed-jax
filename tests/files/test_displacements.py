import pytest

from viperleed_jax.files.displacements import SEARCH_HEADER_PATTERN
from viperleed_jax.files.displacements import SECTION_HEADER_PATTERN
from viperleed_jax.files.displacements import match_geo_line
from viperleed_jax.files.displacements import match_vib_line

# Test cases for GEO_LINE_PATTERN
TEST_LINES_GEOMETRY = {
    "Ir L(1-6) z = -0.05 0.05 0.01": ('Ir', 'L(1-6)', 'z', -0.05, 0.05, 0.01),
    "Si x = -0.005 0.005 0.0005": ('Si', None, 'x', -0.005, 0.005, 0.0005),
    "Ir L(1-6) xy[1 0] = -0.03 0.03 0.01": ('Ir', 'L(1-6)', 'xy[1 0]', -0.03, 0.03, 0.01),
    "Abc L(1-6) xy = -0.03 0.03 0.01": ('Abc', 'L(1-6)', 'xy', -0.03, 0.03, 0.01),
    "Ir L(1-6) z = -0.03 0.03 0.01": ('Ir', 'L(1-6)', 'z', -0.03, 0.03, 0.01),
    "Au 1 2 4 x = -0.03 0.03 0.01": ('Au', '1 2 4', 'x', -0.03, 0.03, 0.01),
    "B 1 2 4 abc = -0.03 0.03 0.01" : ('B', '1 2 4', 'abc', -0.03, 0.03, 0.01),
    "Cd L(1-6) azi(ab[c1 c2]) = -0.03 0.03 0.01" : ('Cd', 'L(1-6)', 'azi(ab[c1 c2])', -0.03, 0.03, 0.01),
    "E 5 ab[n1 n2] = -0.005 0": ('E', '5', 'ab[n1 n2]', -0.005, 0., None),
    "F 1 3 xy[0 1] = 0.2": ('F', '1 3', 'xy[0 1]', 0.2, None, None),
}

# Test cases for SECTION_HEADER_PATTERN
TEST_LINES_SECTION = {
    "GEO_DELTA": "GEO_DELTA",
    "VIB_DELTA": "VIB_DELTA",
    "OCC_DELTA": "OCC_DELTA",
    "CONSTRAIN": "CONSTRAIN",
    "NOT_A_SECTION": None,
    "GEO_DELTA_EXTRA": None,
}

# Test cases for SEARCH_HEADER_PATTERN
TEST_LINES_SEARCH = {
    "== SEARCH label1": ("label1",),
    "== search another_label": ("another_label",),
    "== SEARCH complicated_label-123": ("complicated_label-123",),
    "== SEARCH With_Spaces In_Label": ("With_Spaces In_Label",),
}

# Test cases for VIB_DELTA lines
TEST_LINES_VIB = {
    "O 1 = -0.05 0.05 0.02": ('O', '1', -0.05, 0.05, 0.02),
    "Ir_top = -0.05 0.05 0.01": ('Ir_top', None, -0.05, 0.05, 0.01),
    "O 1 = 0.02": ('O', '1', 0.02, None, None),  # Single value offset
    "Si = 0.1": ('Si', None, 0.1, None, None),  # Single value offset without which
    "H 5 = -0.03 0.03": ('H', '5', -0.03, 0.03, None),  # No step
    "C L(1-4) = -0.1 0.1 0.05": ('C', 'L(1-4)', -0.1, 0.1, 0.05),  # With L(1-4)
    "Mn L(2-3) = 0.0": ('Mn', 'L(2-3)', 0.0, None, None),  # Single value with L-range
}

@pytest.mark.parametrize("input, expected", TEST_LINES_GEOMETRY.items(),
                         ids=TEST_LINES_GEOMETRY.keys())
def test_geo_line_regex(input, expected):
    """Check that the regex for the geometry line works as expected."""
    match = match_geo_line(input)
    assert match is not None
    label, which, dir, start, stop, step = match
    e_label, e_which, e_dir, e_start, e_stop, e_step = expected
    assert label == e_label
    assert which == e_which or e_which is None  # optional
    assert dir == e_dir
    assert start == pytest.approx(e_start)
    assert stop == pytest.approx(e_stop)
    assert step == pytest.approx(e_step) or step is None  # optional

@pytest.mark.parametrize("input, expected", TEST_LINES_SEARCH.items(), ids=TEST_LINES_SEARCH.keys())
def test_search_header_regex(input, expected):
    """Check that the regex for the search header works as expected."""
    match = SEARCH_HEADER_PATTERN.match(input)
    assert match is not None
    assert match.group(1) == expected[0]

@pytest.mark.parametrize("input, expected", TEST_LINES_SECTION.items(), ids=TEST_LINES_SECTION.keys())
def test_section_header_regex(input, expected):
    """Check that the regex for the section header works as expected."""
    match = SECTION_HEADER_PATTERN.match(input)
    if expected is None:
        assert match is None
        return
    assert match is not None
    assert match.group(1) == expected

@pytest.mark.parametrize("input, expected", TEST_LINES_VIB.items(), ids=TEST_LINES_VIB.keys())
def test_vib_line_regex(input, expected):
    """Check that the regex for VIB_DELTA lines works as expected."""
    match = match_vib_line(input)
    assert match is not None
    label, which, start, stop, step = match
    e_label, e_which, e_start, e_stop, e_step = expected
    assert label == e_label
    assert which == e_which or e_which is None  # Optional
    assert start == pytest.approx(e_start)
    assert stop == pytest.approx(e_stop) or stop is None
    assert step == pytest.approx(e_step) or step is None  # Optional
