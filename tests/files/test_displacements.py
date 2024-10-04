import pytest

from viperleed_jax.files.displacements import match_geo_line

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
