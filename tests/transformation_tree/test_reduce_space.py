import numpy as np
import pytest

from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)
from viperleed_jax.transformation_tree.reduced_space import (
    Zonotope,
    apply_affine_to_subspace,
    orthonormalize_subspace,
)


@pytest.mark.parametrize(
    'B,ranges,transform,expected_Bm,expected_ranges,expected_bias',
    [
        # identity 2D
        (
            np.eye(2),
            np.array([[0, 0], [1, 1]]),
            AffineTransformer(np.eye(2), np.zeros(2)),
            np.eye(2),
            np.array([[0, 0], [1, 1]]),
            np.zeros(2),
        ),
        # collapse x-y mapping
        (
            np.eye(2),
            np.array([[0, 0], [1, 1]]),
            AffineTransformer(np.array([[1, -1], [0, 0]]), np.zeros(2)),
            np.array([[1, -1], [0, 0]]),
            np.array([[0, 0], [1, 0]]),
            np.zeros(2),
        ),
    ],
)
def test_apply_affine_to_subspace(
    B, ranges, transform, expected_Bm, expected_ranges, expected_bias
):
    Bm, mr, b = apply_affine_to_subspace(B, ranges, transform)
    assert np.allclose(Bm, expected_Bm)
    assert np.allclose(mr, expected_ranges)
    assert np.allclose(b, expected_bias)


# ---- Tests for orthonormalize_subspace ----
@pytest.mark.parametrize(
    'Bm,mr,bias,expected_W,expected_b',
    [
        # full-rank identity -> scale 2x, shift -1
        (
            np.eye(2),
            np.array([[-1, -1], [1, 1]]),
            np.zeros(2),
            2 * np.eye(2),
            np.array([-1, -1]),
        ),
        # zero-rank collapse -> bias-only
        (
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.array([5.0, -3.0]),
            np.zeros((2, 0)),
            np.array([5.0, -3.0]),
        ),
    ],
)
def test_orthonormalize_subspace(Bm, mr, bias, expected_W, expected_b):
    trafo = orthonormalize_subspace(Bm, mr, bias, output_ranges=None)
    assert trafo.weights.shape == expected_W.shape
    assert np.allclose(trafo.weights, expected_W)
    assert np.allclose(trafo.biases, expected_b)


# --- Tests for Zonotope.__init__ and properties ---
@pytest.mark.parametrize(
    'basis,ranges,offset,expect_error',
    [
        (
            np.eye(3),
            np.array([[0, 0], [1, 1]]),
            None,
            True,
        ),  # ranges wrong shape
        (
            np.ones((3,)),
            np.array([[-1, -1, -1], [1, 1, 1]]),
            None,
            True,
        ),  # basis not 2D
        (
            np.eye(2),
            np.array([[0, 0], [1, 1]]),
            np.zeros(3),
            True,
        ),  # offset wrong length
        (
            np.eye(4),
            np.array([[-1, -1, -1, -1], [1, 1, 1, 1]]),
            None,
            False,
        ),  # valid
    ],
)
def test_init_and_properties(basis, ranges, offset, expect_error):
    if expect_error:
        with pytest.raises(ValueError):
            Zonotope(basis, ranges, offset)
    else:
        Z = Zonotope(basis, ranges, offset)
        D, n = basis.shape
        assert Z.dim == D
        assert Z.order == n


# --- Tests for apply_affine ---


def test_apply_affine_identity():
    B = np.eye(3)
    ranges = np.array([[0, 0, 0], [1, 1, 1]])
    Z = Zonotope(B, ranges)
    ident = AffineTransformer(np.eye(3), np.zeros(3))
    Z2 = Z.apply_affine(ident)
    # should be unchanged
    assert isinstance(Z2, Zonotope)
    assert np.allclose(Z2.basis, B)
    assert np.allclose(Z2.ranges, ranges)
    assert np.allclose(Z2.offset, 0)


@pytest.mark.parametrize(
    'W,expected_Bm_shape,expected_offset',
    [
        (np.array([[1, -1, 0], [0, 0, 1]]), (2, 3), np.zeros(2)),
        (np.zeros((2, 3)), (2, 3), np.array([5.0, -3.0])),
    ],
)
def test_apply_affine_variations(W, expected_Bm_shape, expected_offset):
    B = np.eye(3)
    ranges = np.array([[0, 0, 0], [1, 1, 1]])
    bias = expected_offset
    A = AffineTransformer(W, bias)
    Z = Zonotope(B, ranges)
    Z2 = Z.apply_affine(A)
    # basis shape
    assert Z2.basis.shape == expected_Bm_shape
    # range shape unchanged
    assert Z2.ranges.shape == (2, 3)
    # offset matches
    assert np.allclose(Z2.offset, bias)


# --- Tests for normalize ---


def test_normalize_identity():
    B = np.eye(2)
    ranges = np.array([[0, 0], [1, 1]])
    Z = Zonotope(B, ranges)
    T = Z.normalize()
    # identity mapping
    assert isinstance(T, AffineTransformer)
    assert T.in_dim == 2
    assert T.out_dim == 2
    assert np.allclose(T.weights, np.eye(2))
    assert np.allclose(T.biases, 0)


def test_normalize_collapse():
    B = np.eye(3)
    ranges = np.array([[-1, -1, -1], [1, 1, 1]])
    W = np.array([[1, 1, 0], [0, 0, 1]])  # collapse dims 0&1 into first output
    A = AffineTransformer(W, np.zeros(2))
    Z2 = Zonotope(B, ranges).apply_affine(A)
    T = Z2.normalize(output_ranges=[0, 1])
    # Bm rank is 2 => T.in_dim=2, T.out_dim=2
    assert T.in_dim == 2
    assert T.out_dim == 2
    # weights should be orthonormal: W.T @ W == I
    assert np.allclose(T.weights.T @ T.weights, np.eye(2), atol=1e-8)

# --- Tests for is_orthogonal_to ---


@pytest.mark.parametrize(
    'B1, B2, expected',
    [
        # Orthogonal: e1 ⟂ span{e2, e3}
        (np.array([[1, 0, 0]]).T, np.array([[0, 1, 0], [0, 0, 1]]).T, True),
        # Orthogonal despite non-orthonormal columns: [1,1] ⟂ [1,-1]
        (np.array([[1, 1]]).T, np.array([[1, -1]]).T, True),
        # Not orthogonal: e1 ⋂ span{e1,e2} ≠ {0}
        (np.array([[1, 0]]).T, np.array([[1, 1]]).T, False),
    ],
)
def test_is_orthogonal_to_basic(B1, B2, expected):
    ranges1 = np.array([[-1] * B1.shape[1], [1] * B1.shape[1]])
    ranges2 = np.array([[-2] * B2.shape[1], [3] * B2.shape[1]])
    Z1 = Zonotope(B1, ranges1)
    Z2 = Zonotope(B2, ranges2)
    assert Z1.is_orthogonal_to(Z2) is expected
    # symmetry
    assert Z2.is_orthogonal_to(Z1) is expected


def test_is_orthogonal_to_empty_subspace_returns_true():
    # Make Z1 with zero generators after range filtering (all zero-width ranges)
    B1 = np.eye(3)
    ranges1 = np.array([[0, 0, 0], [0, 0, 0]])  # removes all columns
    Z1 = Zonotope(B1, ranges1)

    # Any non-empty Z2
    B2 = np.array([[0, 1, 0]]).T
    ranges2 = np.array([[-1], [1]])
    Z2 = Zonotope(B2, ranges2)

    assert Z1.is_orthogonal_to(Z2) is True
    assert Z2.is_orthogonal_to(Z1) is True


def test_is_orthogonal_to_ranges_irrelevant():
    # Same bases, wildly different ranges -> same orthogonality result
    B1 = np.array([[1, 0, 0], [0, 1, 0]]).T  # e1, e2 in R3
    B2 = np.array([[0, 0, 1]]).T  # e3

    Z1a = Zonotope(B1, np.array([[-1, -1], [1, 1]]))
    Z1b = Zonotope(
        B1, np.array([[2, 0], [2, 0]])
    )  # second column removed -> 1D e1
    Z2 = Zonotope(B2, np.array([[-5], [7]]))

    # e1,e2 ⟂ e3  and e1 ⟂ e3  -> both True
    assert Z1a.is_orthogonal_to(Z2) is True
    assert Z1b.is_orthogonal_to(Z2) is True


def test_is_orthogonal_to_type_error():
    Z = Zonotope(np.eye(2), np.array([[-1, -1], [1, 1]]))
    with pytest.raises(TypeError):
        Z.is_orthogonal_to(object())


# --- Tests for add_orthogonal_same_center ---


def test_add_orthogonal_same_center_basic():
    # Z1: span{e1}, Z2: span{e2}; same center
    B1 = np.array([[1, 0, 0]]).T
    B2 = np.array([[0, 1, 0]]).T
    ranges1 = np.array([[-1], [1]])
    ranges2 = np.array([[0], [2]])
    c = np.array([0.5, -1.0, 3.0])

    Z1 = Zonotope(B1, ranges1, offset=c)
    Z2 = Zonotope(B2, ranges2, offset=c)
    Z = Z1.add_orthogonal_same_center(Z2)

    # center preserved
    assert np.allclose(Z.offset, c)
    # generators concatenated
    assert Z.basis.shape == (3, 2)
    assert np.allclose(Z.basis[:, 0], B1[:, 0])
    assert np.allclose(Z.basis[:, 1], B2[:, 0])
    # ranges stacked in order
    assert Z.ranges.shape == (2, 2)
    assert np.allclose(Z.ranges[:, 0], ranges1[:, 0])
    assert np.allclose(Z.ranges[:, 1], ranges2[:, 0])


def test_add_orthogonal_same_center_non_orthogonal_raises():
    # Z1: span{e1}, Z2: span{e1} (not orthogonal)
    B1 = np.array([[1, 0]]).T
    B2 = np.array([[1, 0]]).T
    r = np.array([[-1], [1]])
    c = np.array([0.0, 0.0])

    Z1 = Zonotope(B1, r, offset=c)
    Z2 = Zonotope(B2, r, offset=c)
    with pytest.raises(ValueError, match='not orthogonal'):
        _ = Z1.add_orthogonal_same_center(Z2)


def test_add_orthogonal_same_center_center_mismatch_raises():
    B1 = np.array([[1, 0, 0]]).T
    B2 = np.array([[0, 1, 0]]).T
    r1 = np.array([[-1], [1]])
    r2 = np.array([[0], [2]])
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([0.0, 0.0, 1e-6])  # small but detectable if tol < 1e-6

    Z1 = Zonotope(B1, r1, offset=c1)
    Z2 = Zonotope(B2, r2, offset=c2)
    with pytest.raises(ValueError, match='centers.*differ'):
        _ = Z1.add_orthogonal_same_center(Z2)


def test_add_orthogonal_same_center_empty_generators_ok():
    # One side empty (all zero-width ranges) + orthogonal other → still fine
    B1 = np.eye(2)
    r1 = np.array([[0, 0], [0, 0]])  # removes both generators
    B2 = np.array([[0, 1]]).T
    r2 = np.array([[-2], [3]])
    c = np.array([1.0, -2.0])

    Z1 = Zonotope(B1, r1, offset=c)
    Z2 = Zonotope(B2, r2, offset=c)
    Z = Z1.add_orthogonal_same_center(Z2)

    assert np.allclose(Z.offset, c)
    assert Z.order == 1
    assert np.allclose(Z.basis, B2)
    assert np.allclose(Z.ranges, r2)


def test_add_orthogonal_same_center_type_and_dim_checks():
    Z = Zonotope(np.eye(2), np.array([[-1, -1], [1, 1]]), offset=np.zeros(2))
    with pytest.raises(TypeError):
        _ = Z.add_orthogonal_same_center(object())

    Z3 = Zonotope(
        np.eye(3), np.array([[-1, -1, -1], [1, 1, 1]]), offset=np.zeros(3)
    )
    with pytest.raises(ValueError, match='dimension mismatch'):
        _ = Z.add_orthogonal_same_center(Z3)