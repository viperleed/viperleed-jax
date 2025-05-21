import pytest
import numpy as np

from viperleed_jax.transformation_tree.reduced_space import apply_affine_to_subspace, orthonormalize_subspace, Zonotope
from viperleed_jax.transformation_tree.linear_transformer import AffineTransformer


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

