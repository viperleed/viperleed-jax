import pytest
import numpy as np

from viperleed_jax.transformation_tree.reduced_space import apply_affine_to_subspace, orthonormalize_subspace
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
