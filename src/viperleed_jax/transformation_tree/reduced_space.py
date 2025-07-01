import numpy as np

from viperleed_jax.lib.math import EPS
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)


def apply_affine_to_subspace(
    basis_vectors: np.ndarray,
    coordinate_ranges: np.ndarray,
    transform: AffineTransformer,
):
    """
    Apply an affine map to a subspace box.

    Parameters
    ----------
    basis_vectors : (n, D) array
        Columns are basis vectors spanning subspace in R^D.
    coordinate_ranges : (2, n) array
        [lows; highs] of coefficients.
    transform : AffineTransformer
        Map R^D→R^M.

    Returns
    -------
    Bm : (M, n) array
        Transformed basis vectors.
    new_ranges : (2, n) array
        Exact min/max of each original coefficient in new basis.
    bias : (M,) array
        The transform.biases.
    """
    D, n = basis_vectors.shape
    if coordinate_ranges.shape != (2, n):
        msg = (
            f'coordinate_ranges must be (2,{n}), got {coordinate_ranges.shape}'
        )
        raise ValueError(msg)
    lows, highs = coordinate_ranges

    W, b = transform.weights, transform.biases
    # project basis correctly:
    Bm = W @ basis_vectors  # -> (M, n)

    # compute new ranges for each coeff
    # (using brute-force corners if n small, but closed form here)
    # we only need per-axis extrema of linear functionals v·x + b
    # Here v_j = j-th column of pseudoinv(Bm) but easier: use SVD root basis
    # For simplicity reuse closed-form per-direction on Bm directly:
    new_mins = np.empty(n)
    new_maxs = np.empty(n)
    # invert: for each basis col i, coordinate xi ranges in [lows[i], highs[i]]
    # z = Bm @ e_i * xi  => extremum at xi boundaries
    for i in range(n):
        vec = Bm[:, i]  # M-vector
        low_pt = vec * lows[i] + b
        high_pt = vec * highs[i] + b
        all_edges = np.vstack([low_pt, high_pt])
        new_mins[i] = all_edges.min()
        new_maxs[i] = all_edges.max()  # since lows[i] ≤ highs[i]
    new_ranges = np.vstack([new_mins, new_maxs])

    return Bm, new_ranges, b


def orthonormalize_subspace(
    Bm: np.ndarray,
    mapped_ranges: np.ndarray,
    bias: np.ndarray,
    output_ranges: np.ndarray = None,
) -> AffineTransformer:
    """
    Given transformed basis Bm and ranges, extract minimal orthonormal
    subspace and normalize coords into output_ranges.

    Returns an AffineTransformer mapping the normalized coords (r,)→R^M.
    """
    M, n = Bm.shape
    if mapped_ranges.shape[0] != 2 or mapped_ranges.shape[1] != n:
        raise ValueError('mapped_ranges must be shape (2,n)')
    lows, highs = mapped_ranges
    # SVD collapse
    U, S, Vh = np.linalg.svd(Bm, full_matrices=False)
    r = int((S > EPS).sum())
    if r == 0:
        return AffineTransformer(np.zeros((M, 0)), bias, out_reshape=(M,))
    Q = U[:, :r]
    # compute pre-range in coords u = Q^T(z)
    V = Q.T @ Bm
    br = Q.T @ bias
    mins = np.empty(r)
    maxs = np.empty(r)
    for i in range(r):
        vi = V[i]
        maxs[i] = np.sum(np.where(vi >= 0, vi * highs, vi * lows)) + br[i]
        mins[i] = np.sum(np.where(vi >= 0, vi * lows, vi * highs)) + br[i]
    new_ranges = np.vstack([mins, maxs])
    # output_ranges
    if output_ranges is None:
        out = np.array([[0.0], [1.0]])
    else:
        out = np.asarray(output_ranges)
        if out.ndim == 1:
            out = out.reshape(2, 1)
        if out.shape[0] != 2 or out.shape[1] not in (1, r):
            raise ValueError('output_ranges must be (2,) or (2,r)')
    if out.shape[1] == 1:
        out = np.tile(out, (1, r))
    ol, oh = out
    scale = (oh - ol) / (maxs - mins)
    shift = ol - scale * mins
    inv_scale = 1 / scale
    # build recon map: u_norm→z
    W_rec = Q @ np.diag(inv_scale)
    b_rec = bias - Q @ (inv_scale * shift)
    return AffineTransformer(W_rec, b_rec, out_reshape=(M,))


class Zonotope:
    """
    Represents an affine zonotope Z = { b + B @ x  |  x_i in [l_i, u_i] }.
    - B:  (D×n) basis matrix (columns are generators)
    - ranges: (2×n) array of [lows; highs]
    - offset: (D,) bias vector b.
    """

    def __init__(
        self, basis: np.ndarray, ranges: np.ndarray, offset: np.ndarray = None
    ):
        basis_arr = np.asarray(basis)
        ranges_arr = np.asarray(ranges)
        if basis_arr.ndim != 2:
            raise ValueError('basis must be shape (D,n)')
        # detect orientation: if ranges matches rows of basis, transpose
        if (
            ranges_arr.shape[1] == basis_arr.shape[0]
            and ranges_arr.shape[1] != basis_arr.shape[1]
        ):
            basis_arr = basis_arr.T
        # now basis_arr is (D, n)
        D, n = basis_arr.shape

        if ranges_arr.shape != (2, n):
            msg = f'ranges must be (2,{n}), got {ranges_arr.shape}'
            raise ValueError(msg)

        if offset is None:
            offset = np.zeros(D)
        offset = np.asarray(offset)
        if offset.shape != (D,):
            msg = f'offset must be length {D}'
            raise ValueError(msg)

        non_zero_range_mask = abs(ranges_arr[0] - ranges_arr[1]) > EPS

        self.basis = basis_arr[:, non_zero_range_mask]
        self.ranges = ranges_arr[:, non_zero_range_mask]
        self.offset = offset

    def apply_affine(self, A: 'AffineTransformer') -> 'Zonotope':
        """Returns a new Zonotope = A(Z), applying Z → { A(z) | z in Z }."""
        Bm, new_ranges, b_new = apply_affine_to_subspace(
            self.basis, self.ranges, A
        )
        # the sub‐function already returns the new offset b_new
        return Zonotope(Bm, new_ranges, offset=b_new)

    def normalize(self, output_ranges: np.ndarray = None) -> AffineTransformer:
        """
        Collapse and orthonormalize the zonotope’s affine image.
        Returns an AffineTransformer T such that
          T(u) reconstructs points in this zonotope for u in output_ranges.
        """
        return orthonormalize_subspace(
            self.basis,
            self.ranges,
            self.offset,
            output_ranges=output_ranges,
        )

    @property
    def dim(self) -> int:
        return self.basis.shape[0]

    @property
    def order(self) -> int:
        return self.basis.shape[1]
