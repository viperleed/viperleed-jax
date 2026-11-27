import numpy as np

from viperleed_jax.lib.math import EPS
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)

class ZonotopeNotOrthogonalError(ValueError):
    """Raised when two Zonotopes are not orthogonal and cannot be added."""


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

    @property
    def dim(self) -> int:
        return self.basis.shape[0]

    @property
    def order(self) -> int:
        return self.basis.shape[1]

    def add_orthogonal_same_center(self, other: 'Zonotope'):
        """Add orhogonal zonotope with same center.

        Return self ⊕ other, but only if:
        (i) offsets (centers) match within `tol`, and
        (ii) span(self.basis) ⟂ span(other.basis) within `tol`.

        Raises
        ------
        TypeError
            If `other` is not a Zonotope.
        ValueError
            If dimensions differ, centers differ, or subspaces are not orthogonal.
        """
        if not isinstance(other, Zonotope):
            raise TypeError('other must be a Zonotope')
        if self.dim != other.dim:
            raise ValueError(f'dimension mismatch: {self.dim} vs {other.dim}')

        if not np.allclose(self.offset, other.offset, atol=EPS, rtol=0.0):
            raise ValueError('centers (offsets) differ beyond tolerance')

        if not self.is_orthogonal_to(other):
            raise ZonotopeNotOrthogonalError(
                'subspaces are not orthogonal within tolerance'
            )

        # Concatenate generators; stack ranges; center stays the same
        if self.order or other.order:
            B = np.hstack([self.basis, other.basis])
            ranges = np.concatenate([self.ranges, other.ranges], axis=1)
        else:
            B = np.zeros((self.dim, 0))
            ranges = np.zeros((2, 0))
        return Zonotope(B, ranges, offset=self.offset.copy())

    def apply_affine(self, A: 'AffineTransformer') -> 'Zonotope':
        """Return a new Zonotope = A(Z), applying Z → { A(z) | z in Z }.

        The coefficients keep their ranges: only the vectors they scale
        are mapped. The new offset is the bias of A; the offset of this
        zonotope is not mapped (leaf zonotopes have none).
        """
        return Zonotope(A.weights @ self.basis, self.ranges, offset=A.biases)

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

    def is_orthogonal_to(self, other):
        """
        Return True if span(self.basis) ⟂ span(other.basis).

        Uses QR to get orthonormal bases Q1, Q2 and checks ||Q1ᵀ Q2||₂ ≤ tol,
        i.e. all principal angles are ~90°.
        """
        if type(other) is not Zonotope:
            raise TypeError('other must be a Zonotope')

        B1 = self.basis
        B2 = other.basis
        if B1.size == 0 or B2.size == 0:
            # Empty subspace is orthogonal to everything
            return True

        # Orthonormal bases for the column spaces
        Q1, _ = np.linalg.qr(B1, mode='reduced')
        Q2, _ = np.linalg.qr(B2, mode='reduced')

        # Cross Gram; spectral norm equals max singular value = cos(min principal angle)
        M = Q1.T @ Q2
        # Use largest singular value (spectral norm)
        smax = np.linalg.svd(M, compute_uv=False)[0]
        return bool(smax <= EPS)
