

import numpy as np

from viperleed_jax.lib_math import EPS
from viperleed_jax.transformation_tree.linear_transformer import AffineTransformer

def compute_reduced_subspace(
    basis_vectors: np.ndarray,
    coordinate_ranges: np.ndarray,
    transform: AffineTransformer,
    output_ranges: np.ndarray = None,
) -> AffineTransformer:
    """
    Given a D×n “box” in ℝᴰ (spanned by `basis_vectors` with per-axis `coordinate_ranges`),
    apply an arbitrary affine `transform`, collapse any redundant directions,
    re-orthonormalize to a minimal r-dimensional subspace, then finally
    rescale those r axes into `output_ranges`.
    Returns an `AffineTransformer` that maps from that normalized r-space
    back into the original ambient space.

    This is intended as a final “normalization” layer in a parameter tree: you
    start with some subspace of free parameters (the D×n box), you apply
    any number of upstream linear/affine layers, and then this routine
    ensures you end up with an orthonormal, minimum-dimensional coordinate
    system whose outputs lie in a neat [low, high] box.

    Parameters
    ----------
    basis_vectors : ndarray, shape (D, n)
        Columns are a basis for the original subspace in ℝᴰ.  The “unit box”
        in coefficient-space is defined by `coordinate_ranges`.
    coordinate_ranges : ndarray, shape (2, n)
        Row 0 = lower bounds, Row 1 = upper bounds for each of the n basis
        coefficients.
    transform : AffineTransformer
        Any affine map (W @ x + b) from ℝᴰ → ℝᴹ.  It may reduce rank or mix
        coordinates arbitrarily.
    output_ranges : ndarray, shape (2, r) or (2,), optional
        Desired [low, high] interval for each of the r orthonormal output axes.
        If shape‐(2,) it is broadcast to all r dims.  Defaults to [0, 1].
        hence r = rank.

    Returns
    -------
    norm_transform : AffineTransformer
        An affine map from the r-dimensional normalized parameter space back
        to ℝᴹ.  It satisfies:

            norm_transform( u ) = W_rec @ u + b_rec

        where u ∈ ℝʳ lives in `output_ranges`, and the columns of W_rec are
        an orthonormal basis for the image of the original transformed subspace.
        The returned object also sets `out_reshape=(M,)`, so calling
        `norm_transform(u)` yields a length-M vector by default.

    Notes
    -----
    1.  Internally we compute
        - Bm = W @ basis_vectors       # project original basis
        - SVD -→ U[:, :r]  = orthonormal basis Q for Im(Bm)
        - pre-ranges in Qᵀ-coords, then scale+shift → `output_ranges`
        - invert that scaling to build a reconstruction map W_rec, b_rec
    2.  If the affine map collapses everything (rank 0), we return a
        zero-column transformer that simply outputs the bias `b` for all u.

    Examples
    --------
    ```python
    B = np.eye(3)  # full R^3
    ranges = np.array([[-1, -1, -1], [1, 1, 1]])
    # collapse x,y into a single axis + pass z through
    aff = AffineTransformer(np.array([[1, 1, 0], [0, 0, 1]]), biases=[0, 0])
    norm = compute_reduced_subspace(B, ranges, aff, output_ranges=[0, 1])
    # now norm.in_dim == 3, norm.out_dim == 2,
    # and norm.weights is 2×3 with orthonormal rows in the collapsed subspace
    ```
    """
    # validate inputs
    if basis_vectors.ndim != 2:
        raise ValueError('basis_vectors must be 2D')
    D, n = basis_vectors.shape
    cr = np.asarray(coordinate_ranges)
    if cr.shape != (2, n):
        raise ValueError(f'coordinate_ranges must be shape (2,{n})')
    lows, highs = cr
    W, b = transform.weights, transform.biases
    M = W.shape[0]

    # project original basis into image
    Bm = W @ basis_vectors  # (M, n)
    U, S, Vh = np.linalg.svd(Bm, full_matrices=False)
    rank = int((S > EPS).sum())
    if rank == 0:
        # zero-dim output
        return AffineTransformer(np.zeros((M, 0)), b, out_reshape=(M,))
    Q = U[:, :rank]  # (M, r)

    # compute pre-normalization ranges in reduced coords
    V = Q.T @ Bm  # (r, n)
    br = Q.T @ b  # (r,)
    mins = np.empty(rank)
    maxs = np.empty(rank)
    for i in range(rank):
        vi = V[i]
        maxs[i] = np.sum(np.where(vi >= 0, vi * highs, vi * lows)) + br[i]
        mins[i] = np.sum(np.where(vi >= 0, vi * lows, vi * highs)) + br[i]

    # desired normalized ranges
    if output_ranges is None:
        out = np.array([[0.0], [1.0]])
    else:
        out = np.asarray(output_ranges)
        if out.ndim == 1:
            out = out.reshape(2, 1)
        if out.shape[0] != 2 or out.shape[1] not in (1, rank):
            raise ValueError('output_ranges must be shape (2,) or (2, r)')
    if out.shape[1] == 1:
        out = np.tile(out, (1, rank))
    out_l, out_h = out

    # compute scale & shift: u_norm = scale*u + shift
    scale = (out_h - out_l) / (maxs - mins)
    shift = out_l - scale * mins

    # build reconstruction: u_norm -> z in R^M
    inv_scale = 1.0 / scale
    # z = Q @ ((u_norm - shift)/scale) + b
    W_rec = Q @ np.diag(inv_scale)  # (M, r)
    b_rec = b - Q @ (inv_scale * shift)  # (M,)
    return AffineTransformer(W_rec, b_rec, out_reshape=(M,))

def apply_affine_to_subspace(
    basis_vectors: np.ndarray,
    coordinate_ranges: np.ndarray,
    transform: AffineTransformer,
):
    """
    Apply an affine map to a subspace box.

    Parameters
    ----------
    basis_vectors : (D, n) array
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
    if basis_vectors.ndim != 2:
        raise ValueError('basis_vectors must be 2D')
    D, n = basis_vectors.shape
    cr = np.asarray(coordinate_ranges)
    if cr.shape != (2, n):
        raise ValueError(f'coordinate_ranges must be shape (2,{n})')
    lows, highs = cr
    W, b = transform.weights, transform.biases
    # project basis
    Bm = W @ basis_vectors
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
        vec = Bm[:, i]
        low_pt = vec * lows[i] + b
        high_pt = vec * highs[i] + b
        new_mins[i] = np.where(lows[i] <= highs[i], low_pt.min(), high_pt.min())
        new_maxs[i] = np.where(highs[i] >= lows[i], high_pt.max(), low_pt.max())
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
