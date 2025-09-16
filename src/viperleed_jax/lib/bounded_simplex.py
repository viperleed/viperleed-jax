import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logit as jax_logit
from functools import partial

from viperleed_jax.lib.math import EPS

def _validate_bounds_numpy(lower, upper, tol=EPS):
    # For eager use before jitting (optional but handy to fail fast).
    lower_sum = float(jnp.sum(lower))
    upper_sum = float(jnp.sum(upper))
    if not jnp.all(lower <= upper):
        raise ValueError('Require lower[i] <= upper[i] for all i.')
    if lower_sum > 1.0 + tol or upper_sum < 1.0 - tol:
        msg = (
            f'Infeasible bounds: need sum(lower) <= 1 <= sum(upper), '
            f'got sum(lower)={lower_sum}, sum(upper)={upper_sum}.'
        )
        raise ValueError(msg)


def bounded_softmax_jax(
    z, lower, upper, temperature=1.0, tol=EPS, max_iter=None
):
    """Capped softmax allocator with fixed-length loop (no 'done' flag)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    n = z.shape[0]
    max_steps = (n + 2) if (max_iter is None) else int(max_iter)

    c0 = lower
    caps0 = upper - lower
    S0 = 1.0 - jnp.sum(c0)
    active0 = caps0 > tol

    temp = jnp.maximum(temperature, 1e-12)

    def body_fn(_, state):
        c, caps, S, active = state

        # softmax over active coords; inactive get -inf -> weight 0
        z_masked = jnp.where(active, z / temp, -jnp.inf)
        z_shift = z_masked - jnp.max(z_masked)
        w = jnp.exp(z_shift)
        W = jnp.sum(w)
        k = jnp.sum(active)

        # if W ~ 0 (no active or extreme logits), fall back to uniform over active
        alloc_weights = jnp.where(
            W > tol,
            w / jnp.where(W > 0, W, 1.0),
            jnp.where(active, 1.0 / jnp.maximum(k, 1), 0.0),
        )

        alloc = S * alloc_weights
        step = jnp.minimum(alloc, caps)

        c_next = c + step
        S_next = S - jnp.sum(step)
        caps_next = caps - step
        active_next = caps_next > tol

        return (c_next, caps_next, S_next, active_next)

    c, caps, S, active = lax.fori_loop(
        0, max_steps, body_fn, (c0, caps0, S0, active0)
    )

    # Residue sweep: if mass remains and capacity exists, distribute proportionally
    def residue_fix(args):
        c, caps, S, active = args
        frac = jnp.where(active, caps, 0.0)
        frac_sum = jnp.sum(frac)
        step = jnp.where(frac_sum > 0, S * frac / frac_sum, 0.0)
        return c + step

    c = lax.cond(
        jnp.logical_and(S > tol, jnp.any(active)),
        residue_fix,
        lambda args: args[0],
        (c, caps, S, active),
    )

    # Final clip and tiny interior correction to hit sum=1 without breaking bounds
    c = jnp.clip(c, lower, upper)
    err = 1.0 - jnp.sum(c)
    interior = jnp.logical_and(c > lower + tol, c < upper - tol)
    m = jnp.sum(interior)
    delta = jnp.where(m > 0, err / m, 0.0)
    c = jnp.clip(c + jnp.where(interior, delta, 0.0), lower, upper)
    return c


@partial(jax.jit, static_argnames=('temperature', 'tol', 'max_iter'))
def bounded_softmax_from_unit(
    x,
    lower,
    upper,
    *,
    temperature=1.0,
    tol=EPS,
    max_iter=None,
):
    r"""
    Convert `x` in [0, 1]^n to a feasible concentration vector `c`.

    The concentration vector `c` lies in the simplex `\sum_i c_i = 1` and
    fulfills the box constraints given by `lower` and `upper`.

    Parameters
    ----------
    x : array_like
        Input vector with components in [0, 1].
    lower : array_like
        Per-component lower bounds for `c`.
    upper : array_like
        Per-component upper bounds for `c`.
    temperature : float, optional
        Softmax temperature used by the allocator. Lower -> peakier allocations.
        Default is 1.0.
    tol : float, optional
        Numerical tolerance used inside the allocator. Default is 1e-8.
    max_iter : int or None, optional
        Maximum iterations for the allocator loop. If None, defaults to n+2.

    Returns
    -------
    c : jax.Array
        Vector satisfying sum(c) == 1 (up to floating error) and
        lower <= c <= upper element wise.

    Notes
    -----
    This is **not** the Euclidean projection. It performs a capped softmax-style
    allocation that honors bounds while distributing mass according to the
    logits.
    """
    x = jnp.asarray(x)
    lower = jnp.asarray(lower, dtype=x.dtype)
    upper = jnp.asarray(upper, dtype=x.dtype)

    # Clip to avoid ±inf logits → NaNs in softmax normalization.
    x_clipped = jnp.clip(x, tol, 1.0 - tol)
    z = jax_logit(x_clipped)

    return bounded_softmax_jax(
        z, lower, upper, temperature=temperature, tol=tol, max_iter=max_iter
    )
