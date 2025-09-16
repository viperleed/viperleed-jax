import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logit as jax_logit


def _validate_bounds_numpy(lower, upper, tol=1e-12):
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
    z, lower, upper, temperature=1.0, tol=1e-12, max_iter=None
):
    z = jnp.asarray(z, dtype=jnp.float64)
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    n = z.shape[0]

    # >>> Python-level defaulting (ok inside jit; it's a plain Python branch)
    max_iter_val = (n + 2) if (max_iter is None) else int(max_iter)
    # make it a JAX scalar for comparisons later
    max_iter_val = jnp.asarray(max_iter_val, dtype=jnp.int32)

    c0 = lower
    caps0 = upper - lower
    S0 = 1.0 - jnp.sum(c0)
    active0 = caps0 > tol

    def cond_fn(state):
        c, caps, S, active, it = state
        return (S > tol) & jnp.any(active) & (it < max_iter_val)

    def body_fn(state):
        c, caps, S, active, it = state
        temp = jnp.maximum(temperature, 1e-12)
        z_masked = jnp.where(active, z / temp, -jnp.inf)
        z_shift = z_masked - jnp.max(z_masked)
        w = jnp.exp(z_shift)
        W = jnp.sum(w)
        k = jnp.sum(active)

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
        return (c_next, caps_next, S_next, active_next, it + 1)

    c, caps, S, active, _ = lax.while_loop(
        cond_fn, body_fn, (c0, caps0, S0, active0, jnp.array(0, jnp.int32))
    )

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

    c = jnp.clip(c, lower, upper)
    err = 1.0 - jnp.sum(c)
    interior = jnp.logical_and(c > lower + 1e-12, c < upper - 1e-12)
    m = jnp.sum(interior)
    delta = jnp.where(m > 0, err / m, 0.0)
    c = jnp.clip(c + jnp.where(interior, delta, 0.0), lower, upper)
    return c


def bounded_softmax_from_unit(
    x,
    lower,
    upper,
    *,
    temperature: float = 1.0,
    tol: float = 1e-8,
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

    z = jax_logit(x)
    return bounded_softmax_jax(
        z, lower, upper, temperature=temperature, tol=tol, max_iter=max_iter
    )
