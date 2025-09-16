# test_logits_from_x_logit.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logit as jax_logit

from viperleed_jax.lib.bounded_simplex import bounded_softmax_from_unit

_TEST_EPS = 1e-6
_ITERATIONS = 5000


# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture(params=[2, 3, 5], scope='module')
def dims(request):
    return request.param


@pytest.fixture(scope='module')
def eps():
    return _TEST_EPS


_BOUNDS = [  # format lower, upper
    # 2D cases
    (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
    (np.array([0.2, 0.3]), np.array([0.7, 0.8])),
    (np.array([0.0, 0.5]), np.array([0.5, 1.0])),
    (np.array([0.4, 0.4]), np.array([0.6, 0.6])),  # tight box
    # 3D cases
    (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
    (np.array([0.1, 0.1, 0.1]), np.array([0.7, 0.7, 0.7])),
    (np.array([0.2, 0.3, 0.1]), np.array([0.5, 0.6, 0.4])),
    (np.array([0.0, 0.0, 0.5]), np.array([0.5, 0.5, 0.5])),  # one fixed dim
    (np.array([0.3, 0.3, 0.3]), np.array([0.4, 0.4, 0.4])),  # tight box
    # 5D cases
    (np.array([0.1, 0.1, 0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5, 0.5, 0.5])),
    (
        np.array([0.2, 0.1, 0.1, 0.1, 0.1]),
        np.array([0.4, 0.4, 0.4, 0.1, 0.1]),
    ),  # two dims fixed
]


# -----------------------------
# Helper(s)
# -----------------------------


def uniform_vector(low, high, rng):
    """Draw a vector uniformly between low and high (elementwise)."""
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    return rng.uniform(low, high).astype(np.float64)


# -----------------------------
# Tests
# -----------------------------

class TestBoundedSoftmaxFromUnit:
    """Tests for bounded_softmax_from_unit function."""

    @pytest.mark.parametrize('bounds', _BOUNDS)
    def test_iteration_min_max_observed(self, rng, bounds):
        """
        Run many random x in [0,1]^3 and track per-dim mins/maxs of outputs.
        Ensure every draw is feasible and summarize observed min/max.
        """
        low, high = bounds

        mins = np.ones_like(low, dtype=np.float64) * 1.0
        maxs = np.ones_like(low, dtype=np.float64) * -1.0

        for _ in range(_ITERATIONS):
            x = uniform_vector(np.zeros_like(low), np.ones_like(high), rng)
            projected = bounded_softmax_from_unit(x, low, high)
            mins = np.minimum(mins, projected)
            maxs = np.maximum(maxs, projected)

            assert (projected >= low).all()
            assert (projected <= high).all()
            assert np.isclose(projected.sum(), 1.0, rtol=0, atol=1e-12)

        print(f'Bounds: {low} to {high}')
        # The observed mins/maxs must be within bounds
        assert (mins >= low).all()
        assert (maxs <= high).all()

        # the observed mins should be reasonably close to the lower bounds
        assert (abs(mins - low) < 0.001).all()
        # the observed maxs should be reasonably close to the upper bounds
        assert (abs(high - maxs) < 0.001).all()

    def test_permutation_equivariance(self, rng):
        """
        With symmetric bounds (same per-dim), permuting x should permute c.
        """
        d = 5
        # Symmetric bounds for all coordinates
        lower = np.full(d, 0.05, dtype=np.float64)
        upper = np.full(d, 0.9, dtype=np.float64)
        x = uniform_vector(np.zeros(d), np.ones(d), rng)

        # A random permutation
        perm = rng.permutation(d)
        invperm = np.argsort(perm)

        c = np.asarray(bounded_softmax_from_unit(x, lower, upper))
        c_perm = np.asarray(bounded_softmax_from_unit(x[perm], lower, upper))

        # Permuting inputs should permute the outputs the same way
        np.testing.assert_allclose(c[perm], c_perm, rtol=0, atol=1e-12)
        # And vice-versa
        np.testing.assert_allclose(c, c_perm[invperm], rtol=0, atol=1e-12)

    def test_equivalent_x_yield_equivalent_c(self, rng):
        """If the range is equal, the same x_i should yield the same c_i."""
        lower = np.array([0.2, 0.2, 0.1], dtype=np.float64)
        upper = np.array([0.6, 0.6, 0.9], dtype=np.float64)

        for _ in range(_ITERATIONS):
            # Draw a 2D point and duplicate one coord to make a 3D point:
            x_2d = uniform_vector(np.zeros(2), np.ones(2), rng)
            x = np.array([x_2d[0], x_2d[0], x_2d[1]], dtype=np.float64)
            c = np.asarray(bounded_softmax_from_unit(x, lower, upper))
            assert np.isclose(c[0], c[1], rtol=0, atol=1e-12)

    def test_temperature_effect_entropy(self, rng):
        """
        Lower temperature should generally produce a lower-entropy (peakier) allocation,
        all else equal. We allow up to 1% of random trials to violate this due to
        caps/saturation/ties or numerical quirks.
        """

        def entropy(p):
            p = jnp.asarray(p, dtype=jnp.float64)
            p = jnp.clip(p, 1e-12, 1.0)
            return float(-jnp.sum(p * jnp.log(p)))

        d = 5
        failures = 0

        for _ in range(_ITERATIONS):
            lower = rng.uniform(0.0, 0.2, size=d).astype(np.float64)
            upper = (lower + rng.uniform(0.3, 0.8, size=d)).astype(np.float64)
            x = uniform_vector(np.zeros(d), np.ones(d), rng)

            c_hot = np.asarray(
                bounded_softmax_from_unit(x, lower, upper, temperature=2.0)
            )
            c_cold = np.asarray(
                bounded_softmax_from_unit(x, lower, upper, temperature=0.2)
            )

            # Expect colder temp to be <= entropy than hotter (peakier).
            if entropy(c_cold) > entropy(c_hot):
                failures += 1

        rate = failures / _ITERATIONS
        assert rate <= 0.01, (
            f'Entropy ordering violated too often: {failures}/{_ITERATIONS} '
            f'trials ({rate:.2%}) > 1% threshold'
        )

    def test_grad_defined_interior(self):
        """
        Gradients w.r.t. x exist away from clipping and away from saturating caps.
        Use bounds that rarely cap (loose upper) and x away from {0,1}.
        """
        d = 5
        lower = np.zeros(d, dtype=np.float64)
        upper = np.ones(
            d, dtype=np.float64
        )  # effectively no caps; pure simplex
        x = np.linspace(0.2, 0.8, d, dtype=np.float64)

        def loss(x_):
            c = bounded_softmax_from_unit(x_, lower, upper, temperature=1.0)
            return jnp.sum(c**2)  # non-constant, smooth objective

        g = jax.grad(loss)(jnp.asarray(x))
        assert jnp.all(jnp.isfinite(g)), 'Non-finite gradient values'

    def test_handles_extreme_x_with_zeros_ones(self):
        """
        x containing exact 0s and 1s should not produce NaNs; result still feasible.
        (logit(0/1) -> +/-inf, allocator should handle via masked softmax logic)
        """
        x = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        lower = np.array([0.1, 0.05, 0.1, 0.05], dtype=np.float64)
        upper = np.array([0.6, 0.7, 0.6, 0.7], dtype=np.float64)

        c = bounded_softmax_from_unit(x, lower, upper)
        assert jnp.all(jnp.isfinite(c))
        assert jnp.isclose(jnp.sum(c), 1.0, rtol=0, atol=1e-12)
        assert jnp.all(c >= lower - 1e-12)
        assert jnp.all(c <= upper + 1e-12)

    def test_sum_lower_equals_one_returns_lower(self):
        """
        If sum(lower) == 1, there is no remaining mass. The allocator should return exactly lower.
        """
        lower = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        upper = np.array(
            [0.8, 0.9, 0.95], dtype=np.float64
        )  # irrelevant, caps unused
        x = np.array([0.1, 0.5, 0.9], dtype=np.float64)  # arbitrary

        c = np.asarray(bounded_softmax_from_unit(x, lower, upper))
        np.testing.assert_allclose(c, lower, rtol=0, atol=1e-12)
