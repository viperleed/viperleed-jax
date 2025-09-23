import jax
import jax.numpy as jnp
import pytest
import numpy as np

from viperleed_jax.lib.derived_quantities.normalized_occupations import (
    apply_bounded_simplex,
)
from viperleed_jax.lib.derived_quantities.normalized_occupations import (
    NormalizedOccupations,
)
from bounded_simplex_helpers import (
    BOUNDS,
    SAMPLING_ITERATIONS,
    uniform_vector,
    SAMPLING_TOLERANCE,
)


def _stack_in_vecs(x, lo, hi):
    return jnp.stack([x, lo, hi], axis=1)


def _feasible_vacancy_interval(low_np: np.ndarray, high_np: np.ndarray):
    """
    For sum(x)=1-v with box low<=x<=high, feasibility requires:
      sum(low) <= 1 - v <= sum(high).
    Thus:
      v ∈ [max(0, 1 - sum(high)), min(1, 1 - sum(low))].
    """
    v_min = max(0.0, 1.0 - float(np.sum(high_np)))
    v_max = min(1.0, 1.0 - float(np.sum(low_np)))
    return v_min, v_max


class TestApplyBoundedSimplex:
    @pytest.mark.parametrize('n', [1, 3, 8])
    def test_shapes_and_types(self, n):
        x = jnp.linspace(0.1, 0.9, n)
        lo = jnp.zeros(n)
        hi = jnp.ones(n)
        in_vecs = _stack_in_vecs(x, lo, hi)

        out = apply_bounded_simplex(in_vecs, max_vacancy=0.8, min_vacancy=0.0)

        assert out.shape == (n,)
        assert isinstance(out, jax.Array)

    @pytest.mark.parametrize(
        'x, lo, hi, min_vac, max_vac',
        [
            # already OK sum<=1, wide vacancy bounds
            (jnp.array([0.2, 0.3, 0.1]), jnp.zeros(3), jnp.ones(3), 0.0, 0.8),
            # sum>1, needs adjustment
            (jnp.array([0.7, 0.7, 0.1]), jnp.zeros(3), jnp.ones(3), 0.0, 0.8),
            # tight element-wise box
            (
                jnp.array([0.4, 0.4, 0.4]),
                jnp.array([0.2, 0.25, 0.1]),
                jnp.array([0.5, 0.45, 0.3]),
                0.0,
                0.9,
            ),
        ],
    )
    def test_respects_bounds_and_vacancy_interval(
        self, x, lo, hi, min_vac, max_vac
    ):
        atol = 1e-6
        in_vecs = _stack_in_vecs(x, lo, hi)
        out = apply_bounded_simplex(
            in_vecs, max_vacancy=max_vac, min_vacancy=min_vac
        )

        # element-wise bounds (inequalities — keep as-is)
        assert jnp.all(out >= lo - atol)
        assert jnp.all(out <= hi + atol)

        # implied vacancy and sum==1 with approx
        vac = 1.0 - jnp.sum(out)
        assert float(vac) >= (min_vac - 1e-6)
        assert float(vac) <= (max_vac + 1e-6)
        assert float(jnp.sum(out) + vac) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize('bounds', BOUNDS)
    def test_iteration_min_max_observed(self, rng, bounds):
        """
        Run many random x in [0,1]^n and track per-dim mins/maxs of outputs.
        Ensure every draw is feasible and summarize observed min/max.
        """
        low_np, high_np = bounds
        n = low_np.shape[0]
        low = jnp.asarray(low_np)
        high = jnp.asarray(high_np)

        mins = np.full_like(low_np, 1.0, dtype=np.float64)
        maxs = np.full_like(low_np, -1.0, dtype=np.float64)

        # Use permissive vacancy bounds so the mapper can explore the box.
        # This makes feasibility trivial: sum(out) ∈ [0, 1].
        min_vac, max_vac = 0.0, 1.0

        for _ in range(SAMPLING_ITERATIONS):
            # random x in [0,1]^n (as in your example)
            x = jnp.asarray(uniform_vector(np.zeros(n), np.ones(n), rng))
            in_vecs = _stack_in_vecs(x, low, high)

            projected = apply_bounded_simplex(
                in_vecs, max_vacancy=max_vac, min_vacancy=min_vac
            )

            # Feasibility checks
            assert jnp.all(projected >= low - 1e-12)
            assert jnp.all(projected <= high + 1e-12)

            vac = float(1.0 - jnp.sum(projected))
            # Sum with vacancy is ~1
            assert float(jnp.sum(projected) + vac) == pytest.approx(
                1.0, abs=1e-12
            )
            # Vacancy in [min_vac, max_vac]
            assert vac >= min_vac - 1e-12
            assert vac <= max_vac + 1e-12

            # Track observed componentwise mins/maxs
            p_np = np.asarray(projected)
            mins = np.minimum(mins, p_np)
            maxs = np.maximum(maxs, p_np)

        # The observed mins/maxs must lie within bounds
        assert np.all(mins >= low_np - 1e-12)
        assert np.all(maxs <= high_np + 1e-12)

        # With many draws, we should get reasonably close to the bounds
        # (skip "closeness" on dimensions that are fixed by low==high — they’ll match exactly)
        is_fixed = np.isclose(low_np, high_np)
        if not np.all(is_fixed):
            assert np.all(
                np.abs(mins[~is_fixed] - low_np[~is_fixed]) < SAMPLING_TOLERANCE
            )
            assert np.all(
                np.abs(high_np[~is_fixed] - maxs[~is_fixed])
                < SAMPLING_TOLERANCE
            )
        else:
            # When all dims are fixed, mins/maxs equal bounds exactly
            assert mins.tolist() == pytest.approx(low_np.tolist(), abs=1e-12)
            assert maxs.tolist() == pytest.approx(high_np.tolist(), abs=1e-12)

    @pytest.mark.parametrize('bounds', BOUNDS)
    def test_iteration_min_max_observed_fixed_vacancy(self, rng, bounds):
        """
        Like the exploration test, but with a fixed vacancy. For each box,
        compute the feasible vacancy interval and test at its endpoints
        and midpoint. Skip boxes that cannot support any fixed vacancy.
        """
        low_np, high_np = bounds
        n = low_np.shape[0]
        low = jnp.asarray(low_np)
        high = jnp.asarray(high_np)

        v_min, v_max = _feasible_vacancy_interval(low_np, high_np)
        if v_min > v_max + 1e-15:
            pytest.skip(
                f'Infeasible bounds for any fixed vacancy: '
                f'sum(low)={np.sum(low_np):.6f}, sum(high)={np.sum(high_np):.6f}'
            )

        vac_candidates = [v_min, 0.5 * (v_min + v_max), v_max]

        for fixed_vac in vac_candidates:
            mins = np.full_like(low_np, 1.0, dtype=np.float64)
            maxs = np.full_like(low_np, -1.0, dtype=np.float64)

            for _ in range(SAMPLING_ITERATIONS):
                # random x in [0,1]^n
                x = jnp.asarray(
                    rng.uniform(0.0, 1.0, size=n), dtype=jnp.float64
                )
                in_vecs = _stack_in_vecs(x, low, high)

                projected = apply_bounded_simplex(
                    in_vecs, max_vacancy=fixed_vac, min_vacancy=fixed_vac
                )

                # Feasibility
                assert jnp.all(projected >= low - 1e-12)
                assert jnp.all(projected <= high + 1e-12)

                vac = float(1.0 - jnp.sum(projected))
                assert vac == pytest.approx(fixed_vac, abs=1e-10)
                assert float(jnp.sum(projected) + vac) == pytest.approx(
                    1.0, abs=1e-12
                )

                # Track per-dim extrema
                p_np = np.asarray(projected)
                mins = np.minimum(mins, p_np)
                maxs = np.maximum(maxs, p_np)

            # Observed extrema must sit within the bounds
            assert np.all(mins >= low_np - 1e-12)
            assert np.all(maxs <= high_np + 1e-12)

            # When some dims are fixed (low==high), we should hit them exactly
            fixed_mask = np.isclose(low_np, high_np)
            if np.any(fixed_mask):
                assert mins[fixed_mask].tolist() == pytest.approx(
                    low_np[fixed_mask].tolist(), abs=1e-12
                )
                assert maxs[fixed_mask].tolist() == pytest.approx(
                    high_np[fixed_mask].tolist(), abs=1e-12
                )

            # Optional: if the fixed-vacancy sum S=1-v can reach the lower/upper
            # edges for at least one coordinate (given others at their opposites),
            # we expect mins/maxs to get close. This is conservative—skip if not provable.
            S = 1.0 - fixed_vac
            # maximal feasible sum min_i can allow while others at upper:
            can_touch_low = S <= (np.sum(high_np) - high_np + low_np)
            can_touch_high = S >= (np.sum(low_np) - low_np + high_np)

            if np.any(can_touch_low):
                idx = np.where(can_touch_low)[0]
                assert np.all(
                    np.abs(mins[idx] - low_np[idx]) < SAMPLING_TOLERANCE
                )
            if np.any(can_touch_high):
                idx = np.where(can_touch_high)[0]
                assert np.all(
                    np.abs(maxs[idx] - high_np[idx]) < SAMPLING_TOLERANCE
                )

    def test_fixed_component_is_respected(self):
        atol = 1e-7
        # fix the middle component to 0.3 by lo==hi
        x = jnp.array([0.6, 0.1, 0.6])
        lo = jnp.array([0.0, 0.3, 0.0])
        hi = jnp.array([1.0, 0.3, 1.0])
        in_vecs = _stack_in_vecs(x, lo, hi)

        out = apply_bounded_simplex(in_vecs, max_vacancy=0.8, min_vacancy=0.0)

        assert float(out[1]) == pytest.approx(0.3, abs=atol)
        assert jnp.all(out >= lo - atol)
        assert jnp.all(out <= hi + atol)
        # sanity: sum<=1 (allow tiny epsilon)
        assert float(1.0 - jnp.sum(out)) >= -1e-6

    @pytest.mark.parametrize('fixed_vac', [0.0, 0.15, 0.5])
    def test_fixed_vacancy_is_respected(self, fixed_vac):
        atol = 1e-6
        x = jnp.array([0.6, 0.6, 0.1])
        lo = jnp.zeros(3)
        hi = jnp.ones(3)
        in_vecs = _stack_in_vecs(x, lo, hi)

        out = apply_bounded_simplex(
            in_vecs, max_vacancy=fixed_vac, min_vacancy=fixed_vac
        )
        vac = 1.0 - jnp.sum(out)

        assert float(vac) == pytest.approx(fixed_vac, abs=atol)
        assert jnp.all(out >= lo - atol)
        assert jnp.all(out <= hi + atol)

    def test_jit_equivalence(self):
        x = jnp.array([0.4, 0.6, 0.4, 0.3])
        lo = jnp.array([0.1, 0.0, 0.2, 0.1])
        hi = jnp.array([0.8, 0.9, 0.7, 0.4])
        in_vecs = _stack_in_vecs(x, lo, hi)

        f_jit = jax.jit(apply_bounded_simplex)

        out_eager = apply_bounded_simplex(in_vecs, 0.6, 0.0)
        out_jit = f_jit(in_vecs, 0.6, 0.0)

        # pytest.approx for vectors: compare lists for portability
        assert out_jit.tolist() == pytest.approx(
            out_eager.tolist(), rel=1e-6, abs=1e-7
        )

    def test_grad_is_finite(self):
        # gradient through x only (lo/hi fixed)
        x = jnp.array([0.4, 0.6, 0.4, 0.3])
        lo = jnp.array([0.1, 0.0, 0.2, 0.1])
        hi = jnp.array([0.8, 0.9, 0.7, 0.4])

        def loss(x_vec):
            in_vecs = _stack_in_vecs(x_vec, lo, hi)
            w = jnp.arange(in_vecs.shape[0], dtype=x_vec.dtype)
            return jnp.dot(apply_bounded_simplex(in_vecs, 0.6, 0.0), w)

        g = jax.grad(loss)(x)
        assert jnp.all(jnp.isfinite(g)), 'Gradient contains NaN/Inf'

    def test_handles_n_equals_one(self):
        atol = 1e-7
        x = jnp.array([0.8])
        lo = jnp.array([0.2])
        hi = jnp.array([0.9])
        in_vecs = _stack_in_vecs(x, lo, hi)

        out = apply_bounded_simplex(in_vecs, max_vacancy=0.8, min_vacancy=0.0)
        assert out.shape == (1,)
        assert float(out[0]) >= (float(lo[0]) - atol)
        assert float(out[0]) <= (float(hi[0]) + atol)
        vac = 1.0 - float(out[0])
        assert vac >= -1e-6
        assert vac <= 0.8 + 1e-6