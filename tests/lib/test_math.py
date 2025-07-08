"""Test Module R-factor"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from scipy.special import sph_harm, spherical_jn

from viperleed_jax.dense_quantum_numbers import DENSE_L, DENSE_M
from viperleed_jax.lib.math import (
    EPS,
    _divide_zero_safe,
    apply_fun_grouped,
    bessel,
    cart_to_polar,
    project_onto_plane_sum_1,
    safe_norm,
    spherical_harmonics_components,
    spherical_to_cart,
    mirror_across_plane_sum_1,
)


def scipy_bessel(n, z):
    scipy_results = [spherical_jn(n, z) for n in range(n + 1)]
    return jnp.array(scipy_results).T


class TestDivideZeroSafe:
    # Test case for basic functionality
    def test_basic_division(self):
        numerator = jnp.array([1.0, 2.0, 3.0])
        denominator = jnp.array([2.0, 1.0, 0.0])
        result = _divide_zero_safe(numerator, denominator)
        expected_result = jnp.array([0.5, 2.0, 0.0])
        jnp.allclose(result, expected_result)

    # Test case for custom limit value
    def test_custom_limit_value(self):
        numerator = jnp.array([1.0, 2.0, 3.0])
        denominator = jnp.array([2.0, 1.0, 0.0])
        result = _divide_zero_safe(numerator, denominator, limit_value=999.0)
        expected_result = jnp.array([0.5, 2.0, 999.0])
        jnp.allclose(result, expected_result)

    # Test case for handling zero division safely
    def test_zero_division(self):
        numerator = jnp.array([1.0, 2.0, 3.0])
        denominator = jnp.array([0.0, 0.0, 0.0])
        result = _divide_zero_safe(numerator, denominator)
        expected_result = jnp.array([0.0, 0.0, 0.0])
        jnp.allclose(result, expected_result)

    # Test case for handling zero division safely with custom limit value
    def test_zero_division_with_custom_limit_value(self):
        numerator = jnp.array([1.0, 2.0, 3.0])
        denominator = jnp.array([0.0, 0.0, 0.0])
        result = _divide_zero_safe(numerator, denominator, limit_value=-999.0)
        expected_result = jnp.array([-999.0, -999.0, -999.0])
        jnp.allclose(result, expected_result)

    # Test case for handling division with non-zero denominator
    def test_non_zero_division(self):
        numerator = jnp.array([1.0, 2.0, 3.0])
        denominator = jnp.array([2.0, 4.0, 5.0])
        result = _divide_zero_safe(numerator, denominator)
        expected_result = jnp.array([0.5, 0.5, 0.6])
        jnp.allclose(result, expected_result)

    # Test case for handling division with non-zero denominator with custom limit value
    def test_non_zero_division_with_custom_limit_value(self):
        numerator = jnp.array([1.0, 2.0, 3.0])
        denominator = jnp.array([2.0, 4.0, 5.0])
        result = _divide_zero_safe(numerator, denominator, limit_value=-999.0)
        expected_result = jnp.array([0.5, 0.5, 0.6])
        jnp.allclose(result, expected_result)


class TestSafeNorm:
    # Test case for basic functionality
    def test_safe_norm_basic(self):
        vector = jnp.array([1.0, 2.0, 3.0])
        result = safe_norm(vector)
        expected_result = jnp.sqrt(14.0 + EPS**2)
        assert result == pytest.approx(expected_result)

    # Test case for handling zero vector
    def test_safe_norm_zero_vector(self):
        vector = jnp.array([0.0, 0.0, 0.0])
        result = safe_norm(vector)
        expected_result = EPS * 1e-2
        assert result == pytest.approx(expected_result)

    # Test case for handling small vector
    def test_safe_norm_small_vector(self):
        vector = jnp.array([1e-6, 1e-6, 1e-6])
        result = safe_norm(vector)
        expected_result = jnp.sqrt(3e-12 + (EPS / 100) ** 2)
        assert result == pytest.approx(expected_result)

    # Test case for handling large vector
    def test_safe_norm_large_vector(self):
        vector = jnp.array([1e6, 1e6, 1e6])
        result = safe_norm(vector)
        expected_result = jnp.sqrt(3e12 + EPS**2)
        assert result == pytest.approx(expected_result)

    # Test case for handling large vector
    def test_safe_norm_negative(self):
        vector = jnp.array([-1.0, -1.0, 0.0])
        result = safe_norm(vector)
        expected_result = jnp.sqrt(2 + EPS**2)
        assert result == pytest.approx(expected_result)


# todo
KNOW_CART_TO_POLAR = {
    'basic case': (
        np.array([1.0, 2.0, 3.0]),
        np.array([3.741657, 1.300247, 0.982794]),
    ),
    'positive x axis': (
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, np.pi / 2, 0.0]),
    ),
    'positive y axis': (
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, np.pi / 2, np.pi / 2]),
    ),
    'positive z axis': (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    'negative x axis': (
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, np.pi / 2, np.pi]),
    ),
    'negative y axis': (
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, np.pi / 2, -np.pi / 2]),
    ),
    'negative z axis': (
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, np.pi, 0.0]),
    ),
}

RETURN_TRANSFORM_VALUES = ()


class TestCartToPolar:
    @pytest.mark.parametrize(
        'known_value',
        list(KNOW_CART_TO_POLAR.values()),
        ids=list(KNOW_CART_TO_POLAR.keys()),
    )
    def test_known_values(self, known_value):
        cart_value, polar_value = known_value
        assert np.array(cart_to_polar(cart_value)) == pytest.approx(polar_value)

    @pytest.mark.parametrize(
        'cart_value',
        list(KNOW_CART_TO_POLAR.values()),
        ids=list(KNOW_CART_TO_POLAR.keys()),
    )
    def test_return_transform(self, cart_value):
        cart_value, _ = cart_value
        assert np.array(
            spherical_to_cart(cart_to_polar(cart_value))
        ) == pytest.approx(cart_value, abs=2e-8)

    @pytest.mark.parametrize(
        'cart_value',
        list(KNOW_CART_TO_POLAR.values()),
        ids=list(KNOW_CART_TO_POLAR.keys()),
    )
    def test_finite_jacobians(self, cart_value):
        cart_value, _ = cart_value
        reverse_jacobian = jax.jacrev(cart_to_polar)(cart_value)
        forward_jacobian = jax.jacfwd(cart_to_polar)(cart_value)
        assert np.all(np.isfinite(reverse_jacobian))
        assert np.all(np.isfinite(forward_jacobian))

    @pytest.mark.xfail(
        reason='Jacobian is made to be finite; needs revisiting.'
    )
    @pytest.mark.parametrize(
        'cart_value',
        list(KNOW_CART_TO_POLAR.values()),
        ids=list(KNOW_CART_TO_POLAR.keys()),
    )
    def test_composition_jacobians(self, cart_value):
        cart_value, _ = cart_value
        composition = lambda x: spherical_to_cart(cart_to_polar(x))
        reverse_jacobian = jax.jacrev(composition)(cart_value)
        forward_jacobian = jax.jacfwd(composition)(cart_value)
        assert np.array(reverse_jacobian) == pytest.approx(
            np.identity(3), abs=1e-8
        )
        assert np.array(forward_jacobian) == pytest.approx(
            np.identity(3), abs=1e-8
        )


class TestHARMONY:
    # Testcase for basic functionality by comparing it with scipy
    @pytest.mark.parametrize(
        'C',
        [
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([0.1, -0.3, -0.4]),
            jnp.array([-0.1, 1.0, 1.0]),
        ],
    )
    def test_HARMONY_normal_case(self, C):
        # Define input vector C for normal case
        LMAX = 1
        # Compute expected output using sph_harm directly
        expected_output = sph_harm(
            DENSE_M[2 * LMAX],
            DENSE_L[2 * LMAX],
            jnp.array([jnp.arctan2(C[2] + EPS, C[1] + EPS)]),
            jnp.array([jnp.arccos((C[0] + EPS) / safe_norm(C))]),
        )
        # Compare with the output of HARMONY function
        assert spherical_harmonics_components(LMAX, C) == pytest.approx(
            expected_output
        )

    def test_HARMONY_division_by_zero(self):
        # Define input vector C where division by zero might occur
        C = jnp.array([1.0, 0.0, 0.0])
        LMAX = 2
        # Compute expected output manually as sph_harm will raise errors
        expected_output = sph_harm(
            DENSE_M[2 * LMAX], DENSE_L[2 * LMAX], jnp.pi * 0.25, 0
        )
        # Compare with the output of HARMONY function
        assert spherical_harmonics_components(LMAX, C) == pytest.approx(
            expected_output
        )


class TestBessel:
    # rudimentary tests for bessel functions only, since
    # more detailed tests are present in the spbessax package
    @pytest.mark.parametrize('n1', [5, 10, 15])
    def test_bessel_real_case(self, n1):
        # Test case for normal scenario
        z = jnp.array([0.1, 1.0, 2.0, 3.0])
        assert bessel(z, n1) == pytest.approx(scipy_bessel(n1, z))

    @pytest.mark.parametrize('n1', [5, 10, 15])
    def test_bessel_imaginary_case(self, n1):
        # Test case for edge case where z=0
        z = jnp.array([0.1, 1.0, 2.0, 3.0]) - 2.0j
        # For z=0, the expression should be directly computed without vectorization
        assert bessel(z, n1) == pytest.approx(scipy_bessel(n1, z))


class TestMirrorAcrossSum1Plane:
    def test_mirror_preserves_distance_to_plane(self):
        """Test that the distance to the plane is preserved after mirroring."""
        x = jnp.array([3.0, 0.0, 0.0])
        mirror = mirror_across_plane_sum_1(x)
        deviation_before = jnp.sum(x) - 1.0
        deviation_after = 1.0 - jnp.sum(mirror)
        assert deviation_before == pytest.approx(deviation_after)

    def test_mirror_of_vector_on_plane_is_identity(self):
        """Test that a vector on the plane is unchanged after mirroring."""
        x = jnp.array([0.5, 0.25, 0.25])
        mirror = mirror_across_plane_sum_1(x)
        assert jnp.allclose(mirror, x)

    def test_mirror_moves_sum_symmetric_around_1(self):
        """Test that the mirror result has a sum symmetric to 1."""
        x = jnp.array([0.7, 0.2, 0.2])
        x_sum = jnp.sum(x)
        mirror = mirror_across_plane_sum_1(x)
        mirror_sum = jnp.sum(mirror)
        assert jnp.allclose(x_sum + mirror_sum, 2.0)

    def test_zero_vector_mirroring(self):
        """Test mirroring of a zero vector."""
        x = np.zeros(5)
        mirror = mirror_across_plane_sum_1(x)
        expected = np.array([0.4, 0.4, 0.4, 0.4, 0.4])
        assert mirror == pytest.approx(expected)


class TestProjectSum1Plane:
    def test_projection_preserves_sum(self):
        """Test that the projection results in a vector summing to 1."""
        x = jnp.array([3.0, 0.0, 0.0])
        proj = project_onto_plane_sum_1(x)
        assert jnp.allclose(
            jnp.sum(proj), 1.0
        ), f'Sum after projection was {jnp.sum(proj)}'

    def test_projection_of_vector_on_plane_is_identity(self):
        """Test that a vector already on the plane remains unchanged."""
        x = jnp.array([0.5, 0.25, 0.25])
        assert jnp.sum(x), pytest.approx(1.0)  # Pre-check
        proj = project_onto_plane_sum_1(x)
        assert jnp.allclose(
            proj, x
        ), 'Projection altered a point already on the plane.'

    def test_projection_is_linear_up_to_offset(self):
        """Test that projecting two vectors and summing is predictable."""
        x = jnp.array([2.0, 0.0, 0.0])
        y = jnp.array([0.0, 3.0, 0.0])
        proj_x = project_onto_plane_sum_1(x)
        proj_y = project_onto_plane_sum_1(y)
        proj_sum = project_onto_plane_sum_1(x + y)

        # Projection is affine, not linear, so we check:
        assert jnp.sum(proj_x) == pytest.approx(1.0)
        assert jnp.sum(proj_y) == pytest.approx(1.0)
        assert jnp.sum(proj_sum) == pytest.approx(1.0)

    def test_high_dimensional_projection(self):
        """Test projection in higher dimensions."""
        x = jnp.arange(1, 11, dtype=float)  # 10D vector
        proj = project_onto_plane_sum_1(x)
        assert jnp.sum(proj) == pytest.approx(
            1.0
        ), f'Sum after projection was {jnp.sum(proj)}'

    def test_zero_vector_projection(self):
        """Test projection of a zero vector."""
        x = jnp.zeros(5)
        proj = project_onto_plane_sum_1(x)
        expected = jnp.ones(5) / 5
        assert np.allclose(
            proj, expected
        ), f'Projected zero vector is {proj}, expected {expected}'


class TestApplyFunGrouped:
    def test_apply_fun_grouped_basic_doubling(self):
        """Test doubling groups individually."""
        v = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ind = (0, 1, 0, 1, 2, 2)

        def fun(x):
            return x * 2

        result = apply_fun_grouped(v, ind, fun)
        expected = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])

        assert result == pytest.approx(expected)

    def test_apply_fun_grouped_identity(self):
        """Test with the identity function (should return input unchanged)."""
        v = jnp.array([1.0, 2.0, 3.0, 4.0])
        ind = (0, 0, 1, 1)

        result = apply_fun_grouped(v, ind, lambda x: x)

        assert result == pytest.approx(v)

    def test_apply_fun_grouped_sum_normalization(self):
        """Normalize each group so its sum is 1."""
        v = jnp.array([2.0, 2.0, 1.0, 3.0])
        ind = (0, 0, 1, 1)

        def normalize(x):
            return x / jnp.sum(x)

        result = apply_fun_grouped(v, ind, normalize)

        expected = jnp.array(
            [
                0.5,
                0.5,  # [2,2] normalized
                0.25,
                0.75,  # [1,3] normalized
            ]
        )

        assert result == pytest.approx(expected)

    def test_apply_fun_grouped_empty_group_behavior(self):
        """Ensure behavior is fine when some group index is missing (e.g., no '2' label)."""
        v = jnp.array([1.0, 2.0, 3.0])
        ind = (0, 0, 1)  # No group '2'

        def fun(x):
            return x + 1

        result = apply_fun_grouped(v, ind, fun)
        expected = jnp.array([2.0, 3.0, 4.0])

        assert result == pytest.approx(expected)

    def test_apply_fun_grouped_singleton_groups(self):
        """Test where each element is its own group."""
        v = jnp.array([10.0, 20.0, 30.0])
        ind = (0, 1, 2)

        def fun(x):
            return x * 0.1

        result = apply_fun_grouped(v, ind, fun)
        expected = jnp.array([1.0, 2.0, 3.0])

        assert result == pytest.approx(expected)


if __name__ == '__main__':
    pytest.main([__file__])
