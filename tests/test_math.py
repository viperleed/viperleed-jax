"""Test Module R-factor"""
import pytest

from scipy.special import spherical_jn

from jax import numpy as jnp
from src.lib_math import _divide_zero_safe
from src.lib_math import *

def scipy_bessel(n, z):
    scipy_results = [
        spherical_jn(n, z) for n in range(n+1)
    ]
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

class TestHARMONY:
    # Testcase for basic functionality by comparing it with sph_harm
    def test_HARMONY_normal_case(self):
        # Define input vector C for normal case
        C = jnp.array([1.0, 2.0, 3.0])
        LMAX = 2
        # Compute expected output using sph_harm directly
        expected_output = sph_harm(DENSE_M[2*LMAX], DENSE_L[2*LMAX], jnp.array([jnp.arctan2(3.0+EPS, 2.0+EPS)]), jnp.array([jnp.arccos((1.0+EPS)/safe_norm(C))]), n_max=LMAX)
        # Compare with the output of HARMONY function
        assert jnp.allclose(HARMONY(C, LMAX), expected_output)

    def test_HARMONY_division_by_zero(self):
        # Define input vector C where division by zero might occur
        C = jnp.array([1.0, 0.0, 0.0])
        LMAX = 2
        # Compute expected output manually as sph_harm will raise errors
        expected_output = sph_harm(DENSE_M[2*LMAX], DENSE_L[2*LMAX], jnp.pi*0.25, 0, n_max=LMAX)
        # Compare with the output of HARMONY function
        assert jnp.allclose(HARMONY(C, LMAX), expected_output)

class TestBessel:
    # rudimentary tests for bessel functions only, since
    # more detailed tests are present in the spbessax package
    @pytest.mark.parametrize("n1", [5, 10, 15])
    def test_bessel_real_case(self, n1):
        # Test case for normal scenario
        z = jnp.array([0.1, 1., 2., 3.])
        assert bessel(z, n1) == pytest.approx(scipy_bessel(n1, z))

    @pytest.mark.parametrize("n1", [5, 10, 15])
    def test_bessel_imaginary_case(self, n1):
        # Test case for edge case where z=0
        z = jnp.array([0.1, 1., 2., 3.]) - 2.0j
        # For z=0, the expression should be directly computed without vectorization
        assert bessel(z, n1) == pytest.approx(scipy_bessel(n1, z))


if __name__ == "__main__":
    pytest.main([__file__])