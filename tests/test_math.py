"""Test Module R-factor"""
import pytest

from jax import numpy as jnp
from src.lib_math import _divide_zero_safe
from src.lib_math import *


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
        expected_output = sph_harm(DENSE_M[2*LMAX], DENSE_L[2*LMAX], jnp.array([np.arctan2(3.0+EPS, 2.0+EPS)]), jnp.array([np.arccos((1.0+EPS)/safe_norm(C))]), n_max=LMAX)
        # Compare with the output of HARMONY function
        assert jnp.allclose(HARMONY(C, LMAX), expected_output)

    def test_HARMONY_division_by_zero(self):
        # Define input vector C where division by zero might occur
        C = jnp.array([1.0, 0.0, 0.0])
        LMAX = 2
        # Compute expected output manually as sph_harm will raise errors
        expected_output = sph_harm(DENSE_M[2*LMAX], DENSE_L[2*LMAX], np.pi*0.25, 0, n_max=LMAX)
        # Compare with the output of HARMONY function
        assert jnp.allclose(HARMONY(C, LMAX), expected_output, atol=1e-03) #very big because arccos is sensible to small changes

class TestBessel:
    def test_real_case(self):
        # Test case for normal scenario
        z = 1.5
        n1 = 3
        expected_output = jax.vmap(custom_spherical_jn, (0, None))(jnp.arange(n1), z)
        assert jnp.allclose(masked_bessel(z, n1), expected_output)

    def test_imaginary_case(self):
        # Test case for edge case where z=0
        z = -2.0j
        n1 = 3
        # For z=0, the expression should be directly computed without vectorization
        expected_output = jax.vmap(custom_spherical_jn, (0, None))(jnp.arange(n1), z)
        assert jnp.allclose(bessel(z, n1), expected_output)

class TestMaskedBessel:
    def test_normal_case(self):
        # Test case for normal scenario
        z = 1.5
        n1 = 3
        result = masked_bessel(z, n1)
        assert result.dtype == jnp.complex128
        assert result.shape == (n1,)

    def test_small_z(self):
        # Test case for small z
        z = 1e-8j
        n1 = 3
        result = masked_bessel(z, n1)
        assert result.dtype == jnp.complex128
        assert result.shape == (n1,)
        # The result should be close to [1.0, 0.0, 0.0] because bessel function is approximated as 1.0 for very small z
        assert jnp.allclose(result, np.array([1.0, 0.0, 0.0]))

    def test_zero_z(self):
        # Test case for z = 0
        z = 0.0
        n1 = 3
        result = masked_bessel(z, n1)
        assert result.dtype == jnp.complex128
        assert result.shape == (n1,)
        # The result should be [1.0, 0.0, 0.0] because bessel function is approximated as 1.0 for z = 0
        assert jnp.allclose(result, np.array([1.0, 0.0, 0.0]))

if __name__ == "__main__":
    pytest.main([__file__])