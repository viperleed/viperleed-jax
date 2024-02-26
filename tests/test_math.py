"""Test Module R-factor"""
import pytest

from jax import numpy as jnp
from src.lib_math import _divide_zero_safe


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
