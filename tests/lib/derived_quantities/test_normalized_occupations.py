import jax.numpy as jnp
import pytest

from viperleed_jax.lib.derived_quantities.normalized_occupations import (
    _normalize_atom_occ_vector,
    normalize_occ_vector,
)


def test_normalize_atom_occ_vector_below_threshold():
    """Test normalization when sum <= 1: should return input unchanged."""
    v = jnp.array([0.3, 0.3])
    result = _normalize_atom_occ_vector(v)
    assert jnp.allclose(result, v), f'Expected {v}, got {result}'


def test_normalize_atom_occ_vector_above_threshold():
    """Test normalization when sum > 1: should project."""
    v = jnp.array([0.6, 0.6])  # Sum = 1.2
    result = _normalize_atom_occ_vector(v)
    assert jnp.allclose(
        jnp.sum(result), 1.0
    ), f'Sum after normalization should be 1, got {jnp.sum(result)}'
    # Should maintain relative proportions approximately
    assert result[0] > 0 and result[1] > 0

def test_normalize_occ_vector_simple_case():
    """Test normalization across multiple atoms."""
    occ_vector = jnp.array([0.6, 0.6, 0.2])
    atom_ids = (0, 0, 1)  # Must be static

    result = normalize_occ_vector(occ_vector, atom_ids)

    # Group 0 (first two elements) should be normalized
    assert jnp.allclose(
        jnp.sum(result[:2]), 1.0
    ), f'Sum of group 0 should be 1, got {jnp.sum(result[:2])}'
    # Group 1 (last element) sum is 0.2, should stay unchanged
    assert jnp.allclose(
        result[2], 0.2
    ), f'Element 2 should stay 0.2, got {result[2]}'
