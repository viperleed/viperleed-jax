import pytest
import jax.numpy as jnp
from viperleed_jax.parameters.linear_transfomer import LinearTransformer


# Sample test cases
def test_linear_transformer_initialization():
    weights = [[1, 2], [3, 4]]
    biases = [5, 6]
    transformer = LinearTransformer(weights, biases)
    assert transformer.n_free_params == 2
    assert transformer.weights.shape == (2, 2)
    assert transformer.biases.shape == (2,)
    assert transformer.out_reshape is None


def test_call_with_correct_shape():
    weights = [[1, 2], [3, 4]]
    biases = [5, 6]
    transformer = LinearTransformer(weights, biases)
    free_params = [0.5, 0.25]
    result = transformer(free_params)
    expected = jnp.array([6., 8.5])  # [1*0.5 + 2*0.25 + 5, 3*0.5 + 4*0.25 + 6]
    assert result ==  pytest.approx(expected)


def test_call_with_incorrect_shape():
    weights = [[1, 2], [3, 4]]
    biases = [5, 6]
    transformer = LinearTransformer(weights, biases)
    free_params = [0.5]  # Incorrect length
    with pytest.raises(ValueError, match="Free parameters have wrong shape"):
        transformer(free_params)


def test_call_without_free_params():
    weights = jnp.zeros(shape=(2, 0))
    biases = [5, 6]
    transformer = LinearTransformer(weights, biases)
    result = transformer([])  # Empty list should just return biases
    assert result == pytest.approx(jnp.array([5, 6]))


def test_call_with_out_reshape():
    weights = [[1, 2], [3, 4]]
    biases = [5, 6]
    transformer = LinearTransformer(weights, biases, out_reshape=(2, 1))
    free_params = [0.5, 0.25]
    result = transformer(free_params)
    expected = jnp.array([[6], [8.5]])
    assert result.shape == (2, 1)
    assert result == pytest.approx(expected)


def test_repr():
    weights = [[1, 2], [3, 4]]
    biases = [5, 6]
    transformer = LinearTransformer(weights, biases)
    assert (
        repr(transformer)
        == "LinearTransformer(weights=(2, 2), biases=(2,), out_reshape=None)"
    )
