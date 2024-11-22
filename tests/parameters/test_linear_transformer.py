import jax.numpy as jnp
import pytest

from viperleed_jax.parameters.linear_transformer import (
    LinearTransformer,
    Transformer,
)


class TestLinearTransformer:
    """Test the functionality of the LinearTransformer class."""

    def test_linear_transformer_initialization(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = LinearTransformer(weights, biases)
        assert transformer.n_free_params == 2
        assert transformer.weights.shape == (2, 2)
        assert transformer.biases.shape == (2,)
        assert transformer.out_reshape is None

    def test_call_with_correct_shape(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = LinearTransformer(weights, biases)
        free_params = [0.5, 0.25]
        result = transformer(free_params)
        expected = jnp.array(
            [6.0, 8.5]
        )  # [1*0.5 + 2*0.25 + 5, 3*0.5 + 4*0.25 + 6]
        assert result == pytest.approx(expected)

    def test_call_with_incorrect_shape(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = LinearTransformer(weights, biases)
        free_params = [0.5]  # Incorrect length
        with pytest.raises(
            ValueError, match='Free parameters have wrong shape'
        ):
            transformer(free_params)

    def test_call_without_free_params(self):
        weights = jnp.zeros(shape=(2, 0))
        biases = [5, 6]
        transformer = LinearTransformer(weights, biases)
        result = transformer([])  # Empty list should just return biases
        assert result == pytest.approx(jnp.array([5, 6]))

    def test_call_with_out_reshape(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = LinearTransformer(weights, biases, out_reshape=(2, 1))
        free_params = [0.5, 0.25]
        result = transformer(free_params)
        expected = jnp.array([[6], [8.5]])
        assert result.shape == (2, 1)
        assert result == pytest.approx(expected)

    def test_repr(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = LinearTransformer(weights, biases)
        assert (
            repr(transformer)
            == 'LinearTransformer(weights=(2, 2), biases=(2,), out_reshape=None)'
        )


class TestAdheresToTransformerABC:
    """Test that LinearTransformer adheres to the Transformer ABC."""

    def test_transformer_implements_required_methods(self):
        class DummyTransformer(Transformer):
            def __call__(self, input_params):
                pass

            def in_dim(self):
                return 0

            def out_dim(self):
                return 0

            def compose(self, other):
                pass

            def __eq__(self, other):
                return True

        dummy_transformer = DummyTransformer()
        assert isinstance(dummy_transformer, Transformer)

    def test_linear_transformer_adheres_to_transformer_abc(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = LinearTransformer(weights, biases)

        assert hasattr(transformer, '__call__')
        assert callable(transformer)
        assert hasattr(transformer, 'in_dim')
        assert hasattr(transformer, 'out_dim')
        assert hasattr(transformer, 'compose')
        assert hasattr(transformer, '__eq__')

    def test_linear_transformer_compose_with_abc(self):
        weights1 = [[1, 2], [3, 4]]
        biases1 = [5, 6]
        transformer1 = LinearTransformer(weights1, biases1)

        weights2 = [[0.5, 0.25], [0.75, 1.5]]
        biases2 = [1, -1]
        transformer2 = LinearTransformer(weights2, biases2)

        composed_transformer = transformer2.compose(transformer1)

        assert composed_transformer.in_dim == transformer1.in_dim
        assert composed_transformer.out_dim == transformer2.out_dim
