import numpy as np
import pytest

from viperleed_jax.transformation_tree.linear_transformer import (
    LinearMap,
    AffineTransformer,
    Transformer,
)


class TestAffineTransformer:
    """Test the functionality of the AffineTransformer class."""

    def test_linear_transformer_initialization(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = AffineTransformer(weights, biases)
        assert transformer.n_free_params == 2
        assert transformer.weights.shape == (2, 2)
        assert transformer.biases.shape == (2,)
        assert transformer.out_reshape is None

    def test_call_with_correct_shape(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = AffineTransformer(weights, biases)
        free_params = [0.5, 0.25]
        result = transformer(free_params)
        expected = np.array(
            [6.0, 8.5]
        )  # [1*0.5 + 2*0.25 + 5, 3*0.5 + 4*0.25 + 6]
        assert result == pytest.approx(expected)

    def test_call_with_incorrect_shape(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = AffineTransformer(weights, biases)
        free_params = [0.5]  # Incorrect length
        with pytest.raises(
            ValueError, match='Free parameters have wrong shape'
        ):
            transformer(free_params)

    def test_call_without_free_params(self):
        weights = np.zeros(shape=(2, 0))
        biases = [5, 6]
        transformer = AffineTransformer(weights, biases)
        result = transformer([])  # Empty list should just return biases
        assert result == pytest.approx(np.array([5, 6]))

    def test_call_with_out_reshape(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = AffineTransformer(weights, biases, out_reshape=(2, 1))
        free_params = [0.5, 0.25]
        result = transformer(free_params)
        expected = np.array([[6], [8.5]])
        assert result.shape == (2, 1)
        assert result == pytest.approx(expected)

    def test_repr(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = AffineTransformer(weights, biases)
        assert (
            repr(transformer)
            == 'AffineTransformer(weights=(2, 2), biases=(2,), out_reshape=None)'
        )


class TestAffineTransformerCompositions:
    """Test the composition behavior of the AffineTransformer class."""

    def test_composition_of_two_transformers(self):
        weights1 = np.array([[1, 2], [3, 4]])
        biases1 = np.array([5, 6])
        transformer1 = AffineTransformer(weights1, biases1)

        weights2 = np.array([[0.5, 0.25], [0.75, 1.5]])
        biases2 = np.array([1, -1])
        transformer2 = AffineTransformer(weights2, biases2)

        composed_transformer = transformer2.compose(transformer1)

        # Expected composition
        expected_weights = weights1 @ weights2
        expected_biases = weights1 @ biases2 + biases1

        assert composed_transformer.weights == pytest.approx(expected_weights)
        assert composed_transformer.biases == pytest.approx(expected_biases)

    def test_composition_with_identity_transformer(self):
        """Ensure that composing with an identity transformer does not change the behavior."""
        identity_weights = np.eye(2)
        identity_biases = np.array([0, 0])
        identity_transformer = AffineTransformer(
            identity_weights, identity_biases
        )

        weights = np.array([[1, 2], [3, 4]])
        biases = np.array([5, 6])
        transformer = AffineTransformer(weights, biases)

        # Compose with identity
        composed_transformer = transformer.compose(identity_transformer)

        # Identity composition should result in the same transformer
        assert composed_transformer.weights == pytest.approx(
            transformer.weights
        )
        assert composed_transformer.biases == pytest.approx(transformer.biases)

    def test_composition_with_mismatched_dimensions(self):
        weights1 = np.array([[1, 2], [3, 4]])
        biases1 = np.array([5, 6])
        transformer1 = AffineTransformer(weights1, biases1)

        weights2 = np.array([[0.5, 0.25, 0.1], [0.75, 1.5, 0.2]])
        biases2 = np.array([1, -1])
        transformer2 = AffineTransformer(weights2, biases2)

        # Expect a ValueError due to dimension mismatch
        with pytest.raises(ValueError):
            transformer1.compose(transformer2)

    def test_composition_of_three_transformers(self, subtests):
        weights1 = np.array([[1, 2], [3, 4]])
        biases1 = np.array([5, 6])
        transformer1 = AffineTransformer(weights1, biases1)

        weights2 = np.array([[0.5, 0.25], [0.75, 1.5]])
        biases2 = np.array([1, -1])
        transformer2 = AffineTransformer(weights2, biases2)

        weights3 = np.array([[2, 0], [0, 0.5]])
        biases3 = np.array([0.5, 0.25])
        transformer3 = AffineTransformer(weights3, biases3)

        composed_transformer = transformer3.compose(transformer2).compose(
            transformer1
        )

        # Expected composition
        intermediate_weights = weights1 @ weights2
        intermediate_biases = weights1 @ biases2 + biases1
        expected_weights = intermediate_weights @ weights3
        expected_biases = intermediate_weights @ biases3 + intermediate_biases

        with subtests.test('transformer weights'):
            assert composed_transformer.weights == pytest.approx(
                expected_weights
            )
        with subtests.test('transformer biases'):
            assert composed_transformer.biases == pytest.approx(expected_biases)

        # Test that the composition has the expected effect on input
        input_params = [0.5, 1.0]  # Example input
        intermediate_result = transformer3(input_params)
        intermediate_result = transformer2(intermediate_result)
        final_result = transformer1(intermediate_result)

        composed_result = composed_transformer(input_params)
        with subtests.test('apply to input'):
            assert composed_result == pytest.approx(final_result)


class TestAdheresToTransformerABC:
    """Test that Affineransformer adheres to the Transformer ABC."""

    def test_linear_transformer_adheres_to_transformer_abc(self):
        weights = [[1, 2], [3, 4]]
        biases = [5, 6]
        transformer = AffineTransformer(weights, biases)

        assert hasattr(transformer, '__call__')
        assert callable(transformer)
        assert hasattr(transformer, 'in_dim')
        assert hasattr(transformer, 'out_dim')
        assert hasattr(transformer, 'compose')
        assert hasattr(transformer, '__eq__')

    def test_linear_transformer_compose_with_abc(self):
        weights1 = [[1, 2], [3, 4]]
        biases1 = [5, 6]
        transformer1 = AffineTransformer(weights1, biases1)

        weights2 = [[0.5, 0.25], [0.75, 1.5]]
        biases2 = [1, -1]
        transformer2 = AffineTransformer(weights2, biases2)

        composed_transformer = transformer2.compose(transformer1)

        assert composed_transformer.in_dim == transformer1.in_dim
        assert composed_transformer.out_dim == transformer2.out_dim


class TestLinearMap:
    """Test the functionality of the LinearMap class."""

    def test_linear_map_initialization(self):
        weights = [[1, 2], [3, 4]]
        linear_map = LinearMap(weights)
        assert linear_map.n_free_params == 2
        assert linear_map.weights.shape == (2, 2)
        assert linear_map.biases == pytest.approx(0)  # Biases must be zero
        assert linear_map.out_reshape is None

    def test_call_with_correct_shape(self):
        weights = [[1, 2], [3, 4]]
        linear_map = LinearMap(weights)
        free_params = [0.5, 0.25]
        result = linear_map(free_params)
        expected = np.array([1.0, 2.5])  # [1*0.5 + 2*0.25, 3*0.5 + 4*0.25]
        assert result == pytest.approx(expected)

    def test_call_with_incorrect_shape(self):
        weights = [[1, 2], [3, 4]]
        linear_map = LinearMap(weights)
        free_params = [0.5]  # Incorrect length
        with pytest.raises(
            ValueError, match='Free parameters have wrong shape'
        ):
            linear_map(free_params)

    def test_call_with_out_reshape(self):
        weights = [[1, 2], [3, 4]]
        linear_map = LinearMap(weights, out_reshape=(2, 1))
        free_params = [0.5, 0.25]
        result = linear_map(free_params)
        expected = np.array([[1.0], [2.5]])
        assert result.shape == (2, 1)
        assert result == pytest.approx(expected)

    def test_repr(self):
        weights = [[1, 2], [3, 4]]
        linear_map = LinearMap(weights)
        assert (
            repr(linear_map)
            == 'LinearMap(weights=(2, 2), biases=(2,), out_reshape=None)'
        )

    def test_composition_with_linear_transformer(self):
        weights1 = np.array([[1, 2], [3, 4]])
        weights2 = np.array([[0.5, 0.25], [0.75, 1.5]])
        biases2 = np.array([1, -1])
        linear_map = LinearMap(weights1)
        transformer = AffineTransformer(weights2, biases2)

        composed_transformer = transformer.compose(linear_map)

        # Validate composition
        expected_weights = weights1 @ weights2
        expected_biases = weights1 @ biases2
        assert composed_transformer.weights == pytest.approx(expected_weights)
        assert composed_transformer.biases == pytest.approx(expected_biases)

    def test_composition_with_linear_map(self):
        weights1 = np.array([[1, 2], [3, 4]])
        weights2 = np.array([[0.5, 0.25], [0.75, 1.5]])
        linear_map1 = LinearMap(weights1)
        linear_map2 = LinearMap(weights2)

        composed_map = linear_map2.compose(linear_map1)

        # Validate composition
        expected_weights = weights1 @ weights2
        assert composed_map.weights == pytest.approx(expected_weights)
        assert composed_map.biases == pytest.approx(0)
