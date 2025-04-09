"""Module linear_transformer."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-10-07'

from abc import ABC, abstractmethod

import numpy as np


class Transformer(ABC):
    """Abstract base class for transformations."""

    @abstractmethod
    def __call__(self, input_params):
        """Apply the transformation to the input parameters."""

    @abstractmethod
    def in_dim(self):
        """Input dimensionality of the transformer."""

    @abstractmethod
    def out_dim(self):
        """Output dimensionality of the transformer."""

    @abstractmethod
    def compose(self, other):
        """Compose this transformer with another transformer."""

    @abstractmethod
    def __eq__(self, other):
        """Check equality between two transformers."""


class LinearTransformer(Transformer):
    """Linear transformation class that implements an affine transformation."""

    def __init__(self, weights, biases, out_reshape=None):
        self.weights = np.array(weights)
        self.n_free_params = self.weights.shape[1]
        self.biases = np.array(biases)
        self.out_reshape = out_reshape

        self._in_dim = self.n_free_params
        self._out_dim = self.biases.shape[0]

        # consistency check of dimensions
        if self.weights.shape[0] != self._out_dim:
            msg = (
                f'Weight matrix shape {self.weights.shape} does not match '
                f'bias shape {self.biases.shape}'
            )
            raise ValueError(msg)
        if self.out_reshape is not None and self._out_dim != np.prod(
            self.out_reshape
        ):
            msg = (
                f'Output reshape {self.out_reshape} does not match '
                f'output dimension {self._out_dim}'
            )
            raise ValueError(msg)

    @property
    def in_dim(self):
        """Return the input dimensionality of the transformer."""
        return self._in_dim

    @property
    def out_dim(self):
        """Return the output dimensionality of the transformer."""
        return self._out_dim

    @property
    def is_injective(self):
        """Check if the transformation is injective."""
        return np.linalg.matrix_rank(self.weights) == self.in_dim

    @property
    def is_surjective(self):
        """Check if the transformation is surjective."""
        return np.linalg.matrix_rank(self.weights.T) == self.out_dim

    @property
    def is_bijective(self):
        """Check if the transformation is bijective."""
        return self.is_injective and self.is_surjective

    def __call__(self, free_params):
        """Apply the transformation to the input parameters."""
        if self.n_free_params == 0:
            result = self.biases
            if self.out_reshape is not None:
                return result.reshape(self.out_reshape)
            return result

        if len(free_params) != self.n_free_params:
            msg = 'Free parameters have wrong shape'
            raise ValueError(msg)
        result = self.weights @ free_params + self.biases  # Ax + b
        if self.out_reshape is not None:
            result = result.reshape(self.out_reshape)
        return result

    def __eq__(self, other):
        """Check equality between two transformers."""
        if not isinstance(other, LinearTransformer):
            return False
        if not np.array_equal(self.weights, other.weights):
            return False
        if not np.array_equal(self.biases, other.biases):
            return False
        if self.out_reshape is None and other.out_reshape is not None:
            return True
        return self.out_reshape == other.out_reshape

    def compose(self, other):
        """Compose this transformer with another transformer.

        Creates a new transformer object that applies the transformation of
        other first and then the transformation of this object. The new weights
        and biases are easily calculated by matrix multiplication.

        For compatible transformers l1, l2, l3 and input x, it holds that:
        l3(l2(l1(x))) == l1.compose(l2).compose(l3)(x)
        """
        if not isinstance(other, LinearTransformer):
            msg = 'Can only compose with another LinearTransformer'
            raise TypeError(msg)
        if self.out_dim != other.in_dim:
            msg = (
                f'Cannot compose transformers with shapes {self._out_dim} '
                'and {other.in_dim}'
            )
            raise ValueError(msg)
        new_weights = other.weights @ self.weights
        new_biases = other.weights @ self.biases + other.biases
        return LinearTransformer(new_weights, new_biases, other.out_reshape)

    def select_rows(self, bool_mask):
        """Select rows of the transformer based on a boolean mask.

        This method creates a new transformer with the same weights and biases,
        but only the rows of the weights and biases that are selected by the
        boolean mask. The boolean mask should be a 1D array of booleans with
        the same length as the number of rows in the transformer.
        """
        # check that bool mask is a valid shape
        _bool_mask = np.asarray(bool_mask)
        new_weights = self.weights[_bool_mask]
        new_biases = self.biases[_bool_mask]
        return LinearTransformer(new_weights, new_biases, (_bool_mask.sum(),))

    def __repr__(self):
        """Return a string representation of the transformer."""
        return (
            f'LinearTransformer(weights={self.weights.shape}, '
            f'biases={self.biases.shape}, out_reshape={self.out_reshape})'
        )

    def boolify(self):
        return LinearTransformer(
            np.bool_(self.weights), np.bool_(self.biases), self.out_reshape
        )


class LinearMap(LinearTransformer):
    """A linear map is a LinearTransformer with biases set to zero."""

    def __init__(self, weights, out_reshape=None):
        weights = np.array(weights)
        super().__init__(weights, np.zeros(weights.shape[0]), out_reshape)


def stack_transformers(transformers):
    """Stack a list of transformers into a single transformer."""
    weights = np.vstack([transformer.weights for transformer in transformers])
    biases = np.hstack([transformer.biases for transformer in transformers])
    return LinearTransformer(
        weights, biases, (np.sum([t.out_dim for t in transformers]),)
    )
