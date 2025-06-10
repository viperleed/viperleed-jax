"""Module linear_transformer."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-10-07'

from abc import ABC, abstractmethod

import numpy as np

from viperleed_jax.lib_math import EPS

class Transformer(ABC):
    """Abstract base class for transformations."""

    @abstractmethod
    def __call__(self, input_params):
        """Apply the transformation to the input parameters."""

    @property
    @abstractmethod
    def in_dim(self):
        """Input dimensionality of the transformer."""

    @property
    @abstractmethod
    def out_dim(self):
        """Output dimensionality of the transformer."""

    @property
    @abstractmethod
    def is_injective(self):
        """Check if the transformation is injective."""

    @property
    @abstractmethod
    def is_surjective(self):
        """Check if the transformation is surjective."""

    @property
    def is_bijective(self):
        """Check if the transformation is bijective."""
        return self.is_injective and self.is_surjective

    @abstractmethod
    def compose(self, other):
        """Compose this transformer with another transformer."""

    @abstractmethod
    def __eq__(self, other):
        """Check equality between two transformers."""

    @abstractmethod
    def __hash__(self):
        """Calculate a hash for the transformer."""

    def __matmul__(self, other):
        """Compose this transformer with another using @ operator."""
        return self.compose(other)


class AffineTransformer(Transformer):
    """Class that implements an affine transformation."""

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
        if self.in_dim == 0:
            return False
        return np.linalg.matrix_rank(self.weights) == self.in_dim

    @property
    def is_surjective(self):
        """Check if the transformation is surjective."""
        if self.in_dim == 0:
            return False
        return np.linalg.matrix_rank(self.weights.T) == self.out_dim

    def __call__(self, free_params):
        """Apply the transformation to the input parameters."""
        if self.n_free_params == 0:
            result = self.biases
            if self.out_reshape is not None:
                return result.reshape(self.out_reshape)
            return result

        if free_params.size != self.n_free_params:
            msg = 'Free parameters have wrong shape'
            raise ValueError(msg)
        result = self.weights @ free_params.reshape(-1) + self.biases  # Ax + b
        if self.out_reshape is not None:
            result = result.reshape(self.out_reshape)
        return result

    def __eq__(self, other):
        """Check equality between two transformers."""
        if not isinstance(other, AffineTransformer):
            return False
        if not np.array_equal(self.weights, other.weights):
            return False
        if not np.array_equal(self.biases, other.biases):
            return False
        return self.out_reshape == other.out_reshape

    def compose(self, other):
        """Compose this transformer with another transformer.

        Creates a new transformer object that applies the transformation of
        other first and then the transformation of this object. The new weights
        and biases are easily calculated by matrix multiplication.

        For compatible transformers l1, l2, l3 and input x, it holds that:
        l3(l2(l1(x))) == l1.compose(l2).compose(l3)(x)
        """
        if not isinstance(other, AffineTransformer):
            msg = 'Can only compose with another AffineTransformer'
            raise TypeError(msg)
        if self.out_dim != other.in_dim:
            msg = (
                f'Cannot compose transformers with shapes {self._out_dim} '
                'and {other.in_dim}'
            )
            raise ValueError(msg)
        new_weights = other.weights @ self.weights
        new_biases = other.weights @ self.biases + other.biases
        return AffineTransformer(new_weights, new_biases, other.out_reshape)

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
        return AffineTransformer(new_weights, new_biases, (_bool_mask.sum(),))

    def pseudo_inverse(self):
            """Calculate the pseudo-inverse of the affine transformation."""
            if self.in_dim == 0:
                raise ValueError('Cannot calculate pseudo-inverse of zero map')

            # Create a homogeneous matrix representation of the affine transformation
            homogeneous_weights = np.block([
                [self.weights, self.biases],
                [np.zeros((1, self.in_dim)), np.ones(1)]
            ])

            # Calculate the pseudo-inverse of the homogeneous matrix
            pseudo_inverse_homogeneous = np.linalg.pinv(homogeneous_weights)

            # Extract the pseudo-inverse of the weights and the new biases
            pseudo_inverse_weights = pseudo_inverse_homogeneous[:self.in_dim, :self.out_dim]
            pseudo_inverse_biases = pseudo_inverse_homogeneous[:self.in_dim, self.out_dim]

            return AffineTransformer(pseudo_inverse_weights, pseudo_inverse_biases, self.out_reshape)

    def __repr__(self):
        """Return a string representation of the transformer."""
        return (
            f'AffineTransformer(weights={self.weights.shape}, '
            f'biases={self.biases.shape}, out_reshape={self.out_reshape})'
        )

    def boolify(self):
        return AffineTransformer(
            np.bool_(self.weights), np.bool_(self.biases), self.out_reshape
        )

    def __hash__(self):
        """Calculate a hash for the transformer.

        This is necessary for jit compilation in jax. The hash is calculated
        based on the weights, biases, and output reshape of the transformer.
        """
        return hash(
            (tuple(self.weights.flatten()),
             tuple(self.biases),
             self.out_reshape)
        )

class LinearMap(AffineTransformer):
    """A linear map is a AffineTransformer with biases set to zero."""

    def __init__(self, weights, out_reshape=None):
        weights = np.array(weights)
        super().__init__(weights, np.zeros(weights.shape[0]), out_reshape)

    def pseudo_inverse(self):
        """Calculate the pseudo-inverse of the linear map."""
        if self.in_dim == 0:
            raise ValueError('Cannot calculate pseudo-inverse of zero map')
        return LinearMap(np.linalg.pinv(self.weights))

    def __repr__(self):
        """Return a string representation of the transformer."""
        return (
            f'LinearMap(weights={self.weights.shape}, '
            f'out_reshape={self.out_reshape})'
        )

    def compose(self, other):
        """Compose this linear map with another transformer."""
        if not isinstance(other, LinearMap):
            msg = 'Can only compose with another LinearMap'
            return super().compose(other)
        if self.out_dim != other.in_dim:
            msg = (
                f'Cannot compose transformers with shapes {self._out_dim} '
                'and {other.in_dim}'
            )
            raise ValueError(msg)
        new_weights = other.weights @ self.weights
        return LinearMap(new_weights, other.out_reshape)


def stack_transformers(transformers):
    """Stack a list of transformers into a single transformer."""
    weights = np.vstack([transformer.weights for transformer in transformers])
    biases = np.hstack([transformer.biases for transformer in transformers])
    return AffineTransformer(
        weights, biases, (np.sum([t.out_dim for t in transformers]),)
    )
