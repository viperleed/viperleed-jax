import numpy as np
import jax.numpy as jnp


class LinearTransformer:
    """Linear transformation class that applies a weight matrix and a bias vector to an input."""

    def __init__(self, weights, biases, out_reshape=None):
        self.weights = np.array(weights)
        self.n_free_params = self.weights.shape[1]
        self.biases = np.array(biases)
        self.out_reshape = out_reshape

        self._in_dim = self.n_free_params
        self._out_dim = self.biases.shape[0]

        # consistency check of dimensions
        if self.weights.shape[0] != self._out_dim:
            raise ValueError(
                f"Weight matrix shape {self.weights.shape} does not match "
                f"bias shape {self.biases.shape}"
            )
        if self.out_reshape is not None:
            if self._out_dim != np.prod(self.out_reshape):
                raise ValueError(
                    f"Output reshape {self.out_reshape} does not match bias "
                    f"shape {self.biases.shape}"
                )

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def out_dim(self):
        return self._out_dim

    def __call__(self, free_params):
        if self.n_free_params == 0:
            return self.biases
        free_params = jnp.asarray(free_params)
        if len(free_params) != self.n_free_params:
            raise ValueError("Free parameters have wrong shape")
        result = self.weights @ free_params + self.biases  # Ax + b
        if self.out_reshape is not None:
            result = result.reshape(self.out_reshape)
        return result

    def __repr__(self):
        return f"LinearTransformer(weights={self.weights.shape}, biases={self.biases.shape}, out_reshape={self.out_reshape})"

    def tree_flatten(self):
        aux_data = {
            "n_free_params": self.n_free_params,
            "weights": self.weights,
            "biases": self.biases,
            "out_reshape": self.out_reshape,
            "_in_dim": self._in_dim,
            "_out_dim": self._out_dim,
        }
        children = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, children, aux_data):
        frozen_parameter_space = cls.__new__
        for kw, value in aux_data.items():
            setattr(frozen_parameter_space, kw, value)
        return frozen_parameter_space
