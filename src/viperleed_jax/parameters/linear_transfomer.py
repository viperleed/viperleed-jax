import jax.numpy as jnp


class LinearTransformer:
    """Linear transformation class that applies a weight matrix and a bias vector to an input."""

    def __init__(self, weights, biases, out_reshape=None):
        self.weights = jnp.array(weights)
        self.n_free_params = self.weights.shape[1]
        self.biases = jnp.array(biases)
        self.out_reshape = out_reshape

    def __call__(self, free_params):
        if self.n_free_params == 0:
            return self.biases
        if isinstance(free_params, float):
            free_params = jnp.array([free_params])
        free_params = jnp.array(free_params)
        if len(free_params) != self.n_free_params:
            raise ValueError("Free parameters have wrong shape")
        result = self.weights @ free_params + self.biases
        if self.out_reshape is not None:
            result = result.reshape(self.out_reshape)
        return result

    def __repr__(self):
        return f"LinearTransformer(weights={self.weights.shape}, biases={self.biases.shape}, out_reshape={self.out_reshape})"
