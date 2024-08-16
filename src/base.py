from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

# Transformer
@register_pytree_node_class
class LinearTransformer():
    def __init__(self, weights, biases, out_reshape=None):
        self.weights = jnp.array(weights)
        self.n_free_params = self.weights.shape[1]
        self.biases = jnp.array(biases)
        self.out_reshape = out_reshape

    def __call__(self, free_params):
        if self.n_free_params == 0:
            return self.biases
        if len(free_params) != self.n_free_params:
            raise ValueError("Free parameters have wrong shape")
        result =  self.weights @ free_params + self.biases
        if self.out_reshape is not None:
            result =  result.reshape(self.out_reshape)
        return result

    def tree_flatten(self):
        aux_data = {
            'n_free_params': self.n_free_params,
            'weights': self.weights,
            'biases': self.biases,
            'out_reshape': self.out_reshape
        }
        children = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, children, aux_data):
        frozen_parameter_space = cls.__new__
        for kw, value in aux_data.items():
            setattr(frozen_parameter_space, kw, value)
        return frozen_parameter_space
