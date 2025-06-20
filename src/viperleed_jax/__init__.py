"""ViPErLEED JAX Plugin."""

from jax import config

from .atom_basis import AtomBasis
from .parameter_space import ParameterSpace
from .tensor_calculator import TensorLEEDCalculator

# set JAX configuration
config.update('jax_debug_nans', False)
config.update('jax_enable_x64', True)
config.update('jax_disable_jit', False)
config.update('jax_log_compiles', False)
