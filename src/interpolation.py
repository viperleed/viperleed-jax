"""Module interpolation

This module is a reworking of scipy's and my Bspline interpolation methods.
It can interpolate functions efficiently and in a JAX-compatible way."""

import jax
from jax import numpy as jnp
from scipy import interpolate

