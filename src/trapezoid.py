"""Hacky copy of jax.scipy.integrate.trapezoid because that is not available in python 3.8 apparently..."""
from __future__ import annotations
from functools import partial

import jax
from jax import numpy as jnp

from jax._src.numpy import util
from jax._src.typing import Array, ArrayLike
import jax.numpy as jnp

@partial(jax.jit, static_argnames=('axis',))
def trapezoid(y: ArrayLike, x: ArrayLike | None = None, dx: ArrayLike = 1.0,
              axis: int = -1) -> Array:
  # TODO(phawkins): remove this annotation after fixing jnp types.
  dx_array: Array
  if x is None:
    util.check_arraylike('trapezoid', y)
    y_arr, = util.promote_dtypes_inexact(y)
    dx_array = jnp.asarray(dx)
  else:
    util.check_arraylike('trapezoid', y, x)
    y_arr, x_arr = util.promote_dtypes_inexact(y, x)
    if x_arr.ndim == 1:
      dx_array = jnp.diff(x_arr)
    else:
      dx_array = jnp.moveaxis(jnp.diff(x_arr, axis=axis), axis, -1)
  y_arr = jnp.moveaxis(y_arr, axis, -1)
  return 0.5 * (dx_array * (y_arr[..., 1:] + y_arr[..., :-1])).sum(-1)
