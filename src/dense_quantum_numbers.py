"""Module for dense quantum number indexing"""
from functools import partial
import numpy as np
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit, vmap

MAXIMUM_LMAX = 18

# TODO: come up with a faster version of this
def _dense_quantum_numbers(LMAX):
    valid_quantum_numbers = np.empty(((LMAX+1)*(LMAX+1), (LMAX+1)*(LMAX+1), 4), dtype=int)
    for L in range(LMAX+1):
        for LP in range(LMAX+1):
            for M in range(-L, L+1):
                for MP in range(-LP, LP+1):
                    valid_quantum_numbers[(L+1)*(L+1)-L+M-1][(LP+1)*(LP+1)-LP+MP-1] = [L, LP, M, MP]
    return jnp.array(valid_quantum_numbers)

_FULL_DENSE_QUANTUM_NUMBERS = _dense_quantum_numbers(2*MAXIMUM_LMAX)
DENSE_QUANTUM_NUMBERS = {
    l: _FULL_DENSE_QUANTUM_NUMBERS[:(l+1)**2,:(l+1)**2,:]
    for l in range(2*MAXIMUM_LMAX+1)
}
DENSE_L = {
    l: DENSE_QUANTUM_NUMBERS[l][:,0,0] for l in range(2*MAXIMUM_LMAX+1)
}
DENSE_M = {
    l: DENSE_QUANTUM_NUMBERS[l][:,0,2] for l in range(2*MAXIMUM_LMAX+1)
}
MINUS_ONE_POW_M = {
    l: jnp.power(-1, DENSE_M[l]) for l in range(2*MAXIMUM_LMAX+1)
}

@partial(jit, static_argnames=('LMAX',))
def map_l_array_to_compressed_quantum_index(array, LMAX):
    """Takes an array of shape (LMAX+1) with values for each L and maps it to a
    dense form of shape (LMAX+1)*(LMAX+1) with the values for each L replicated
    (2L+1) times. I.e. an array
    [val(l=0), val(l=1), val(l=2), ...] is mapped to
    [val(l=0), val(l=1), val(l=1), val(l=1), val(l=2), ...].
    """
    broadcast_l_index = DENSE_L[LMAX]
    mapped_array = jnp.asarray(array)[broadcast_l_index]
    return mapped_array
