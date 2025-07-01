"""Module for dense quantum number indexing."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-02-17'

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

MAXIMUM_LMAX = 18

# load precalculated dense quantum numbers
_FULL_DENSE_QUANTUM_NUMBERS = np.load(
    Path(__file__).parent / 'dense_quantum_numbers.npy', allow_pickle=False
)


# TODO: come up with a faster version of this
def _asymmetric_dense_quantum_numbers(lmax_1, lmax_2):
    valid_quantum_numbers = np.empty(
        ((lmax_1 + 1) * (lmax_1 + 1), (lmax_2 + 1) * (lmax_2 + 1), 4), dtype=int
    )
    for L in range(lmax_1 + 1):
        for LP in range(lmax_2 + 1):
            for M in range(-L, L + 1):
                for MP in range(-LP, LP + 1):
                    valid_quantum_numbers[(L + 1) * (L + 1) - L + M - 1][
                        (LP + 1) * (LP + 1) - LP + MP - 1
                    ] = [L, LP, M, MP]
    # map the array
    return jnp.array(valid_quantum_numbers)


def _dense_quantum_numbers(lmax):
    return _asymmetric_dense_quantum_numbers(lmax, lmax)


DENSE_QUANTUM_NUMBERS = {
    l: _FULL_DENSE_QUANTUM_NUMBERS[: (l + 1) ** 2, : (l + 1) ** 2, :]
    for l in range(2 * MAXIMUM_LMAX + 1)
}
DENSE_L = {
    l: DENSE_QUANTUM_NUMBERS[l][:, 0, 0] for l in range(2 * MAXIMUM_LMAX + 1)
}
DENSE_M = {
    l: DENSE_QUANTUM_NUMBERS[l][:, 0, 2] for l in range(2 * MAXIMUM_LMAX + 1)
}
MINUS_ONE_POW_M = {
    l: jnp.power(-1, DENSE_M[l]) for l in range(2 * MAXIMUM_LMAX + 1)
}


def map_l_array_to_compressed_quantum_index(array, LMAX):
    """Map an array of shape (LMAX+1) to a compressed quantum number index.

    Takes an array of shape (LMAX+1) with values for each L and maps it to a
    dense form of shape (LMAX+1)*(LMAX+1) with the values for each L replicated
    (2L+1) times. I.e. an array
    [val(l=0), val(l=1), val(l=2), ...] is mapped to
    [val(l=0), val(l=1), val(l=1), val(l=1), val(l=2), ...].
    """
    if array.shape[0] != LMAX + 1:
        raise ValueError('Array shape does not match LMAX')
    broadcast_l_index = DENSE_L[LMAX]
    return jnp.asarray(array)[broadcast_l_index]

_vmapped_l_array_to_compressed_quantum_index = jax.vmap(
    jax.vmap(map_l_array_to_compressed_quantum_index, in_axes=(0, None)),
    in_axes=(0, None),
)

vmapped_l_array_to_compressed_quantum_index = jax.jit(
    _vmapped_l_array_to_compressed_quantum_index,
    static_argnums=(1,),
)
