"""Module utils."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-02-27'

import jax


def estimate_function_cost(f, *args):
    fun_cost = jax.jit(f).lower(*args).compile().cost_analysis()
    fun_cost = int(fun_cost[0]['flops'])
    jac_cost = jax.jit(jax.jacfwd(f)).lower(*args).compile().cost_analysis()
    jac_cost = int(jac_cost[0]['flops'])
    print(
        f'Function Cost:\t{fun_cost} FLOPS\n'
        f'Jacfwd Cost:\t{jac_cost} FLOPS\n'
    )
