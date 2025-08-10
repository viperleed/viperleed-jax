"""Module utils."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-02-27'

import csv
import datetime
import logging
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

logger = logging.getLogger(__name__)


def _simple_sum_hash(x):
    return hash(x.tobytes())


class HashableArray:
    """Hashable Array wrapper for JAX.

    Helps to make JAX arrays hashable, so they can be used static args in jit.
    This speeds up the computation of functions that use these arrays as
    arguments.

    Based on https://github.com/google/jax/issues/4572#issuecomment-709809897.
    """

    def __init__(self, val):
        self.val = val

    def __hash__(self):
        """Calculate the hash of the array."""
        return _simple_sum_hash(self.val)

    def __eq__(self, other):
        """Check equality of two HashableArray instances."""
        return isinstance(other, HashableArray) and jnp.all(
            jnp.equal(self.val, other.val)
        )


def check_jax_devices():
    """Check if JAX can detect GPU devices.

    If no GPU devices are detected, a warning is logged.
    """
    try:
        gpu_devices = jax.devices('gpu')
    except RuntimeError:
        logger.warning(
            'JAX could not detect any GPU devices. The execution will default '
            'to CPU only which can be significantly slower.'
        )
        return
    logger.info(f'JAX detected {len(gpu_devices)} GPU(s).')

def check_jax_compilation_cache():
    """Check if JAX compilation cache is set.

    If not set, a warning is logged. The compilation cache can significantly
    speed up the execution of JAX functions by reusing compiled code across
    different runs.
    """
    if jax.config.jax_compilation_cache_dir is None:
        logger.warning(
            'JAX compilation cache is not set. This can lead to slower '
            'performance due to repeated compilations. See JAX documentation '
            'for instruction on how to set it up.'
        )
    else:
        logger.debug(
            'JAX compilation cache is set to '
            f'{jax.config.jax_compilation_cache_dir}.'
        )


def get_best_v0r_on_grid(calculator, parameters):
    meta_tree = calculator.parameter_space.meta_tree
    v0r_trafo = meta_tree.root.transformer_to_descendent(meta_tree.v0r_node)
    inverse_v0r_trafo = v0r_trafo.pseudo_inverse()
    v0r_step_params = [
        inverse_v0r_trafo(step) for step in calculator._v0r_shift_steps
    ]

    v0r_r_factors = []
    for v0r_step in v0r_step_params:
        param_vector = np.concatenate([v0r_step, parameters[1:]])
        v0r_r_factors.append(calculator.R(param_vector))
    argmin = np.argmin(v0r_r_factors)
    best_v0r_param = v0r_step_params[np.argmin(v0r_r_factors)]
    if argmin == 0 or argmin == len(v0r_step_params):
        best_v0r = calculator._v0r_transformer(best_v0r_param)[0]
        msg = 'Pre-optimized V0r shift is at the edge of the given range. '
        msg += f'Best shift = {best_v0r:.1f} eV.'
        logger.warning(msg)
    return best_v0r_param


def benchmark_calculator(
    calculator,
    free_params=None,
    n_repeats=5,
    csv_file_path=None,
    use_grad=True,
):
    """
    Benchmarks the execution of two methods on the given calculator object.

    Parameters
    ----------
      calculator: an object with methods jit_R and jit_grad_R.
      free_params: parameters passed to the jit functions.
      n_repeats: number of times to repeat the timed execution.
      csv_file_path: (optional) path to a CSV file where benchmark results will
        be appended.

    Returns
    -------
      A tuple of (r_fac_compile_time, r_fac_time, grad_compile_time, grad_time).
    """
    if n_repeats < 1:
        raise ValueError('Number of repeats must be greater than 0.')

    if free_params is None:
        free_params = np.array([0.55] * calculator.n_free_parameters)

    # timer object
    perf = time.perf_counter

    # --- Benchmark for jit_R (R factor) ---
    # Measure compile time
    start = perf()
    try:
        calculator.R(free_params).block_until_ready()
    except Exception as e:
        if 'RESOURCE_EXHAUSTED: Out of memory' in str(e):
            msg = (
                'Ran out of memory during R-factor calculation. Try using '
                'a smaller l_max cutoff or decreasing the used batch sizes.'
            )
            logger.error(msg)
            raise RuntimeError(msg) from e
        raise e
    r_fac_compile_time = perf() - start

    # Measure average execution time over n_repeats
    start_total = perf()
    for _ in range(n_repeats):
        calculator.R(free_params).block_until_ready()
    r_fac_time = (perf() - start_total) / n_repeats

    # If the compile time is less than 3x the average execution time,
    # it likely means the function was already compiled.
    if r_fac_compile_time < 3 * r_fac_time:
        logger.warning(
            'R factor compilation time is suspiciously low. '
            'The function may have been precompiled.'
        )

    # --- Benchmark for grad_R (gradients) ---
    if use_grad:
        start = perf()
        try:
            calculator.grad_R(free_params).block_until_ready()
        except Exception as e:
            if 'RESOURCE_EXHAUSTED: Out of memory' in str(e):
                msg = (
                    'Ran out of memory during gradient calculation. Try '
                    'using a smaller l_max cutoff or decreasing the used '
                    'batch sizes. Alternatively, you can use methods that '
                    'do not require gradients.'
                )
                logger.error(msg)
                raise RuntimeError(msg) from e
            raise e
        grad_compile_time = perf() - start

        start_total = perf()
        for _ in range(n_repeats):
            calculator.grad_R(free_params).block_until_ready()
        grad_time = (perf() - start_total) / n_repeats

        if grad_compile_time < 3 * grad_time:
            logger.warning(
                'Gradient compilation time is suspiciously low. '
                'The function may have been precompiled.'
            )
    else:
        grad_compile_time = np.nan
        grad_time = np.nan

    # Prepare the results with a timestamp
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'l_max': calculator.max_l_max,
        'r_fac_compile_time': r_fac_compile_time,
        'r_fac_time': r_fac_time,
        'grad_compile_time': grad_compile_time,
        'grad_time': grad_time,
    }

    # Optionally append the results to a CSV file
    if csv_file_path:
        try:
            file_path = Path(csv_file_path)
            write_header = not file_path.exists()
            with file_path.open('a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=results.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(results)
        except Exception as e:
            logger.error(f'Error writing to CSV file: {e}')

    return r_fac_compile_time, r_fac_time, grad_compile_time, grad_time


def format_benchmark_results(results):
    """Format the benchmark results for display.

    Parameters
    ----------
      results: a tuple of (r_compile_time, r_time,
        grad_compile_time, grad_time).

    Returns
    -------
      A formatted string.
    """
    r_fac_compile_time, r_fac_time, grad_compile_time, grad_time = results

    result_str = (
        'Benchmark results:\n'
        f'\tR factor comp. time:\t{r_fac_compile_time:.4f} s\n'
        f'\tR factor exec. time:\t{1000 * r_fac_time:.4f} ms\n'
    )
    if not np.isnan(grad_compile_time) and not np.isnan(grad_time):
        result_str += (
            f'\tGrad comp. time:\t{grad_compile_time:.4f} s\n'
            f'\tGrad exec. time:\t{1000 * grad_time:.4f} ms\n'
        )
    else:
        result_str += '\tGradients not evaluated.\n'
    return result_str


def estimate_function_cost(f, *args):
    fun_cost = jax.jit(f).lower(*args).compile().cost_analysis()
    fun_cost = int(fun_cost[0]['flops'])
    jac_cost = jax.jit(jax.jacfwd(f)).lower(*args).compile().cost_analysis()
    jac_cost = int(jac_cost[0]['flops'])
    logger.info(
        f'Function Cost:\t{fun_cost} FLOPS\n'
        f'Jacfwd Cost:\t{jac_cost} FLOPS\n'
    )
