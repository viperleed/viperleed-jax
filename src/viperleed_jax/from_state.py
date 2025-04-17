"""Module from_state."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-08-28'

from jax import config

config.update('jax_debug_nans', False)
config.update('jax_enable_x64', True)
config.update('jax_disable_jit', False)
config.update('jax_log_compiles', False)
import logging
import os
import shutil
import tempfile

from viperleed.calc import LOGGER as logger
from viperleed.calc.run import run_calc

from from_objects import calculator_from_objects

def calculator_from_paths(inputs_path,
                          tensor_path,
                          displacements_path,
                          **kwargs):

    # Run ViPErLEED initialization from inputs_path
    last_state = run_viperleed_initialization(inputs_path)
    # get objects from last_state
    slab, rpars = last_state.slab, last_state.rpars

    # delegate to calculator_from_objects
    return calculator_from_objects(
        slab, rpars, tensor_path, displacements_path, **kwargs
    )

def run_viperleed_initialization(calc_path):
    """Run ViPErLEED initialization with the input data in calc_path.

    The calculation runs in a temporary directory, so the input data is not
    modified.
    """
    with tempfile.TemporaryDirectory() as tmp_calc_dir:
        tmp_calc_path = Path(tmp_calc_dir)
        # copy all files from calc_path to tmp_calc_path
        shutil.copytree(calc_path, tmp_calc_path, dirs_exist_ok=True)

        home = Path.cwd()
        # run viperleed.calc with the test data, but only initialization
        os.chdir(tmp_calc_path)
        try:
            exit_code, state_recorder = run_calc(
                'test_unrelaxed', preset_params={'RUN': [0]}
            )
        finally:  # always change back to the original directory
            os.chdir(home)

    if exit_code != 0:
        msg = f'ViPErLEED Initialization failed with exit code {exit_code}.'
        raise RuntimeError(msg)

    console_handler = logging.StreamHandler()

    logger.addHandler(console_handler)
    logger.info('ViPErLEED initialization successful')

    # get slab and rparams from state
    return state_recorder.last_state
