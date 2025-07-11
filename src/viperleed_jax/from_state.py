"""Module from_state."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-08-28'

import logging
import os
import shutil
import tempfile
from pathlib import Path

from viperleed.calc import LOGGER
from viperleed.calc.run import run_calc

from viperleed_jax.from_objects import (
    setup_tl_calculator,
    setup_tl_parameter_space,
)


def calculator_from_paths(
    inputs_path, tensor_path, displacements_id=0, **kwargs
):
    """Create a TensorLEEDCalculator from input paths.

    This function initializes the ViPErLEED calculation from the given input
    paths, reads the necessary files, including DISPLACEMENTS, and sets up the
    parameter space for the calculator. If the displacements_id is provided,
    it skips to the corresponding block in the DISPLACEMENTS file.

    Parameters
    ----------
    inputs_path : path-like
        Path to the directory containing the input files for the viperleed.calc
        initialization.
    tensor_path : path-like
        Path to the tensor file, which contains the pre-calculated tensors for
        the calculator.
    displacements_path : path-like
        Path to the DISPLACEMENTS file, which contains the displacements data.
    displacements_id : int, optional
        The index of the displacements block to use from the DISPLACEMENTS file.
        Defaults to 0, which means the first block will be used.
    **kwargs : dict
        Additional keyword arguments passed to `setup_tl_calculator`.

    Returns
    -------
    TensorLEEDCalculator
        A TensorLEEDCalculator object initialized with the slab, rpars, and
        the parameter space set up from the input files.
    """
    # Run ViPErLEED initialization from inputs_path
    last_state = run_viperleed_initialization(inputs_path)
    # get objects from last_state
    slab, rpars = last_state.slab, last_state.rpars

    # Create the parameter space.
    # We do this now, because if anything fails here, we don't want to waste
    # time reading the tensor files.

    # create parameter space
    parameter_space = setup_tl_parameter_space(slab, rpars)

    disp_file = rpars.vlj_displacements

    if disp_file.offsets is not None:
        LOGGER.debug('Applying offsets from displacements file.')
        parameter_space.apply_offsets(disp_file.offsets)

    # skip ahead to the block with the given displacements_id
    for _ in range(displacements_id + 1):
        search_block = disp_file.next(2.0)
    parameter_space.apply_search_segment(search_block)

    phaseshifts_path = Path(inputs_path) / 'PHASESHIFTS'
    # delegate to calculator_from_objects
    calculator = setup_tl_calculator(
        slab, rpars, tensor_path, phaseshifts_path, **kwargs
    )
    # set the parameter space
    calculator.set_parameter_space(parameter_space)
    return calculator


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
                'tmp_init', preset_params={
                    'RUN': [0],'BACKEND': {'search': 'viperleed-jax'}}
            )
        finally:  # always change back to the original directory
            os.chdir(home)

    if exit_code != 0:
        msg = f'ViPErLEED Initialization failed with exit code {exit_code}.'
        raise RuntimeError(msg)

    console_handler = logging.StreamHandler()

    LOGGER.addHandler(console_handler)
    LOGGER.info('ViPErLEED initialization successful')

    # get slab and rparams from state
    return state_recorder.last_state
