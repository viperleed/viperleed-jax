import jax
from jax import config
config.update("jax_debug_nans", False)
config.update("jax_enable_x64", True)
config.update("jax_disable_jit", False)
config.update("jax_log_compiles", False)
import jax.numpy as jnp

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

import tempfile
import os, shutil
import logging
import zipfile

from viperleed_jax.data_structures import ReferenceData
from viperleed_jax.tensor_calculator import TensorLEEDCalculator
from viperleed_jax.files import phaseshifts as ps
from viperleed_jax.files.tensors import read_tensor_zip
from viperleed_jax.parameter_space import SiteEl

from viperleed.calc.run import run_calc
from viperleed.calc.files.displacements import readDISPLACEMENTS
from viperleed.calc import LOGGER as logger
from viperleed.calc.files.phaseshifts import readPHASESHIFTS
from viperleed.calc.files.iorfactor import beamlist_to_array


def calculator_from_state(calc_path, tensor_path, l_max:int, **kwargs):

    last_state = run_viperleed_initialization(calc_path)
    slab, rpars = last_state.slab, last_state.rpars

    if not rpars.expbeams:
        raise RuntimeError('No (pseudo)experimental beams loaded. This is required '
                        'for the structure optimization.')

    # # read DISPLACEMENTS if not yet loaded
    # if not rpars.fileLoaded['DISPLACEMENTS']:
    #     readDISPLACEMENTS(rpars, calc_path/'DISPLACEMENTS')
    #     rpars.fileLoaded['DISPLACEMENTS'] = True
    #     logger.debug('DISPLACEMENTS file loaded')

    # parameters needed to interpret the tensor data
    ref_calc_lmax = rpars.LMAX.max
    n_beams = len(rpars.ivbeams)
    n_energies = len(np.arange(rpars.THEO_ENERGIES.start,                           # TODO: would be good to make this a property of the EnergyRange class
                            rpars.THEO_ENERGIES.stop+0.01,
                            rpars.THEO_ENERGIES.step))

    logger.info(f'Starting to interpret tensor file {tensor_path.name}.')
    logger.debug(f'Using lmax={ref_calc_lmax}, n_beams={n_beams}, '
                 f'n_energies={n_energies}.')

    # read tensor file
    tensors = read_tensor_zip(tensor_path, ref_calc_lmax, n_beams, n_energies)

    logger.debug(f'Finished reading tensor file.')

    non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
    sorted_tensors = [tensors[f'T_{at.num}'] for at in non_bulk_atoms]

    # Combine data into a ReferenceData object
    logger.debug('Combining tensor data into ReferenceData object.')
    ref_data = ReferenceData(sorted_tensors, fix_lmax=l_max)
    logger.debug('ReferenceData object created successfully.')

    # read Phaseshift data using existing phaseshift reader
    phaseshifts_path = calc_path / 'PHASESHIFTS'
    _, raw_phaseshifts, _, _ = readPHASESHIFTS(
        slab, rpars, readfile=phaseshifts_path, check=True, ignoreEnRange=False)

    # get site element order
    site_el_map = ps.phaseshift_site_el_order(slab, rpars)

    # interpolate phaseshifts
    phaseshifts = ps.Phaseshifts(raw_phaseshifts, ref_data.energies, l_max,
                                 phaseshift_map=site_el_map)

    iv_beam_indices = [beam.hk for beam in rpars.ivbeams]

    logger.debug('Initializing Tensor LEED calculator.')
    calculator = TensorLEEDCalculator(ref_data, phaseshifts, slab,
                                      rpars, **kwargs)

    # free up memory for large objects that are no longer needed
    del sorted_tensors
    del tensors
    del raw_phaseshifts

    return calculator, slab, rpars, ref_data, phaseshifts

def run_viperleed_initialization(calc_path):
    """Runs ViPErLEED initialization with the input data in calc_path.

    The calculation runs in a temporary directory, so the input data is not
    modified.
    """
    with tempfile.TemporaryDirectory() as tmp_calc_dir:
        tmp_calc_path = Path(tmp_calc_dir)
        # copy all files from calc_path to tmp_calc_path
        shutil.copytree(calc_path, tmp_calc_path, dirs_exist_ok=True)

        home = Path.cwd()
        # run viperleed.calc with the test data, but only intialization
        os.chdir(tmp_calc_path)
        try:
            exit_code, state_recorder = run_calc('test_unrelaxed', preset_params={'RUN':[0]})
        finally: # always change back to the original directory
            os.chdir(home)

    if exit_code != 0:
        raise RuntimeError(f'ViPErLEED Initialization failed with exit code {exit_code}')

    consoleHandler = logging.StreamHandler()

    logger.addHandler(consoleHandler)
    logger.info('ViPErLEED initialization successful')

    # get slab and rparams from state
    last_state = state_recorder.last_state
    return last_state
