"""Module from_object."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-04-17'

from jax import config

config.update('jax_debug_nans', False)
config.update('jax_enable_x64', True)
config.update('jax_disable_jit', False)
config.update('jax_log_compiles', False)

import numpy as np
from viperleed.calc import LOGGER as logger
from viperleed.calc.files.phaseshifts import readPHASESHIFTS

from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.data_structures import process_tensors
from viperleed_jax.files import phaseshifts as ps
from viperleed_jax.files.displacements.file import DisplacementsFile
from viperleed_jax.files.tensors import read_tensor_zip
from viperleed_jax.parameter_space import ParameterSpace
from viperleed_jax.tensor_calculator import TensorLEEDCalculator
from viperleed_jax.utils import check_jax_devices


def calculator_from_objects(
    slab,
    rpars,
    tensor_path,
    displacements_path,
    t_leed_l_max=None,
    recalculate_ref_tmatrices=None,
    **kwargs,
):
    # set LMAX cutoff
    ref_calc_lmax = rpars.LMAX.max
    if t_leed_l_max is None:
        t_leed_l_max = ref_calc_lmax

    # decide on T-matrix recalculation
    if recalculate_ref_tmatrices is None:
        recalculate_ref_tmatrices = rpars.SEARCH_RECALC_TMATRICES

    # log info or warning on used GPU/CPU
    check_jax_devices()

    # load and read the DISPLACEMENTS file
    disp_file = DisplacementsFile()
    disp_file.read(displacements_path)

    # Create the parameter space.
    # We do this now, because if anything fails here, we don't want to waste
    # time reading the tensor files.
    logger.debug('Creating parameter space.')
    atom_basis = AtomBasis(slab)
    parameter_space = ParameterSpace(atom_basis, rpars)

    # take the blocks from the displacements file
    # TODO: take care of multiple blocks!

    offsets_block = disp_file.offsets_block()
    search_block = (
        disp_file.first_block()
    )  # TODO,FIXME: can only do first block for now
    parameter_space.apply_displacements(offsets_block, search_block)

    # parameters needed to interpret the tensor data
    ref_calc_lmax = rpars.LMAX.max
    n_beams = len(rpars.ivbeams)
    n_energies = len(
        np.arange(
            rpars.THEO_ENERGIES.start,  # TODO: would be good to make this a property of the EnergyRange class
            rpars.THEO_ENERGIES.stop + 0.01,
            rpars.THEO_ENERGIES.step,
        )
    )

    logger.info(f'Starting to interpret tensor file {tensor_path.name}.')
    logger.debug(
        f'Reading tensor file with lmax={ref_calc_lmax},'
        f'n_beams={n_beams}, n_energies={n_energies}.'
    )

    # read tensor file
    tensors = read_tensor_zip(tensor_path, ref_calc_lmax, n_beams, n_energies)

    logger.debug('Finished reading tensor file.')

    non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
    sorted_tensors = [tensors[f'T_{at.num}'] for at in non_bulk_atoms]

    # Combine data into a ReferenceData object
    logger.debug('Combining tensor data ...')
    ref_calc_params, ref_calc_results = process_tensors(
        sorted_tensors, fix_lmax=t_leed_l_max
    )
    logger.debug('Tensor processing successful.')

    # read Phaseshift data using existing phaseshift reader
    phaseshifts_path = calc_path / 'PHASESHIFTS'
    _, raw_phaseshifts, _, _ = readPHASESHIFTS(
        slab, rpars, readfile=phaseshifts_path, check=True, ignoreEnRange=False
    )

    # get site element order
    site_el_map = ps.phaseshift_site_el_order(slab, rpars)

    # interpolate phaseshifts
    phaseshifts = ps.Phaseshifts(
        raw_phaseshifts,
        ref_calc_params.energies,
        t_leed_l_max,
        phaseshift_map=site_el_map,
    )

    logger.debug('Initializing Tensor LEED calculator.')
    calculator = TensorLEEDCalculator(
        ref_calc_params,
        ref_calc_results,
        phaseshifts,
        slab,
        rpars,
        recalculate_ref_t_matrices=recalculate_ref_tmatrices,
        **kwargs,
    )

    # free up memory for large objects that are no longer needed
    del sorted_tensors
    del tensors
    del raw_phaseshifts

    calculator.set_parameter_space(parameter_space)

    return calculator
