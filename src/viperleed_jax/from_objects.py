"""Module from_object."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-04-17'

import numpy as np
from viperleed.calc import LOGGER as logger
from viperleed.calc.files.phaseshifts import readPHASESHIFTS

from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.files import phaseshifts as ps
from viperleed_jax.files.tensors import read_tensor_zip
from viperleed_jax.parameter_space import ParameterSpace
from viperleed_jax.ref_calc_data import process_tensors
from viperleed_jax.tensor_calculator import TensorLEEDCalculator
from viperleed_jax.utils import check_jax_devices


def setup_tl_parameter_space(slab, rpars):
    """Create a ParameterSpace object for the given slab and rpars.

    This function initializes the atom basis and creates a ParameterSpace
    that contains and parses all the symmetry information from the slab.

    Parameters
    ----------
    slab : Slab
        The slab object containing the atomic structure.
    rpars : Rparams
        The Run parameters from viperleed.calc.

    Returns
    -------
    ParameterSpace
        A ParameterSpace object that contains the atom basis and symmetry
        information.
    """
    logger.debug('Creating parameter space.')
    atom_basis = AtomBasis(slab)
    return ParameterSpace(atom_basis, rpars)


def setup_tl_calculator(
    slab,
    rpars,
    tensor_path,
    phaseshifts_path,
    **kwargs,
):
    """Set up a TensorLEEDCalculator from slab, rpars, and tensor file.

    This function reads the tensor file, processes the tensors, and initializes
    a TensorLEEDCalculator object. It also reads the phaseshift data and
    prepares the necessary parameters for the calculator.

    Parameters
    ----------
    slab : Slab
        The slab object containing the atomic structure.
    rpars : Rparams
        The Run parameters from viperleed.calc.
    tensor_path : path-like
        The path to the tensor file.
    phaseshifts_path : path-like
        The path to the phaseshift file.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the TensorLEEDCalculator.

    Returns
    -------
    TensorLEEDCalculator
        The TensorLEEDCalculator object. The calculator is not yet initialized,
        with a parameter space.
    """
    # log info or warning on used GPU/CPU
    check_jax_devices()

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

    # determine the l_max for the LEED calculation
    t_leed_l_max = _determine_l_max(rpars)

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
    logger.debug('Reading and interpolating phaseshifts.')
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
        **kwargs,
    )

    # explicitly free up memory for large objects that are no longer needed
    del sorted_tensors
    del tensors
    del raw_phaseshifts

    return calculator


def _determine_l_max(rpars):
    if rpars.VLJ_CONFIG['t-leed-l_max'] == -1:
        # use the maximum l_max from the reference calculation
        return rpars.LMAX.max

    # use the l_max from the rparams
    return rpars.VLJ_CONFIG['t-leed-l_max']
