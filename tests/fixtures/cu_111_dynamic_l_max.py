import os
from pathlib import Path

import numpy as np

import pytest
from pytest_cases import fixture

from tests.fixtures.calc_info import DeltaAmplitudeCalcInfo

from viperleed.calc.files.phaseshifts import readPHASESHIFTS

from viperleed_jax.from_state import run_viperleed_initialization
from viperleed_jax.files.tensors import read_tensor_zip
from viperleed_jax.data_structures import ReferenceData
from viperleed_jax.files import phaseshifts as ps
from viperleed_jax.tensor_calculator import TensorLEEDCalculator
from viperleed_jax.files.deltas import Transform as delta_transform
from viperleed_jax.parameter_space import ParameterSpace
from viperleed_jax.parameters.geo_parameters import GeoParamBound
from viperleed_jax.parameters.vib_parameters import VibParamBound
from viperleed_jax.parameters.v0r_parameters import V0rParamBound
from viperleed_jax.files.deltas import Transform as delta_transform

_DATA_PATH = Path(__file__).parent.parent / 'test_data' / 'Cu_111' /'dynamic_l_max'
_REFERENCE_FILE_PATH = _DATA_PATH / 'Cu_111_dynamic_l_max_TensErLEED_reference.npz'
_REFERENCE_DATA = np.load(_REFERENCE_FILE_PATH)

_COMPARE_PARAMS = _REFERENCE_DATA['parameters']
_COMPARE_DELTA_AMPLITUDES = _REFERENCE_DATA['tenserleed_delta_amplitudes']
_COMPARE_PARAMS_AMPS = [(_COMPARE_PARAMS[i,:], _COMPARE_DELTA_AMPLITUDES[i,:])
                        for i in range(len(_COMPARE_PARAMS))]
_COMPARE_ABS = 8.8e-5

######################
# Cu111 dynamic LMAX #
######################

@pytest.fixture(scope='session')
def cu_111_dynamic_l_max_info():
    input_path = _DATA_PATH
    tensor_path = input_path / 'Tensors' / 'Tensors_001.zip'
    return DeltaAmplitudeCalcInfo(
        input_path=input_path,
        tensor_path=tensor_path,
        n_beams=9,
        max_l_max=10,
        energies = np.array([
        2.21859169,  2.32100487,  2.42391562,  2.52727366,  2.63103557,
        2.73516273,  2.8396225 ,  2.94438601,  3.04942703,  3.15472341,
        3.26025462,  3.36600304,  3.47195196,  3.57808733,  3.68439579,
        3.7908659 ,  3.89748669,  4.00424814,  4.11114168,  4.2181592 ,
        4.32529306,  4.43253708,  4.53988361,  4.64732838,  4.75486517,
        4.86248875,  4.97019529,  5.07798052,  5.18583965,  5.29376984,
        5.40176773,  5.50982904,  5.61795235,  5.72613382,  5.83437061,
        5.94266081,  6.0510025 ,  6.15939236,  6.26782942,  6.37631083,
        6.48483562,  6.59340143,  6.70200729,  6.81065083,  6.91933107,
        7.02804708,  7.13679647,  7.24557924,  7.35439348,  7.46323729,
        7.57211161,  7.68101406,  7.78994322,  7.89889955,  8.00788212,
        8.11688805,  8.22591877,  8.33497238,  8.44404888,  8.55314732,
        8.66226578,  8.77140617,  8.88056564,  8.98974419,  9.09894276,
        9.20815849,  9.3173914 ,  9.42664242,  9.53590965,  9.6451931 ,
        9.75449276,  9.86380768,  9.9731369 , 10.08248138, 10.19184017,
        10.30121231, 10.41059875, 10.51999664, 10.62940788, 10.73883247,
        10.84826851, 10.95771694, 11.06717682, 11.17664719, 11.286129 ],
        dtype=np.float64)
    )


@pytest.fixture(scope='session')
def cu_111_dynamic_l_max_state_after_init(cu_111_dynamic_l_max_info):
    state_after_init = run_viperleed_initialization(cu_111_dynamic_l_max_info.input_path)
    slab, rpars = state_after_init.slab, state_after_init.rpars
    return slab, rpars

@pytest.fixture(scope='session')
def cu_111_dynamic_l_max_raw_phaseshifts(cu_111_dynamic_l_max_info, cu_111_dynamic_l_max_state_after_init):
    slab, rpars = cu_111_dynamic_l_max_state_after_init
    phaseshifts_path = cu_111_dynamic_l_max_info.input_path / 'PHASESHIFTS'
    _, raw_phaseshifts, _, _ = readPHASESHIFTS(
        slab, rpars, readfile=phaseshifts_path, check=True, ignoreEnRange=False)
    return raw_phaseshifts

@pytest.fixture(scope='session')
def cu_111_dynamic_l_max_phaseshifts(cu_111_dynamic_l_max_info,
                                     cu_111_dynamic_l_max_state_after_init,
                                     cu_111_dynamic_l_max_raw_phaseshifts):
    slab, rpars = cu_111_dynamic_l_max_state_after_init
    phaseshift_map = ps.phaseshift_site_el_order(slab, rpars)
    return ps.Phaseshifts(
        cu_111_dynamic_l_max_raw_phaseshifts,
        cu_111_dynamic_l_max_info.energies,
        l_max=cu_111_dynamic_l_max_info.max_l_max,
        phaseshift_map=phaseshift_map)


@pytest.fixture(scope='session')
def cu_111_dynamic_l_max_read_tensor_zip(cu_111_dynamic_l_max_info):
    return read_tensor_zip(cu_111_dynamic_l_max_info.tensor_path,
                           lmax=cu_111_dynamic_l_max_info.max_l_max,
                           n_beams=cu_111_dynamic_l_max_info.n_beams,
                           n_energies=cu_111_dynamic_l_max_info.n_energies)


@pytest.fixture(scope='session')
def cu_111_dynamic_l_max_read_ref_data(cu_111_dynamic_l_max_read_tensor_zip,
                                       cu_111_dynamic_l_max_state_after_init):
    slab, _ = cu_111_dynamic_l_max_state_after_init
    non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
    sorted_tensors = [cu_111_dynamic_l_max_read_tensor_zip[f'T_{at.num}']
                      for at in non_bulk_atoms]

    return ReferenceData(sorted_tensors, fix_lmax=10) # TODO: fix!!

@pytest.fixture(scope='session')
def cu_111_dynamic_l_max_tensor_calculator(cu_111_dynamic_l_max_state_after_init, cu_111_dynamic_l_max_phaseshifts, cu_111_dynamic_l_max_read_ref_data):
    slab, rpars = cu_111_dynamic_l_max_state_after_init
    calculator = TensorLEEDCalculator(cu_111_dynamic_l_max_read_ref_data,
                                cu_111_dynamic_l_max_phaseshifts,
                                slab,
                                rpars,
                                recalculate_ref_t_matrices=False)
    return calculator

@pytest.fixture(scope='session')
def cu_111_dynamic_l_max_parameter_space(cu_111_dynamic_l_max_state_after_init):
    slab, _ = cu_111_dynamic_l_max_state_after_init
    parameter_space = ParameterSpace(slab)
    ## GEOMETRY
    # Fix layer 0
    parameter_space.geo_params.fix_layer(1, z_offset=0.)
    parameter_space.geo_params.fix_layer(2, z_offset=0.)
    parameter_space.geo_params.fix_layer(3, z_offset=0.)
    parameter_space.geo_params.fix_layer(4, z_offset=0.)

    # symmetry constrained xyz movements ± 0.15 A for layer 2
    for param in [p for p in parameter_space.geo_params.terminal_params if p.bound is None]:
        param.set_bound(GeoParamBound(-0.05, + 0.05))

    ## VIBRATIONS
    # fix *_def sites (O_def, Fe_def)
    for param in [p for p in parameter_space.vib_params.terminal_params if p.site_element.site.endswith('_def')]:
        parameter_space.vib_params.fix_site_element(param.site_element, None) # None fixes to the default value
    # for param in [p for p in parameter_space.vib_params.terminal_params if p.site_element.site.endswith('_surf')]:
    #     parameter_space.vib_params.fix_site_element(param.site_element, None) # None fixes to the default value


    # # the rest can vary ± 0.05 A
    for param in [p for p in parameter_space.vib_params.terminal_params if p.site_element.site.endswith('_surf')]:
        param.set_bound(VibParamBound(-0.05, + 0.05))

    ## CHEMISTRY
    # no free parameters
    parameter_space.occ_params.remove_remaining_vacancies()

    # V0R
    # set ± 2 eV
    parameter_space.v0r_param.set_bound(V0rParamBound(-2., +2.))
    return parameter_space

@fixture(scope='session')
def cu_111_dynamic_l_max_calculator_with_parameter_space(cu_111_dynamic_l_max_tensor_calculator, cu_111_dynamic_l_max_parameter_space):
    calculator = cu_111_dynamic_l_max_tensor_calculator
    calculator.set_parameter_space(cu_111_dynamic_l_max_parameter_space)
    return calculator

@fixture(scope='session')
@pytest.mark.parametrize('parameters, reference_delta_amplitudes',
                         _COMPARE_PARAMS_AMPS,
                         ids=[str(p) for p in _COMPARE_PARAMS])
def cu_111_dynamic_l_max_tenserleed_reference(parameters,
                                              reference_delta_amplitudes):
    return parameters, reference_delta_amplitudes, _COMPARE_ABS
