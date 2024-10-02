import os
from pathlib import Path

import numpy as np

import pytest
from pytest_cases import fixture

from tests.fixtures.base import LARGE_FILE_PATH
from tests.fixtures.calc_info import DeltaAmplitudeCalcInfo

from viperleed.calc.files.phaseshifts import readPHASESHIFTS

from viperleed_jax.from_state import run_viperleed_initialization
from viperleed_jax.files.tensors import read_tensor_zip
from viperleed_jax.data_structures import ReferenceData
from viperleed_jax.files import phaseshifts as ps
from viperleed_jax.tensor_calculator import TensorLEEDCalculator
from viperleed_jax.parameter_space import ParameterSpace
from viperleed_jax.parameters.geo_parameters import GeoParamBound
from viperleed_jax.parameters.vib_parameters import VibParamBound
from viperleed_jax.parameters.v0r_parameters import V0rParamBound

_DATA_PATH = Path(__file__).parent.parent / 'test_data' / 'Fe2O3_012' /'converged'
_LARGE_DATA_PATH = LARGE_FILE_PATH / 'Fe2O3_012' / 'converged'

_REFERENCE_FILE_PATH_Z = _LARGE_DATA_PATH / 'Fe2O3_012_TensErLEED_reference_z.npz'
_REFERENCE_DATA_Z = np.load(_REFERENCE_FILE_PATH_Z)
_COMPARE_PARAMS_Z = _REFERENCE_DATA_Z['parameters']
_COMPARE_DELTA_AMPLITUDES_Z = _REFERENCE_DATA_Z['tenserleed_delta_amplitudes_z']
_COMPARE_PARAMS_AMPS_Z = [(_COMPARE_PARAMS_Z[i,:], _COMPARE_DELTA_AMPLITUDES_Z[i,:])
                        for i in range(len(_COMPARE_PARAMS_Z))]
_COMPARE_ABS_Z = 8e-4 # for l_max=10

_REFERENCE_FILE_PATH_X = _LARGE_DATA_PATH / 'Fe2O3_012_TensErLEED_reference_x.npz'
_REFERENCE_DATA_X = np.load(_REFERENCE_FILE_PATH_X)
_COMPARE_PARAMS_X = _REFERENCE_DATA_X['parameters']
_COMPARE_DELTA_AMPLITUDES_X = _REFERENCE_DATA_X['tenserleed_delta_amplitudes_x']
_COMPARE_PARAMS_AMPS_X = [(_COMPARE_PARAMS_X[i,:], _COMPARE_DELTA_AMPLITUDES_X[i,:])
                        for i in range(len(_COMPARE_PARAMS_X))]
_COMPARE_ABS_X = 5e-4 # for l_max=10

######################
#   Converged Fe2O3  #
######################

@fixture(scope='session')
def fe2o3_012_converged_info():
    input_path = _DATA_PATH
    tensor_path = _LARGE_DATA_PATH / 'Tensors' / 'Tensors_001.zip'
    return DeltaAmplitudeCalcInfo(
        input_path=input_path,
        tensor_path=tensor_path,
        n_beams=38,
        max_l_max=10,
        energies = np.array([
        1.54641092,  1.65665889,  1.76690674,  1.86872446,  1.97021472,
        2.072294  ,  2.17489934,  2.27797675,  2.38147926,  2.48536658,
        2.58960319,  2.69415855,  2.79900551,  2.90411973,  3.00948   ,
        3.11506748,  3.22086477,  3.32685661,  3.43302965,  3.53937078,
        3.64586902,  3.75251436,  3.85929704,  3.96620846,  4.07324171,
        4.18038845,  4.28764296,  4.39499855,  4.50244999,  4.60999203,
        4.7176199 ,  4.82532883,  4.93311548,  5.04097557,  5.14890528,
        5.25690174,  5.36496115,  5.47308111,  5.58125877,  5.68949223,
        5.79777718,  5.90611362,  6.01449823,  6.1229291 ,  6.2314043 ,
        6.3399229 ,  6.44848156,  6.55707979,  6.66571617,  6.77438879,
        6.88309669,  6.99183893,  7.10061264,  7.20941877,  7.31825495,
        7.42712021,  7.5360136 ,  7.64493513,  7.75388193,  7.86285543,
        7.97185326,  8.08087444,  8.18991947,  8.29898739,  8.40807629,
        8.51718712,  8.62631702,  8.73546791,  8.84463787,  8.9538269 ,
        9.0630331 ,  9.17225838,  9.28149986,  9.39075851,  9.50003338,
        9.6093235 ,  9.71862888,  9.82795048,  9.93728638, 10.04663563,
       10.15599918, 10.26537609, 10.37476635, 10.48416996, 10.59358501,
       10.70301247, 10.81245327, 10.92190361, 11.03136635, 11.14084053,
       11.2503252 , 11.35981941, 11.46932602, 11.57884121, 11.68836594,
       11.79790211, 11.90744591, 12.01699924, 12.12656307, 12.23613358,
       12.34571362, 12.45530319, 12.5649004 , 12.67450523, 12.78411865,
       12.89373875, 13.00336742, 13.11300373, 13.22264671, 13.33229733,
       13.44195557, 13.55161953, 13.66129017, 13.77096844, 13.88065243,
       13.99034309, 14.10004044, 14.2097435 , 14.31945229, 14.42916775,
       14.53888893, 14.64861488, 14.75834751, 14.86808491, 14.97782803,
       15.08757782, 15.19733047, 15.30708981, 15.4168539 , 15.52662277,
       15.63639641, 15.74617481, 15.85595798, 15.96574593, 16.07553864,
       16.18533516, 16.29513741, 16.40494347, 16.51475143, 16.62456703,
       16.73438644, 16.84420586, 16.9540329 , 17.06386375, 17.17369843,
       17.28353691, 17.39337921, 17.50322342, 17.61307335, 17.72292709,
       17.83278084, 17.94264221, 18.05250549, 18.16237068, 18.27224159,
       18.38211632, 18.49199104, 18.60187149, 18.71175385, 18.82164001,
       18.93152809, 19.04142189, 19.15131569, 19.2612133 , 19.37111473,
       19.48101807, 19.59092522, 19.70083427, 19.81074524, 19.92066193,
       20.03057861, 20.14049721, 20.25042152, 20.36034775, 20.47027397,
       20.58020401, 20.69013786, 20.80007172, 20.91000938, 21.01995087,
       21.12989235, 21.23983765, 21.34978485, 21.45973206, 21.56968307,
       21.67963791, 21.78959274, 21.89955139, 22.00951195, 22.1194725 ,
       22.22943687, 22.33940315, 22.44937134, 22.55934143, 22.66931534,
       22.77928734, 22.88926506, 22.99924278, 23.1092205 , 23.21920395,
       23.32918739, 23.43917274, 23.54916   , 23.65914917, 23.76913834,
       23.87913132, 23.98912621, 24.09912109],
        dtype=np.float64)
    )


@fixture(scope='session')
def fe2o3_012_converged_state_after_init(fe2o3_012_converged_info):
    state_after_init = run_viperleed_initialization(fe2o3_012_converged_info.input_path)
    slab, rpars = state_after_init.slab, state_after_init.rpars
    return slab, rpars

@fixture(scope='session')
def fe2o3_012_converged_raw_phaseshifts(fe2o3_012_converged_info, fe2o3_012_converged_state_after_init):
    slab, rpars = fe2o3_012_converged_state_after_init
    phaseshifts_path = fe2o3_012_converged_info.input_path / 'PHASESHIFTS'
    _, raw_phaseshifts, _, _ = readPHASESHIFTS(
        slab, rpars, readfile=phaseshifts_path, check=True, ignoreEnRange=False)
    return raw_phaseshifts

@fixture(scope='session')
def fe2o3_012_converged_phaseshifts(fe2o3_012_converged_info,
                                     fe2o3_012_converged_state_after_init,
                                     fe2o3_012_converged_raw_phaseshifts):
    slab, rpars = fe2o3_012_converged_state_after_init
    phaseshift_map = ps.phaseshift_site_el_order(slab, rpars)
    return ps.Phaseshifts(
        fe2o3_012_converged_raw_phaseshifts,
        fe2o3_012_converged_info.energies,
        l_max=fe2o3_012_converged_info.max_l_max,
        phaseshift_map=phaseshift_map)


@fixture(scope='session')
def fe2o3_012_converged_read_tensor_zip(fe2o3_012_converged_info):
    return read_tensor_zip(fe2o3_012_converged_info.tensor_path,
                           lmax=fe2o3_012_converged_info.max_l_max,
                           n_beams=fe2o3_012_converged_info.n_beams,
                           n_energies=fe2o3_012_converged_info.n_energies)


@fixture(scope='session')
def fe2o3_012_converged_read_ref_data(fe2o3_012_converged_read_tensor_zip,
                                       fe2o3_012_converged_state_after_init):
    slab, _ = fe2o3_012_converged_state_after_init
    non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
    sorted_tensors = [fe2o3_012_converged_read_tensor_zip[f'T_{at.num}']
                      for at in non_bulk_atoms]

    return ReferenceData(sorted_tensors, fix_lmax=10) # TODO: fix!!

@fixture(scope='session')
def fe2o3_012_converged_tensor_calculator(fe2o3_012_converged_state_after_init, fe2o3_012_converged_phaseshifts, fe2o3_012_converged_read_ref_data):
    slab, rpars = fe2o3_012_converged_state_after_init
    calculator = TensorLEEDCalculator(fe2o3_012_converged_read_ref_data,
                                fe2o3_012_converged_phaseshifts,
                                slab,
                                rpars,
                                recalculate_ref_t_matrices=False)
    return calculator

@fixture(scope='session')
def fe2o3_012_converged_tensor_calculator_recalc_t_matrices(fe2o3_012_converged_state_after_init, fe2o3_012_converged_phaseshifts, fe2o3_012_converged_read_ref_data):
    slab, rpars = fe2o3_012_converged_state_after_init
    calculator = TensorLEEDCalculator(fe2o3_012_converged_read_ref_data,
                                fe2o3_012_converged_phaseshifts,
                                slab,
                                rpars,
                                recalculate_ref_t_matrices=True)
    return calculator

@fixture(scope='session')
def fe2o3_012_converged_parameter_space(fe2o3_012_converged_state_after_init):
    slab, _ = fe2o3_012_converged_state_after_init
    parameter_space = ParameterSpace(slab)

    ## GEOMETRY
    # Fix layers 0 and 1
    parameter_space.geo_params.fix_layer(0, z_offset=0.)
    parameter_space.geo_params.fix_layer(1, z_offset=0.)

    # symmetry constrained xyz movements ± 0.15 A for layer 2
    for param in [p for p in parameter_space.geo_params.terminal_params if p.bound is None]:
        param.set_bound(GeoParamBound(-0.15, +0.15))

    ## VIBRATIONS
    # fix *_def sites (O_def, Fe_def)
    for param in [p for p in parameter_space.vib_params.terminal_params
                  if p.site_element.site.endswith('_def')]:
        parameter_space.vib_params.fix_site_element(param.site_element, None) # None fixes to the default value

    # the rest can vary ± 0.05 A
    for param in [p for p in parameter_space.vib_params.terminal_params
                  if p.site_element.site.endswith('_surf')]:
        param.set_bound(VibParamBound(-0.05, +0.05))

    ## CHEMISTRY
    # no free parameters
    parameter_space.occ_params.remove_remaining_vacancies()

    # V0R
    # set ± 2 eV
    parameter_space.v0r_param.set_bound(V0rParamBound(-2., +2.))

    return parameter_space

@fixture(scope='session')
def fe2o3_012_converged_calculator_with_parameter_space(fe2o3_012_converged_tensor_calculator, fe2o3_012_converged_parameter_space):
    calculator = fe2o3_012_converged_tensor_calculator
    calculator.set_parameter_space(fe2o3_012_converged_parameter_space)
    return calculator

@fixture(scope='session')
def fe2o3_012_converged_calculator_with_parameter_space_recalc_t_matrices(fe2o3_012_converged_tensor_calculator_recalc_t_matrices,
                                                                         fe2o3_012_converged_parameter_space):
    calculator = fe2o3_012_converged_tensor_calculator_recalc_t_matrices
    calculator.set_parameter_space(fe2o3_012_converged_parameter_space)
    return calculator

@fixture(scope='session')
@pytest.mark.parametrize('parameters, reference_delta_amplitudes',
                         _COMPARE_PARAMS_AMPS_Z,
                         ids=[str(p) for p in _COMPARE_PARAMS_Z])
def fe2o3_012_converged_tenserleed_reference_z(parameters,
                                              reference_delta_amplitudes):
    return parameters, reference_delta_amplitudes, _COMPARE_ABS_Z

@fixture(scope='session')
@pytest.mark.parametrize('parameters, reference_delta_amplitudes',
                         _COMPARE_PARAMS_AMPS_X,
                         ids=[str(p) for p in _COMPARE_PARAMS_X])
def fe2o3_012_converged_tenserleed_reference_x(parameters,
                                              reference_delta_amplitudes):
    return parameters, reference_delta_amplitudes, _COMPARE_ABS_X
