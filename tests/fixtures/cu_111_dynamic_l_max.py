import os
from pathlib import Path

import numpy as np

import pytest

from tests.fixtures.calc_info import DeltaAmplitudeCalcInfo, DeltaAmplitudeReferenceData

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


######################
# Cu111 dynamic LMAX #
######################

@pytest.fixture(scope='session')
def cu_111_dynamic_l_max_info():
    input_path=Path(__file__).parent.parent / 'test_data' / 'Cu_111_dynamic_lmax'
    tensor_path=input_path / 'Tensors' / 'Tensors_001.zip'
    return DeltaAmplitudeCalcInfo(
        input_path=input_path,
        tensor_path=tensor_path,
        n_beams=9,
        max_l_max=10,
        energies = np.array([
            2.28680897,  2.4583211 ,  2.63103557,  2.80476761,  2.97936988,
            3.15472341,  3.33073044,  3.50731039,  3.68439579,  3.86193037,
            4.03986502,  4.2181592 ,  4.39677715,  4.57568788,  4.75486517,
            4.93428421,  5.11392546,  5.29376984,  5.47380161,  5.65400648,
            5.83437061,  6.01488304,  6.1955328 ,  6.37631083,  6.55720854,
            6.73821783,  6.91933107,  7.10054302,  7.281847  ,  7.46323729,
            7.64471006,  7.82625914,  8.00788212,  8.18957233,  8.37132931,
            8.55314732,  8.7350235 ,  8.91695595,  9.09894276,  9.28097916,
            9.46306324,  9.6451931 ,  9.82736778, 10.00958443, 10.19184017,
            10.37413406, 10.55646515, 10.73883247, 10.92123318, 11.10366535,
            11.286129  ], dtype=np.float64)
    )

@pytest.fixture(scope="session")
def cu_111_dynamic_l_max_delta_file(cu_111_dynamic_l_max_info,):
    delta_path = str(cu_111_dynamic_l_max_info.input_path / 'Deltas') + '/'
    raw = delta_transform(cu_111_dynamic_l_max_tensor_calculator.ref_data.n_energies,
                            delta_path,
                            ['DEL_1_Cu_1', 'DEL_2_Cu_1', 'DEL_3_Cu_1', 'DEL_4_Cu_1', 'DEL_5_Cu_1'])
    cu_111_dynamic_l_max_info.reference_delta_amps = TensErLEEDDeltaReferenceData(raw)


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

@pytest.fixture(scope='session')
def cu_111_dynamic_l_max_calculator_with_parameter_space(cu_111_dynamic_l_max_tensor_calculator, cu_111_dynamic_l_max_parameter_space):
    calculator = cu_111_dynamic_l_max_tensor_calculator
    calculator.set_parameter_space(cu_111_dynamic_l_max_parameter_space)
    return calculator

@pytest.fixture(scope='session')
def cu_111_dynamic_l_max_parameter_delta_file(cu_111_dynamic_l_max_info):
    delta_path = str(cu_111_dynamic_l_max_info.inputs_path / 'Deltas') + '/'
    return delta_transform(cu_111_dynamic_l_max_info.n_energies,
                            delta_path,
                            ['DEL_1_Cu_1', 'DEL_2_Cu_1', 'DEL_3_Cu_1', 'DEL_4_Cu_1', 'DEL_5_Cu_1'])
