from pathlib import Path

import numpy as np
import pytest
from pytest_cases import fixture
from viperleed.calc.files.new_displacements.file import DisplacementsFile
from viperleed.calc.files.phaseshifts import readPHASESHIFTS

from tests.fixtures.base import LARGE_FILE_PATH
from tests.fixtures.calc_info import DeltaAmplitudeCalcInfo
from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.files import phaseshifts as ps
from viperleed_jax.files.tensors import read_tensor_zip
from viperleed_jax.from_state import run_viperleed_initialization
from viperleed_jax.parameter_space import ParameterSpace
from viperleed_jax.ref_calc_data import process_tensors
from viperleed_jax.tensor_calculator import TensorLEEDCalculator

from .base import ComparisonTensErLEEDDeltaAmps

_DATA_PATH = (
    Path(__file__).parent.parent / 'test_data' / 'Cu_111' / 'fixed_l_max'
)
_LARGE_DATA_PATH = LARGE_FILE_PATH / 'Cu_111' / 'fixed_l_max'
_REFERENCE_FILE_PATH = (
    _LARGE_DATA_PATH / 'Cu_111_fixed_l_max_TensErLEED_reference.npz'
)
_REFERENCE_DATA = np.load(_REFERENCE_FILE_PATH)

_COMPARE_ABS = 13.5e-5
_comparison_data = ComparisonTensErLEEDDeltaAmps(_REFERENCE_DATA, _COMPARE_ABS)

######################
#    Cu111 LMAX=10   #
######################


@fixture(scope='session')
def cu_111_fixed_l_max_info():
    input_path = _DATA_PATH
    tensor_path = _LARGE_DATA_PATH / 'Tensors' / 'Tensors_001.zip'
    return DeltaAmplitudeCalcInfo(
        input_path=input_path,
        tensor_path=tensor_path,
        displacements_path=input_path / 'DISPLACEMENTS',
        n_beams=9,
        max_l_max=10,
        energies=np.array(
            [
                2.21859169,
                2.32100487,
                2.42391562,
                2.52727366,
                2.63103557,
                2.73516273,
                2.8396225,
                2.94438601,
                3.04942703,
                3.15472341,
                3.26025462,
                3.36600304,
                3.47195196,
                3.57808733,
                3.68439579,
                3.7908659,
                3.89748669,
                4.00424814,
                4.11114168,
                4.2181592,
                4.32529306,
                4.43253708,
                4.53988361,
                4.64732838,
                4.75486517,
                4.86248875,
                4.97019529,
                5.07798052,
                5.18583965,
                5.29376984,
                5.40176773,
                5.50982904,
                5.61795235,
                5.72613382,
                5.83437061,
                5.94266081,
                6.0510025,
                6.15939236,
                6.26782942,
                6.37631083,
                6.48483562,
                6.59340143,
                6.70200729,
                6.81065083,
                6.91933107,
                7.02804708,
                7.13679647,
                7.24557924,
                7.35439348,
                7.46323729,
                7.57211161,
                7.68101406,
                7.78994322,
                7.89889955,
                8.00788212,
                8.11688805,
                8.22591877,
                8.33497238,
                8.44404888,
                8.55314732,
                8.66226578,
                8.77140617,
                8.88056564,
                8.98974419,
                9.09894276,
                9.20815849,
                9.3173914,
                9.42664242,
                9.53590965,
                9.6451931,
                9.75449276,
                9.86380768,
                9.9731369,
                10.08248138,
                10.19184017,
                10.30121231,
                10.41059875,
                10.51999664,
                10.62940788,
                10.73883247,
                10.84826851,
                10.95771694,
                11.06717682,
                11.17664719,
                11.286129,
            ],
            dtype=np.float64,
        ),
    )


@fixture(scope='session')
def cu_111_fixed_l_max_state_after_init(cu_111_fixed_l_max_info):
    state_after_init = run_viperleed_initialization(
        cu_111_fixed_l_max_info.input_path
    )
    slab, rpars = state_after_init.slab, state_after_init.rpars
    return slab, rpars


@fixture(scope='session')
def cu_111_fixed_l_max_raw_phaseshifts(
    cu_111_fixed_l_max_info, cu_111_fixed_l_max_state_after_init
):
    slab, rpars = cu_111_fixed_l_max_state_after_init
    phaseshifts_path = cu_111_fixed_l_max_info.input_path / 'PHASESHIFTS'
    _, raw_phaseshifts, _, _ = readPHASESHIFTS(
        slab, rpars, readfile=phaseshifts_path, check=True, ignoreEnRange=False
    )
    return raw_phaseshifts


@fixture(scope='session')
def cu_111_fixed_l_max_phaseshifts(
    cu_111_fixed_l_max_info,
    cu_111_fixed_l_max_state_after_init,
    cu_111_fixed_l_max_raw_phaseshifts,
):
    slab, rpars = cu_111_fixed_l_max_state_after_init
    phaseshift_map = ps.phaseshift_site_el_order(slab, rpars)
    return ps.Phaseshifts(
        cu_111_fixed_l_max_raw_phaseshifts,
        cu_111_fixed_l_max_info.energies,
        l_max=cu_111_fixed_l_max_info.max_l_max,
        phaseshift_map=phaseshift_map,
    )


@fixture(scope='session')
def cu_111_fixed_l_max_read_tensor_zip(cu_111_fixed_l_max_info):
    return read_tensor_zip(
        cu_111_fixed_l_max_info.tensor_path,
        lmax=cu_111_fixed_l_max_info.max_l_max,
        n_beams=cu_111_fixed_l_max_info.n_beams,
        n_energies=cu_111_fixed_l_max_info.n_energies,
    )


@fixture(scope='session')
def cu_111_fixed_l_max_read_ref_data(
    cu_111_fixed_l_max_read_tensor_zip, cu_111_fixed_l_max_state_after_init
):
    slab, _ = cu_111_fixed_l_max_state_after_init
    non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
    sorted_tensors = [
        cu_111_fixed_l_max_read_tensor_zip[f'T_{at.num}']
        for at in non_bulk_atoms
    ]

    return process_tensors(sorted_tensors, fix_lmax=10)  # TODO: fix!!


@fixture(scope='session')
def cu_111_fixed_l_max_tensor_calculator(
    cu_111_fixed_l_max_state_after_init,
    cu_111_fixed_l_max_phaseshifts,
    cu_111_fixed_l_max_read_ref_data,
):
    slab, rpars = cu_111_fixed_l_max_state_after_init
    ref_calc_params, ref_calc_result = cu_111_fixed_l_max_read_ref_data
    return TensorLEEDCalculator(
        ref_calc_params,
        ref_calc_result,
        cu_111_fixed_l_max_phaseshifts,
        slab,
        rpars,
        recalculate_ref_t_matrices=False,
    )


@fixture(scope='session')
def cu_111_fixed_l_max_tensor_calculator_recalc_t_matrices(
    cu_111_fixed_l_max_state_after_init,
    cu_111_fixed_l_max_phaseshifts,
    cu_111_fixed_l_max_read_ref_data,
):
    slab, rpars = cu_111_fixed_l_max_state_after_init
    ref_calc_params, ref_calc_result = cu_111_fixed_l_max_read_ref_data
    return TensorLEEDCalculator(
        ref_calc_params,
        ref_calc_result,
        cu_111_fixed_l_max_phaseshifts,
        slab,
        rpars,
        recalculate_ref_t_matrices=True,
    )


@fixture(scope='session')
def cu_111_fixed_l_max_parameter_space(
    cu_111_fixed_l_max_state_after_init, cu_111_fixed_l_max_info
):
    slab, rparams = cu_111_fixed_l_max_state_after_init
    atom_basis = AtomBasis(slab)
    parameter_space = ParameterSpace(atom_basis, rparams)

    # displacements file
    disp_file = DisplacementsFile()
    disp_file.read(cu_111_fixed_l_max_info.displacements_path)

    if disp_file.offsets is not None:
        parameter_space.apply_offsets(disp_file.offsets)
    search_block = disp_file.next(0.9)
    parameter_space.apply_search_segment(search_block)
    return parameter_space


@fixture(scope='session')
def cu_111_fixed_l_max_calculator_with_parameter_space(
    cu_111_fixed_l_max_tensor_calculator, cu_111_fixed_l_max_parameter_space
):
    calculator = cu_111_fixed_l_max_tensor_calculator
    calculator.set_parameter_space(cu_111_fixed_l_max_parameter_space)
    return calculator


@fixture(scope='session')
def cu_111_fixed_l_max_calculator_with_parameter_space_recalc_t_matrices(
    cu_111_fixed_l_max_tensor_calculator_recalc_t_matrices,
    cu_111_fixed_l_max_parameter_space,
):
    calculator = cu_111_fixed_l_max_tensor_calculator_recalc_t_matrices
    calculator.set_parameter_space(cu_111_fixed_l_max_parameter_space)
    return calculator


@fixture(scope='session')
@pytest.mark.parametrize(
    'parameters, expected',
    _comparison_data,
    ids=_comparison_data.ids,
)
def cu_111_fixed_l_max_tenserleed_reference(parameters, expected):
    return parameters, expected
