from pathlib import Path

import numpy as np
import pytest
from pytest_cases import fixture
from viperleed.calc.files.new_displacements.file import DisplacementsFile
from viperleed.calc.files.phaseshifts import readPHASESHIFTS

from tests.fixtures.base import LARGE_FILE_PATH
from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.files import phaseshifts as ps
from viperleed_jax.files.tensors import read_tensor_zip
from viperleed_jax.from_state import run_viperleed_initialization
from viperleed_jax.parameter_space import ParameterSpace
from viperleed_jax.ref_calc_data import process_tensors


_DATA_PATH = (
    Path(__file__).parent.parent / 'test_data' / 'Pt25Rh75_O_3x1'
)


@fixture(scope='session')
def pt25rh75_o_3x1_state_after_init():
    state_after_init = run_viperleed_initialization(_DATA_PATH)
    slab, rpars = state_after_init.slab, state_after_init.rpars
    return slab, rpars

@fixture(scope='session')
def pt25rh75_o_3x1_parameter_space(pt25rh75_o_3x1_state_after_init):

    slab, rparams = pt25rh75_o_3x1_state_after_init
    atom_basis = AtomBasis(slab)
    parameter_space = ParameterSpace(atom_basis, rparams)

    # displacements file
    disp_file = DisplacementsFile()
    disp_file.read(_DATA_PATH / 'DISPLACEMENTS')

    if disp_file.offsets is not None:
        parameter_space.apply_offsets(disp_file.offsets)
    search_block = disp_file.next(0.9)
    parameter_space.apply_search_segment(search_block)
    return parameter_space
