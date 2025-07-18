"""Module structures of tests."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-26'

from collections import namedtuple
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional

from pytest_cases import case


class Tag(IntEnum):
    """Enumeration of tags to use for cases."""

    PARAMETER_SPACE_SIZE_TOTAL = auto()
    PARAMETER_SPACE_SIZE_SYMMETRY = auto()
    Z_ONLY_ATOM = auto()
    IN_PLANE_1D_ATOMS = auto()
    FREE_ATOMS = auto()


ParameterSpaceSize = namedtuple(
    'parameter_space_size',
    [
        'n_v0r',
        'n_geo',
        'n_vib',
        'n_occ',
    ],
)


@dataclass
class ParameterSpaceInfo:
    """Information about parameter space."""

    total_size: Optional[ParameterSpaceSize] = None
    symmetry_size: Optional[ParameterSpaceSize] = None
    user_constrained_size: Optional[ParameterSpaceSize] = None
    free_size: Optional[ParameterSpaceSize] = None


CU_111_INFO = ParameterSpaceInfo(
    total_size=ParameterSpaceSize(1, 15, 5, 5),
    symmetry_size=ParameterSpaceSize(1, 5, 5, 5),
)

Fe2O3_012_INFO = ParameterSpaceInfo(
    total_size=ParameterSpaceSize(1, 90, 30, 30),
    symmetry_size=ParameterSpaceSize(1, 45, 15, 15),
)

Fe3O4_111_INFO = ParameterSpaceInfo(
    total_size=ParameterSpaceSize(1, 66, 22, 22),
    symmetry_size=ParameterSpaceSize(1, 17, 12, 12),
)

PT_111_10x10_TE_INFO = ParameterSpaceInfo(
    total_size=ParameterSpaceSize(1, 1245, 415, 415),
    symmetry_size=ParameterSpaceSize(1, 229, 92, 92),
)

PT25RH75_O_3x1_INFO = ParameterSpaceInfo()


class CaseStatesAfterInit:
    """Collection of cases with structures after viperleed.calc init."""

    @case(
        tags=[
            Tag.PARAMETER_SPACE_SIZE_TOTAL,
            Tag.PARAMETER_SPACE_SIZE_SYMMETRY,
            Tag.Z_ONLY_ATOM,
        ]
    )
    def case_cu_111_dynamic_l_max(self, state_cu_111_dynamic_l_max):
        state = state_cu_111_dynamic_l_max
        return state, CU_111_INFO

    @case(
        tags=[
            Tag.PARAMETER_SPACE_SIZE_TOTAL,
            Tag.PARAMETER_SPACE_SIZE_SYMMETRY,
            Tag.Z_ONLY_ATOM,
        ]
    )
    def case_cu_111_fixed_l_max(self, state_cu_111_fixed_l_max):
        state = state_cu_111_fixed_l_max
        return state, CU_111_INFO

    @case(
        tags=[
            Tag.PARAMETER_SPACE_SIZE_TOTAL,
            Tag.PARAMETER_SPACE_SIZE_SYMMETRY,
            Tag.FREE_ATOMS,
        ]
    )
    def case_fe2o3_012_converged(self, state_fe2o3_012_converged):
        state = state_fe2o3_012_converged
        return state, Fe2O3_012_INFO

    @case(
        tags=[
            Tag.PARAMETER_SPACE_SIZE_TOTAL,
            Tag.PARAMETER_SPACE_SIZE_SYMMETRY,
            Tag.IN_PLANE_1D_ATOMS,
        ]
    )
    def case_fe3o4_111(self, state_fe3o4_111):
        state = state_fe3o4_111
        return state, Fe3O4_111_INFO

    @case(tags=[Tag.PARAMETER_SPACE_SIZE_TOTAL, Tag.IN_PLANE_1D_ATOMS])
    def case_pt_111_10x10_te(self, state_pt_111_10x10_te):
        state = state_pt_111_10x10_te
        return state, PT_111_10x10_TE_INFO

    @case
    def case_pt25rh75_o_3x1(self, init_state_pt25rh75_o_3x1):
        state = init_state_pt25rh75_o_3x1
        return state, PT25RH75_O_3x1_INFO
