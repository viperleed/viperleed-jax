"""Tests for the tensor_calculator module."""
from collections import namedtuple

from pathlib import Path
import pytest
import numpy as np
import jax

import pytest
from pytest_cases import parametrize_with_cases, case

from viperleed_jax.constants import BOHR

class TensorCalculatorsWithInfo:
    """Tensor calculators with basic information for cross-checking."""
    @case(tags="cu_111")
    def case_cu_111_dynamic_l(self,
        cu_111_dynamic_l_max_tensor_calculator,
        cu_111_dynamic_l_max_info):
        calculator = cu_111_dynamic_l_max_tensor_calculator
        return calculator, cu_111_dynamic_l_max_info

class TensorCalculatorsWithSpace:
    """Tensor calculators with parameter space already applied and info on the
    center parameter set and the max abs delta-amplitude value for the
    undisplaced state."""
    @case(tags="cu_111")
    def case_cu_111_dynamic_l(self,
        cu_111_dynamic_l_max_calculator_with_parameter_space):
        center = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        abs = 2.2e-4
        return cu_111_dynamic_l_max_calculator_with_parameter_space, center, abs

    @case(tags="cu_111")
    def case_cu_111_dynamic_l_recalc_t_matrices(self,
        cu_111_dynamic_l_max_calculator_with_parameter_space_recalc_t_matrices):
        center = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        abs = 1e-9
        return cu_111_dynamic_l_max_calculator_with_parameter_space_recalc_t_matrices, center, abs

    @case(tags="cu_111")
    def case_cu_111_fixed_l(self,
        cu_111_dynamic_l_max_calculator_with_parameter_space):
        center = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        abs = 2.2e-4
        return cu_111_dynamic_l_max_calculator_with_parameter_space, center, abs

    @case(tags="cu_111")
    def case_cu_111_fixed_l_recalc_t_matrices(self,
        cu_111_fixed_l_max_calculator_with_parameter_space_recalc_t_matrices):
        center = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        abs = 1e-9
        return cu_111_fixed_l_max_calculator_with_parameter_space_recalc_t_matrices, center, abs

class TensorCalculatorsWithTensErLEEDDeltas:
    """Tensor calculators with parameter space already applied and comparison
    values. Values tagged as 'TensorLEED' are taken directly from a calculation
    with TensErLEED."""
    @case(tags="cu_111")
    def case_cu_111_dynamic_l(
        self,
        cu_111_dynamic_l_max_calculator_with_parameter_space,
        cu_111_dynamic_l_max_tenserleed_reference):
        calculator = cu_111_dynamic_l_max_calculator_with_parameter_space
        params, reference_delta_amplitudes, abs = cu_111_dynamic_l_max_tenserleed_reference
        return calculator, params, reference_delta_amplitudes, abs

    @case(tags="cu_111")
    def case_cu_111_fixed_l(
        self,
        cu_111_fixed_l_max_calculator_with_parameter_space,
        cu_111_fixed_l_max_tenserleed_reference):
        calculator = cu_111_fixed_l_max_calculator_with_parameter_space
        params, reference_delta_amplitudes, abs = cu_111_fixed_l_max_tenserleed_reference
        return calculator, params, reference_delta_amplitudes, abs

    @case(tags="fe2o3_012")
    def case_fe2o3_012_converged_z(
        self,
        fe2o3_012_converged_calculator_with_parameter_space,
        fe2o3_012_converged_tenserleed_reference_z):
        calculator = fe2o3_012_converged_calculator_with_parameter_space
        params, reference_delta_amplitudes, abs = fe2o3_012_converged_tenserleed_reference_z
        return calculator, params, reference_delta_amplitudes, abs

@parametrize_with_cases("calculator, info", cases=TensorCalculatorsWithInfo)
def test_calculator_creation(calculator, info):
    assert calculator is not None
    assert calculator.n_beams == info.n_beams
    assert len(calculator.energies) == info.n_energies

@parametrize_with_cases("calculator, center, abs", cases=TensorCalculatorsWithSpace)
def test_unperturbed_delta_amplitudes(calculator, center, abs):
    """Check that the delta amplitudes for the unperturbed state are zero."""
    delta_amps = calculator.delta_amplitude(center)
    assert delta_amps == pytest.approx(0.0, abs=abs)

@pytest.mark.parametrize("delta",
                         [0.0, 1e-5, 1e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                          0.8, 0.9, 0.9999, 0.99999, 1.0])
@parametrize_with_cases("calculator, center, abs", cases=TensorCalculatorsWithSpace)
def test_perturbed_delta_amplitudes_finite(calculator, center, abs, delta):
    """Check that the delta amplitudes for a perturbed state are finite."""
    delta_amps = calculator.delta_amplitude([delta,] * calculator.n_free_parameters)
    assert np.isfinite(delta_amps).all()

@parametrize_with_cases("calculator, params, reference_delta_amplitudes, abs",
                        cases=TensorCalculatorsWithTensErLEEDDeltas)
def test_compare_known_delta_amplitudes_tenserleed(calculator,
                                                   params,
                                                   reference_delta_amplitudes,
                                                   abs):
    # check that values match
    delta_amps = calculator.jit_delta_amplitude(params)
    assert delta_amps == pytest.approx(reference_delta_amplitudes, abs=abs)
