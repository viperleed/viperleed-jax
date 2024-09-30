"""Tests for the tensor_calculator module."""
from collections import namedtuple

from pathlib import Path
import pytest
import numpy as np
import jax

import pytest
from pytest_cases import parametrize_with_cases, case

from viperleed_jax.constants import BOHR

DeltaIndices = namedtuple('DeltaIndices', ['geo', 'vib',])

CU_111_PARAM_FILE_CORRESPONDENCE = {
    (.5, .5, 1.0, .5, .5) : (DeltaIndices(0, 0), 3.7e-5),
}

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
        abs = 3.2e-5
        return cu_111_dynamic_l_max_calculator_with_parameter_space, center, abs

class TensorCalculatorsWithTensErLEEDDeltas:
    """Tensor calculators with parameter space already applied and comparison
    values. Values tagged as 'TensorLEED' are taken directly from a calculation
    with TensErLEED."""
    @case(tags="cu_111")
    @pytest.mark.parametrize(
        "params, compare_values",
        list(zip([np.array(p) for p in CU_111_PARAM_FILE_CORRESPONDENCE.keys()],
                 CU_111_PARAM_FILE_CORRESPONDENCE.values())),
        ids = [str(np.array(p)) for p in CU_111_PARAM_FILE_CORRESPONDENCE.keys()]
    )
    def case_cu_111_dynamic_l(
        self,
        cu_111_dynamic_l_max_calculator_with_parameter_space,
        cu_111_dynamic_l_max_parameter_delta_file,
        params,
        compare_values):
        calculator = cu_111_dynamic_l_max_calculator_with_parameter_space
        file = cu_111_dynamic_l_max_parameter_delta_file
        file_indices, abs = compare_values
        params = np.array(params)
        return calculator, params, file, file_indices, abs


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

@parametrize_with_cases("calculator, params, file, file_indices, abs", cases=TensorCalculatorsWithTensErLEEDDeltas)
def test_compare_known_delta_amplitudes_tenserleed(calculator, params, file,
                                                   file_indices, abs):
    _, vib_params, geo_params, _ = calculator.parameter_space.split_free_params(params)
    vibs = calculator.parameter_space.vib_transformer(vib_params)
    geos = calculator.parameter_space.geo_transformer(geo_params)

    ref_vibs = file['vib_delta'][file_indices.vib]
    ref_geos = file['geo_delta'][0, file_indices.geo, :]

    # check that input vibs and geos are the same
    assert vibs/BOHR == pytest.approx(ref_vibs, abs=1e-4)
    assert geos == pytest.approx(ref_geos.reshape(geos.shape))

    ref_delta = file['amplitudes_del']
    ref_value = ref_delta[0, :, file_indices.vib, file_indices.geo, :]

    # check that values match
    delta_amps = calculator.delta_amplitude(params)
    assert delta_amps == pytest.approx(ref_value, abs=abs)
