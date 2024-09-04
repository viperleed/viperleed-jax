from pathlib import Path
import pytest
import numpy as np
import jax

import pytest

from viperleed_jax.propagator import calc_propagator
from viperleed_jax.constants import BOHR
from viperleed_jax.lib_math import EPS


@pytest.fixture(scope='session')
def stored_propagator_reference_values():
    file = Path(__file__).parent / 'test_data' / 'reference_values' / 'propagator_reference_values.npz'
    return np.load(file)['propagator_reference_values_l_max_18_e_1e0_v_imag_1e0'], 18, 1.0, 1.0


@pytest.fixture(scope='session')
def stored_propagator_energy_jacobians():
    file = Path(__file__).parent / 'test_data' / 'reference_values' / 'propagator_reference_values.npz'
    return np.load(file)['propagator_reference_energy_jac_values_l_max_8_e_1e0j_v_imag_1e0'], 8, 1.0+.0j, 1.0

@pytest.fixture(scope='session')
def stored_propagator_disp_jacobians():
    file = Path(__file__).parent / 'test_data' / 'reference_values' / 'propagator_reference_values.npz'
    return np.load(file)['propagator_reference_displacement_jac_values_l_max_8_e_1e0_v_imag_1e0'], 8, 1.0+.0j, 1.0



JIT_CALC_PROPAGATOR = jax.jit(calc_propagator, static_argnums=(0,))
JIT_JAC_ENERGY_JIT_CALC_PROPAGATOR = jax.jit(
    jax.jacrev(calc_propagator, argnums=2, holomorphic=True),
    static_argnums=(0,)
)
_abs_calc_propagator = lambda l_max, vec, e, v_imag: abs(calc_propagator(l_max, vec, e, v_imag))
JIT_JAC_ABS_DISP_CALC_PROPAGATOR = jax.jit(
    jax.jacrev(_abs_calc_propagator, argnums=1),
    static_argnums=(0,)
)

TEST_DISP_VECTORS = (
    np.array([0.1, 0.0, 0.0]),
    np.array([0.0, 0.1, 0.0]),
    np.array([0.0, 0.0, 0.1]),
    np.array([0.1, 0.1, 0.0]),
    np.array([0.1, 0.0, 0.1]),
    np.array([0.1, 0.1, 0.1]),
    np.array([-0.1, 0.0, 0.0]),
    np.array([-0.1, -0.1, -0.]),
    np.array([0.0, -0.1, 0.1]),
    np.array([1.0, 0.0, 0.0]),
    np.array([1.0, 2.0, 3.0]),
    np.array([1e-3, 1e-3, 1e-3]),
    np.array([1e-4, 1e-4, 1e-4]),
    np.array([1e-5, 1e-5, 1e-5]),
    np.array([1e-6, 1e-6, 1e-6]),
)

class TestPropagator:
    # Propagators for a vanishing displacement should always yield the
    # identity matrix
    @pytest.mark.parametrize("l_max", range(5, 18))
    def test_propagator_zero_displacement(self, l_max):
        disp_vector = np.array([0.0, 0.0, 0.0])
        propagator = calc_propagator(l_max, disp_vector, 1.0, 1.0)
        assert propagator == pytest.approx(np.identity((l_max+1)**2), abs=1e-8)


    @pytest.mark.parametrize("disp_vector", list(enumerate(TEST_DISP_VECTORS)))
    def test_with_displacement(self, disp_vector, stored_propagator_reference_values):
        """Check that the propagator gives the expected result."""
        n_vector, disp_vector = disp_vector
        stored_propagators, l_max, energy, v_imag = stored_propagator_reference_values
        reference_value = stored_propagators[n_vector]
        # calculate the propagator
        propagator = calc_propagator(l_max, disp_vector, energy, v_imag)
        assert propagator == pytest.approx(reference_value, rel=1e-6, abs=1e-8)

    @pytest.mark.parametrize("disp_vector", list(enumerate(TEST_DISP_VECTORS)))
    def test_jit(self, disp_vector, stored_propagator_reference_values):
        """Check if the jit compiled function gives the same result."""
        n_vector, disp_vector = disp_vector
        stored_propagators, l_max, energy, v_imag = stored_propagator_reference_values
        reference_value = stored_propagators[n_vector]

        # calculate the propagator
        propagator = JIT_CALC_PROPAGATOR(l_max, disp_vector, energy, v_imag)
        assert propagator == pytest.approx(reference_value, rel=1e-6, abs=1e-8)

    @pytest.mark.parametrize("disp_vector", list(enumerate(TEST_DISP_VECTORS)))
    def test_energy_jacobian(self, disp_vector, stored_propagator_energy_jacobians):
        """Check if the jit compiled function gives the same result."""
        n_vector, disp_vector = disp_vector
        stored_propagators, l_max, energy, v_imag = stored_propagator_energy_jacobians
        reference_value = stored_propagators[n_vector]

        # calculate the propagator
        propagator_jac = JIT_JAC_ENERGY_JIT_CALC_PROPAGATOR(l_max, disp_vector, energy, v_imag)
        assert propagator_jac == pytest.approx(reference_value, rel=1e-6, abs=1e-8)

    @pytest.mark.parametrize("disp_vector", list(enumerate(TEST_DISP_VECTORS)))
    def test_displacement_jacobian(self, disp_vector, stored_propagator_disp_jacobians):
        """Check if the jit compiled function gives the same result."""
        n_vector, disp_vector = disp_vector
        stored_propagators, l_max, energy, v_imag = stored_propagator_disp_jacobians
        reference_value = stored_propagators[n_vector]

        # calculate the propagator
        propagator_jac = JIT_JAC_ABS_DISP_CALC_PROPAGATOR(l_max, disp_vector, energy, v_imag)
        assert propagator_jac == pytest.approx(reference_value, rel=5e-5, abs=1e-7)
