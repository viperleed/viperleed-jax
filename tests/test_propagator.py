from pathlib import Path
import pytest
import numpy as np
import jax

import pytest

from viperleed_jax.propagator import calc_propagator, symmetry_tensor
from viperleed_jax.constants import BOHR
from viperleed_jax.lib_math import EPS


@pytest.fixture(scope="session")
def stored_propagator_reference_values():
    file = (
        Path(__file__).parent
        / "test_data"
        / "reference_values"
        / "propagator_reference_values.npz"
    )
    return (
        np.load(file)["reference_values_l_max_18_e_1e0_v_imag_1e0"],
        18,
        1.0,
        1.0,
    )


@pytest.fixture(scope="session")
def stored_propagator_energy_jacobians():
    file = (
        Path(__file__).parent
        / "test_data"
        / "reference_values"
        / "propagator_reference_values.npz"
    )
    return (
        np.load(file)["reference_energy_jac_values_l_max_8_e_1e0j_v_imag_1e0"],
        8,
        1.0 + 0.0j,
        1.0,
    )


@pytest.fixture(scope="session")
def stored_propagator_disp_jacobians():
    file = (
        Path(__file__).parent
        / "test_data"
        / "reference_values"
        / "propagator_reference_values.npz"
    )
    return (
        np.load(file)[
            "reference_displacement_jac_values_l_max_8_e_1e0_v_imag_1e0"
        ],
        8,
        1.0 + 0.0j,
        1.0,
    )


JIT_CALC_PROPAGATOR = jax.jit(calc_propagator, static_argnums=(0,))
JIT_JAC_ENERGY_JIT_CALC_PROPAGATOR = jax.jit(
    jax.jacrev(calc_propagator, argnums=2, holomorphic=True),
    static_argnums=(0,),
)
_abs_calc_propagator = lambda l_max, vec, e, v_imag: abs(
    calc_propagator(l_max, vec, e, v_imag)
)
JIT_JAC_ABS_DISP_CALC_PROPAGATOR = jax.jit(
    jax.jacrev(_abs_calc_propagator, argnums=1), static_argnums=(0,)
)

TEST_DISP_VECTORS = (
    np.array([0.1, 0.0, 0.0]),
    np.array([0.0, 0.1, 0.0]),
    np.array([0.0, 0.0, 0.1]),
    np.array([0.1, 0.1, 0.0]),
    np.array([0.1, 0.0, 0.1]),
    np.array([0.1, 0.1, 0.1]),
    np.array([-0.1, 0.0, 0.0]),
    np.array([-0.1, -0.1, -0.0]),
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
        assert propagator == pytest.approx(
            np.identity((l_max + 1) ** 2), abs=1e-8
        )

    @pytest.mark.parametrize("disp_vector", list(enumerate(TEST_DISP_VECTORS)))
    def test_with_displacement(
        self, disp_vector, stored_propagator_reference_values
    ):
        """Check that the propagator gives the expected result."""
        n_vector, disp_vector = disp_vector
        (
            stored_propagators,
            l_max,
            energy,
            v_imag,
        ) = stored_propagator_reference_values
        reference_value = stored_propagators[n_vector]
        # calculate the propagator
        propagator = calc_propagator(l_max, disp_vector, energy, v_imag)
        assert propagator == pytest.approx(reference_value, rel=1e-6, abs=1e-8)

    @pytest.mark.parametrize("disp_vector", list(enumerate(TEST_DISP_VECTORS)))
    def test_jit(self, disp_vector, stored_propagator_reference_values):
        """Check if the jit compiled function gives the same result."""
        n_vector, disp_vector = disp_vector
        (
            stored_propagators,
            l_max,
            energy,
            v_imag,
        ) = stored_propagator_reference_values
        reference_value = stored_propagators[n_vector]

        # calculate the propagator
        propagator = JIT_CALC_PROPAGATOR(l_max, disp_vector, energy, v_imag)
        assert propagator == pytest.approx(reference_value, rel=1e-6, abs=1e-8)

    @pytest.mark.parametrize("disp_vector", list(enumerate(TEST_DISP_VECTORS)))
    def test_energy_jacobian(
        self, disp_vector, stored_propagator_energy_jacobians
    ):
        """Check if the jit compiled function gives the same result."""
        n_vector, disp_vector = disp_vector
        (
            stored_propagators,
            l_max,
            energy,
            v_imag,
        ) = stored_propagator_energy_jacobians
        reference_value = stored_propagators[n_vector]

        # calculate the propagator
        propagator_jac = JIT_JAC_ENERGY_JIT_CALC_PROPAGATOR(
            l_max, disp_vector, energy, v_imag
        )
        assert propagator_jac == pytest.approx(
            reference_value, rel=1e-6, abs=1e-8
        )

    @pytest.mark.parametrize("disp_vector", list(enumerate(TEST_DISP_VECTORS)))
    def test_displacement_jacobian(
        self, disp_vector, stored_propagator_disp_jacobians
    ):
        """Check if the jit compiled function gives the same result."""
        n_vector, disp_vector = disp_vector
        (
            stored_propagators,
            l_max,
            energy,
            v_imag,
        ) = stored_propagator_disp_jacobians
        reference_value = stored_propagators[n_vector]

        # calculate the propagator
        propagator_jac = JIT_JAC_ABS_DISP_CALC_PROPAGATOR(
            l_max, disp_vector, energy, v_imag
        )
        assert propagator_jac == pytest.approx(
            reference_value, rel=5e-5, abs=1e-7
        )

def rot_matrix(theta):
    """Return a 2D rotation matrix."""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

TEST_PLANE_SYMMETRY_OPERATIONS = {
    # identity
    'identity': np.identity(2),
    # rotations
    'pi/6': rot_matrix(np.pi/6),
    'pi/4': rot_matrix(np.pi/4),
    'pi/2': rot_matrix(np.pi/2),
    'pi': rot_matrix(np.pi),
    '0.123': rot_matrix(0.123),
    # mirror operations
}

class TestSymmetryTensor:
    L_MAX = 8
    ENERGY = 1.0
    V_IMAG = 1.0

    def test_symmetry_tensor_identity(self):
        """Check that the symmetry tensor for the identity is ones."""
        assert symmetry_tensor(self.L_MAX, np.identity(2)) == pytest.approx(
            np.ones(((self.L_MAX + 1) ** 2, (self.L_MAX + 1) ** 2)), abs=1e-8
        )

    @pytest.mark.parametrize("plane_symmetry_operation",
                             TEST_PLANE_SYMMETRY_OPERATIONS.values(),
                             ids=TEST_PLANE_SYMMETRY_OPERATIONS.keys())
    @pytest.mark.parametrize("disp_vector", TEST_DISP_VECTORS)
    def test_symmetry_tensor(self, plane_symmetry_operation, disp_vector):

        # add z to get full 3d symmetry operation
        # NB: coordinates are stored as (z, x, y)!
        rotation_matrix_3d = np.identity(3)
        rotation_matrix_3d[1:3, 1:3] = plane_symmetry_operation

        # apply the symmetry operation to the displacement vector
        disp_vector_sym = rotation_matrix_3d @ disp_vector

        # calculate the propagator for both
        propagator_original = calc_propagator(self.L_MAX, disp_vector,
                                              self.ENERGY, self.V_IMAG)
        propagator_sym = calc_propagator(self.L_MAX, disp_vector_sym,
                                         self.ENERGY, self.V_IMAG)

        # apply the symmetry operation to the propagator
        # (element-wise multiplication)
        propagator_original_sym = propagator_original * symmetry_tensor(self.L_MAX,
                                                                       plane_symmetry_operation)

        # check if the propagator is the same
        assert propagator_sym == pytest.approx(propagator_original_sym,
                                               abs=1e-7)
