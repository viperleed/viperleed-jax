import jax.numpy as jnp
import pytest

from viperleed_jax import atomic_units
from viperleed_jax.constants import BOHR, HARTREE


@pytest.mark.parametrize(
    'displacement_vector_ang, expected_vector',
    [
        ([1.0, 2.0, 3.0], jnp.array([1 / BOHR, 2 / BOHR, -3 / BOHR])),
        ([0.0, 0.0, 0.0], jnp.array([0.0, 0.0, 0.0])),
        ([-1.0, -2.0, -3.0], jnp.array([-1 / BOHR, -2 / BOHR, 3 / BOHR])),
    ],
)
def test_to_internal_displacement_vector(
    displacement_vector_ang, expected_vector
):
    result = atomic_units.to_internal_displacement_vector(
        displacement_vector_ang
    )
    assert result.reshape(-1) == pytest.approx(expected_vector)


@pytest.mark.parametrize(
    'energy_eV, expected_au',
    [
        (1 * HARTREE, 1.0),
        (2 * HARTREE, 2.0),
        (0.0, 0.0),  # 0 eV should be 0 Hartree
        (0.5 * HARTREE, 0.5),
    ],
)
def test_to_atomic_unit_energy(energy_eV, expected_au):
    assert atomic_units.to_atomic_unit_energy(energy_eV) == pytest.approx(
        expected_au
    )


@pytest.mark.parametrize(
    'energy, v_imag, expected',
    [
        (1.0, 0.5, jnp.sqrt(2.0 + 1.0j)),
        (0.5, 0.5, jnp.sqrt(1.0 + 1.0j)),
        (0.0, 1.0, jnp.sqrt(2.0j)),
        (1.0, 0.0, jnp.sqrt(2.0)),
    ],
)
def test_kappa(energy, v_imag, expected):
    assert atomic_units.kappa(energy, v_imag) == pytest.approx(expected)
