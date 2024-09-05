import pytest
import jax.numpy as jnp

from viperleed_jax.atomic_units import kappa
from viperleed_jax import atomic_units


@pytest.mark.parametrize("energy_eV, expected_au", [
    (27.21138602, 1.0),  # 1 Hartree in eV is exactly 1 atomic unit of energy
    (54.42277204, 2.0),  # 2 Hartrees in eV
    (0.0, 0.0),          # 0 eV should be 0 atomic units
    (13.60569301, 0.5),  # 0.5 Hartree in eV
])
def test_to_atomic_unit_energy(energy_eV, expected_au):
    assert atomic_units.to_atomic_unit_energy(energy_eV) == pytest.approx(expected_au)

@pytest.mark.parametrize("energy, v_imag, expected", [
    (1.0, 0.5, jnp.sqrt(2.0 + 1.0j)),
    (0.5, 0.5, jnp.sqrt(1.0 + 1.0j)),
    (0.0, 1.0, jnp.sqrt(2.0j)),
    (1.0, 0.0, jnp.sqrt(2.0)),
])
def test_kappa(energy, v_imag, expected):
    assert atomic_units.kappa(energy, v_imag) == pytest.approx(expected)
