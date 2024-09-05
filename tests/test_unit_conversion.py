import pytest
import jax.numpy as jnp

from viperleed_jax.unit_conversion import kappa

@pytest.mark.parametrize("energy, v_imag, expected", [
    (1.0, 0.5, jnp.sqrt(2.0 + 1.0j)),
    (0.5, 0.5, jnp.sqrt(1.0 + 1.0j)),
    (0.0, 1.0, jnp.sqrt(2.0j)),
    (1.0, 0.0, jnp.sqrt(2.0)),
])
def test_kappa(energy, v_imag, expected):
    assert kappa(energy, v_imag) == pytest.approx(expected)
