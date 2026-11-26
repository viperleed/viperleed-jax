"""Tests for the selection and use of R-factor functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from viperleed.calc.classes.rparams.special.r_factor_type import RFactorType
from viperleed.calc.lib.rfactor import selection
from viperleed.calc.lib.rfactor.pendry import r_pendry

from viperleed_jax.tensor_calculator import (
    TensorLEEDCalculator,
    calc_r_factor,
)

_NAMES = [*(member.name for member in RFactorType), *selection._R1_NAMES]


@pytest.fixture(name='calculator')
def fixture_calculator():
    """Return a calculator without any setup."""
    return TensorLEEDCalculator.__new__(TensorLEEDCalculator)


@pytest.mark.parametrize('name', _NAMES)
def test_set_rfactor(calculator, name):
    """Check that set_rfactor picks the function selected by name."""
    calculator.set_rfactor(name)
    assert calculator.rfactor_func is selection.select_rfactor_by_name(name)


@pytest.mark.parametrize('name', _NAMES)
def test_set_rfactor_function(calculator, name):
    """Check that the name picks the expected R-factor function."""
    expected = {
        'PENDRY': 'r_pendry',
        'R2': 'r_2',
        'ZJ': 'r_zjj',
        'SMOOTH': 'r_s',
    }.get(name, 'r_1')
    calculator.set_rfactor(name)
    assert calculator.rfactor_func.__name__ == expected


def test_default_rfactor_is_pendry(
    fe2o3_012_converged_calculator_with_parameter_space_z,
):
    """Check that the default R-factor of a calculator is Pendry's."""
    calculator = fe2o3_012_converged_calculator_with_parameter_space_z
    assert calculator.rfactor_func is r_pendry


def test_set_rfactor_invalid(calculator):
    """Check that an unknown name is rejected."""
    with pytest.raises(ValueError):
        calculator.set_rfactor('not-an-r-factor')


def _synthetic_beams(energies, shift):
    phases = np.array([0.0, 1.0, 2.0])
    return 1.5 + np.sin(0.3 * (energies[:, None] + shift) + phases)


@pytest.mark.parametrize('name', _NAMES)
@pytest.mark.parametrize('groups', [None, (0, 0, 1)])
def test_calc_r_factor_finite(name, groups):
    """Check that each selectable R factor works under jit."""
    energies = np.arange(50.0, 150.0, 2.0)
    target_grid = jnp.arange(55.0, 145.0, 0.5)
    exp_spline = _exp_spline(energies)
    exp_data = tuple(
        jnp.asarray(d(np.asarray(target_grid)))
        for d in (exp_spline, exp_spline.derivative(),
                  exp_spline.derivative().derivative())
    )
    theo = jnp.asarray(_synthetic_beams(energies, 0.5))
    rfactor_func = selection.select_rfactor_by_name(name)

    def r_factor(shift):
        return calc_r_factor(
            theo,
            shift,
            -4.5,
            rfactor_func,
            jnp.asarray(energies),
            0.5,
            target_grid,
            exp_data,
            groups=groups,
            num_groups=None if groups is None else 2,
        )

    value = jnp.atleast_1d(r_factor(0.0))
    assert value.shape == ((1,) if groups is None else (2,))
    assert np.all(np.isfinite(value))
    grad = jax.grad(lambda s: jnp.sum(r_factor(s)))(0.0)
    assert np.isfinite(grad)


def _exp_spline(energies):
    from viperleed.calc.lib import spline_interpolation

    return spline_interpolation.interpolate_ragged_array(
        energies, _synthetic_beams(energies, 0.0)
    )
