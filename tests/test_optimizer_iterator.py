"""Tests for the OptimizerIterator of viperleed_jax.optimization."""

__authors__ = ('Florian Kraushofer (@fkraushofer)',)
__created__ = '2026-11-26'

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from viperleed_jax.optimization import optimizer as opt
from viperleed_jax.optimization.iterator import OptimizerIterator

_TARGET = np.array([0.3, 0.6])


def _fake_calculator():
    """Return a calculator stand-in with a quadratic R."""
    calculator = MagicMock()
    calculator.n_free_parameters = len(_TARGET)
    calculator.parameter_space.get_parameter_names.return_value = ['a', 'b']
    calculator.R = lambda x: float(np.sum((x - _TARGET)**2))
    calculator.grad_R = lambda x: 2 * (x - _TARGET)
    return calculator


@pytest.mark.parametrize('grad', (True, False), ids=('grad', 'grad-free'))
def test_bfgs(grad):
    """Check that the BFGS optimizers run, with and without gradients."""
    rpars = SimpleNamespace(
        VLJ_ALGO=['BFGS'],
        VLJ_CONFIG={'precondition': False},
        vlj_algo_settings={'BFGS': {'grad': grad}},
        )
    iterator = OptimizerIterator(rpars, _fake_calculator(),
                                 starting_x=[0.5, 0.5])
    (optimizer, result), = iterator
    expected = opt.BFGSOptimizer if grad else opt.GradFreeBFGSOptimizer
    assert type(optimizer) is expected
    np.testing.assert_allclose(result.best_x, _TARGET, atol=1e-3)
