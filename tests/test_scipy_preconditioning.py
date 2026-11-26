"""Tests for the preconditioning of the SciPy-based optimizers."""

__authors__ = ('Florian Kraushofer (@fkraushofer)',)
__created__ = '2026-11-26'

import numpy as np
import pytest

from viperleed_jax.optimization.optimizer import (
    GradFreeSLSQPOptimizer,
    SLSQPOptimizer,
)

# Lower-triangular CMA-ES Cholesky factor with large off-diagonal
# terms, as after the Ag(100) smoke run. With x0[0] at its upper
# bound, transforming the [0, 1] box with the full factor gives
# y-bounds for component 0 that exclude y = 0.
_CHOLESKY = np.array([[0.15, 0.0, 0.0],
                      [-0.07, 0.19, 0.0],
                      [-0.06, 0.13, 0.09]])
_X0 = np.array([1.0, 0.5, 0.5])
_TARGET = np.array([0.9, 0.6, 0.4])


def _make_optimizer(cls, evaluated):
    """Return an optimizer of class cls recording evaluated points."""
    def fun(x):
        evaluated.append(np.array(x))
        return float(np.sum((x - _TARGET)**2))

    def grad(x):
        return 2 * (x - _TARGET)

    if cls is SLSQPOptimizer:
        return cls(fun=fun, grad=grad, cholesky=_CHOLESKY,
                   grad_damp_factor=1.0)
    return cls(fun=fun, cholesky=_CHOLESKY)


@pytest.mark.parametrize('cls', (SLSQPOptimizer, GradFreeSLSQPOptimizer))
def test_start_at_bound(cls):
    """Check that SLSQP starts at x0 and stays inside [0, 1]."""
    evaluated = []
    optimizer = _make_optimizer(cls, evaluated)
    result = optimizer(_X0.copy())
    np.testing.assert_array_equal(evaluated[0], _X0)
    evaluated = np.array(evaluated)
    assert np.all(evaluated >= 0.0)
    assert np.all(evaluated <= 1.0)
    assert result.best_R <= np.sum((_X0 - _TARGET)**2)


def test_bounds_contain_start():
    """Check that the transformed bounds contain y = 0."""
    optimizer = _make_optimizer(SLSQPOptimizer, [])
    optimizer._set_cholesky_related(_X0)
    lower, upper = np.array(optimizer.transform_bounds(_X0)).T
    assert np.all(lower <= 0.0)
    assert np.all(upper >= 0.0)
    # The bounds of y map back exactly to the bounds of x
    np.testing.assert_allclose(optimizer.transform_x(lower, _X0), 0.0,
                               atol=1e-15)
    np.testing.assert_allclose(optimizer.transform_x(upper, _X0), 1.0)
