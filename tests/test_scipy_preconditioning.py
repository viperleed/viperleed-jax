"""Tests for the preconditioning of the SciPy-based optimizers."""

__authors__ = ('Florian Kraushofer (@fkraushofer)',)
__created__ = '2026-11-26'

import logging

import numpy as np
import pytest

from viperleed_jax.optimization.optimizer import (
    GradFreeSLSQPOptimizer,
    LBFGSBOptimizer,
    SLSQPOptimizer,
)

# Lower-triangular CMA-ES Cholesky factor with large off-diagonal
# terms, as CMA-ES produces for correlated parameters. With x0[0] at
# its upper bound, transforming the [0, 1] box with the full factor
# gives y-bounds for component 0 that exclude y = 0.
_CHOLESKY = np.array([[0.15, 0.0, 0.0],
                      [-0.07, 0.19, 0.0],
                      [-0.06, 0.13, 0.09]])
_X0 = np.array([1.0, 0.5, 0.5])
_TARGET = np.array([0.9, 1.2, -0.1])   # outside the box in two components


def _make_optimizer(cls, evaluated):
    """Return an optimizer of class cls recording evaluated points."""
    def fun(x):
        evaluated.append(np.array(x))
        return float(np.sum((x - _TARGET)**2))

    def grad(x):
        evaluated.append(np.array(x))
        return 2 * (x - _TARGET)

    if cls is SLSQPOptimizer:
        return cls(fun=fun, grad=grad, cholesky=_CHOLESKY)
    return cls(fun=fun, cholesky=_CHOLESKY)


@pytest.mark.parametrize('cls', (SLSQPOptimizer, GradFreeSLSQPOptimizer))
def test_start_at_bound(cls):
    """Check that SLSQP starts at x0 and reaches the box minimum in [0, 1]."""
    evaluated = []
    optimizer = _make_optimizer(cls, evaluated)
    result = optimizer(_X0.copy())
    np.testing.assert_array_equal(evaluated[0], _X0)
    evaluated = np.array(evaluated)
    assert np.all(evaluated >= 0.0)
    assert np.all(evaluated <= 1.0)
    np.testing.assert_allclose(result.best_x, np.clip(_TARGET, 0, 1),
                               atol=1e-4)


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


# Non-diagonal factor with strongly non-uniform row norms (standard
# deviations 0.02, ~0.30 and ~0.22) whose column norms differ from them.
_CHOLESKY_NONUNIFORM = np.array([[0.02, 0.0, 0.0],
                                 [0.05, 0.3, 0.0],
                                 [0.02, 0.1, 0.2]])


def test_scale_is_std():
    """Check that a unit step in y moves x by 1/std of each parameter."""
    optimizer = SLSQPOptimizer(fun=np.sum, grad=np.ones_like,
                               cholesky=_CHOLESKY_NONUNIFORM)
    x0 = np.array([0.3, 0.6, 0.5])
    optimizer._set_cholesky_related(x0)
    std = np.sqrt(np.diag(_CHOLESKY_NONUNIFORM @ _CHOLESKY_NONUNIFORM.T))
    np.testing.assert_allclose(optimizer.transform_x(np.ones(3), x0) - x0,
                               1 / std)


def test_grad_chain_rule():
    """Check that transform_grad gives the gradient with respect to y."""
    optimizer = SLSQPOptimizer(fun=np.sum, grad=np.ones_like,
                               cholesky=_CHOLESKY_NONUNIFORM)
    x0 = np.array([0.3, 0.6, 0.5])
    optimizer._set_cholesky_related(x0)

    def fun_of_y(y):
        return np.sum((optimizer.transform_x(y, x0) - _TARGET)**2)

    y = np.array([1e-4, 0.02, -0.03])
    step = 1e-7
    numerical = [(fun_of_y(y + step*e) - fun_of_y(y - step*e)) / (2*step)
                 for e in np.eye(3)]
    analytical = optimizer.transform_grad(
        2 * (optimizer.transform_x(y, x0) - _TARGET)
        )
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5)


def _cholesky_one_small(std_0):
    """Return a Cholesky factor whose first row has norm std_0."""
    return np.array([[std_0, 0.0, 0.0],
                     [0.05, 0.3, 0.0],
                     [0.02, 0.1, 0.2]])


@pytest.mark.parametrize('cholesky', (
    _cholesky_one_small(1e-3),
    _cholesky_one_small(1e-4),
    _cholesky_one_small(1e-6),
    1e-4 * _cholesky_one_small(1.0),
    ), ids=('one 1e-3', 'one 1e-4', 'one 1e-6', 'all 1e-4'))
@pytest.mark.parametrize('cls', (SLSQPOptimizer, GradFreeSLSQPOptimizer))
def test_box_minimum_small_std(cls, cholesky):
    """Check that a small CMA-ES standard deviation does not stall SLSQP."""
    x0 = np.array([1.0, 0.0, 0.5])            # on an upper and a lower bound
    target = np.array([1.3, -0.2, 0.45])      # outside the box in two
    evaluated = []

    def fun(x):
        evaluated.append(np.array(x))
        return float(np.sum((x - target)**2))

    def grad(x):
        evaluated.append(np.array(x))
        return 2 * (x - target)

    if cls is SLSQPOptimizer:
        optimizer = cls(fun=fun, grad=grad, cholesky=cholesky)
    else:
        optimizer = cls(fun=fun, cholesky=cholesky)
    result = optimizer(x0.copy())
    np.testing.assert_array_equal(evaluated[0], x0)
    evaluated = np.array(evaluated)
    assert np.all(evaluated >= 0.0)
    assert np.all(evaluated <= 1.0)
    np.testing.assert_allclose(result.best_x, np.clip(target, 0, 1),
                               atol=1e-4)


@pytest.mark.parametrize('cls', (SLSQPOptimizer, GradFreeSLSQPOptimizer))
@pytest.mark.parametrize('target,warns', (
    (_X0, True),          # starts at the minimum: no step to take
    (_TARGET, False),
    ), ids=('start at minimum', 'start away'))
def test_warn_if_stalled(cls, target, warns, caplog):
    """Check the warning when a SciPy optimizer ends after <= 1 iteration."""
    def fun(x):
        return float(np.sum((x - target)**2))

    def grad(x):
        return 2 * (x - target)

    if cls is SLSQPOptimizer:
        optimizer = cls(fun=fun, grad=grad, cholesky=_CHOLESKY)
    else:
        optimizer = cls(fun=fun, cholesky=_CHOLESKY)
    with caplog.at_level(logging.WARNING):
        optimizer(_X0.copy())
    assert ('stopped after' in caplog.text) == warns


def test_history_gradient_is_true_gradient():
    """Check that the history records the gradient of the objective."""
    target = _TARGET
    with pytest.warns(DeprecationWarning):
        optimizer = SLSQPOptimizer(
            fun=lambda x: float(np.sum((x - target)**2)),
            grad=lambda x: 2 * (x - target),
            cholesky=_CHOLESKY,
            grad_damp_factor=0.5,
            )
    result = optimizer(_X0.copy())
    x = np.reshape(result.x_history, (-1, 3))
    grad = np.reshape(result.grad_R_history, (-1, 3))
    has_grad = ~np.isnan(grad).any(axis=1)
    assert has_grad.any()
    np.testing.assert_allclose(grad[has_grad], 2 * (x[has_grad] - target))


def _ill_conditioned_case():
    """Return an R-like quadratic with one small CMA-ES std, and starts.

    The quadratic's Hessian is the inverse CMA-ES covariance, with
    standard deviations 3e-3 for one parameter and ~0.2 for the others.
    The starts lie one standard deviation away from the minimum.
    """
    n_params, r_min, amplitude = 5, 0.11, 1e-3
    stds = np.array([3e-3, 0.2, 0.17, 0.2, 0.21])
    correlation = np.full((n_params, n_params), 0.3)
    np.fill_diagonal(correlation, 1.0)
    cholesky = np.linalg.cholesky(np.outer(stds, stds) * correlation)
    hessian = np.linalg.inv(cholesky @ cholesky.T)
    x_min = np.full(n_params, 0.5)

    def fun(x):
        return float(r_min + amplitude * (x - x_min) @ hessian @ (x - x_min))

    def grad(x):
        return 2 * amplitude * hessian @ (x - x_min)

    rng = np.random.default_rng(1)
    starts = []
    for _ in range(10):
        direction = rng.normal(size=n_params)
        direction /= np.linalg.norm(direction)
        starts.append(np.clip(x_min + cholesky @ direction, 0, 1))
    return fun, grad, cholesky, r_min, amplitude, starts


_ILL_CONDITIONED = _ill_conditioned_case()


@pytest.mark.parametrize('start', range(10))
def test_ill_conditioned_reaches_minimum(start):
    """Check that gradient SLSQP with default settings does not stop early.

    The first step changes R by less than 1e-5 here: a stop amount of
    1e-5 or 1e-6 ends SLSQP before it reaches the minimum. With 1e-7,
    SLSQP stops more than 1e-5 of the amplitude above the minimum on
    starts 1, 5, 6 and 7; with the default 1e-8, at most 3.1e-6.
    """
    fun, grad, cholesky, r_min, amplitude, starts = _ILL_CONDITIONED
    optimizer = SLSQPOptimizer(fun=fun, grad=grad, cholesky=cholesky)
    result = optimizer(starts[start].copy())
    assert result.best_R - r_min < 1e-5 * amplitude


@pytest.mark.parametrize('cls', (SLSQPOptimizer, LBFGSBOptimizer))
def test_grad_damp_factor_deprecated(cls, recwarn):
    """Check that grad_damp_factor warns once and changes nothing."""
    def run(**kwargs):
        evaluated = []

        def fun(x):
            evaluated.append(np.array(x))
            return float(np.sum((x - _TARGET)**2))

        def grad(x):
            evaluated.append(np.array(x))
            return 2 * (x - _TARGET)

        def fun_and_grad(x):
            return fun(x), grad(x)

        cls(fun=fun, grad=grad, fun_and_grad=fun_and_grad,
            cholesky=_CHOLESKY, **kwargs)(_X0.copy())
        return np.array(evaluated)

    default = run()
    assert not recwarn
    damped = run(grad_damp_factor=0.1)
    assert [w.category for w in recwarn] == [DeprecationWarning]
    np.testing.assert_array_equal(damped, default)
