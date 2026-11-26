"""Tests for the entry points used by the viperleed.calc.vlj hooks."""

__authors__ = ('Florian Kraushofer (@fkraushofer)',)
__created__ = '2026-11-26'

import numpy as np
import pytest
from viperleed.calc.files.new_displacements.file import DisplacementsFile

from tests.fixtures.cu_111_fixed_l_max import (
    _DATA_PATH,
    _LARGE_DATA_PATH,
)
from viperleed_jax import optimization
from viperleed_jax.from_objects import (
    setup_tl_calculator,
    setup_tl_parameter_space,
)
from viperleed_jax.from_state import run_viperleed_initialization
from viperleed_jax.optimization.iterator import OptimizerIterator

_CMAES = optimization.optimizer.CMAESOptimizer
_SLSQP = optimization.optimizer.GradFreeSLSQPOptimizer


def test_optimizer_chain(monkeypatch):
    """Check that a CMAES -> SLSQP chain hands the clipped result on."""
    # Set up rpars and the calculator as calc/vlj/search.py does
    state = run_viperleed_initialization(_DATA_PATH)
    slab, rpars = state.slab, state.rpars
    rpars.VLJ_ALGO = ['CMAES', 'SLSQP']
    rpars.VLJ_CONFIG['preoptimize_v0r'] = False
    rpars.vlj_algo_settings['CMAES'].update(
        pop=6, max_gens=2, convergence_gens=2
    )
    rpars.vlj_algo_settings['SLSQP'].update(grad=False)

    parameter_space = setup_tl_parameter_space(slab, rpars)
    disp_file = DisplacementsFile()
    disp_file.read(_DATA_PATH / 'DISPLACEMENTS')
    if disp_file.offsets is not None:
        parameter_space.apply_offsets(disp_file.offsets)
    parameter_space.apply_search_segment(disp_file.next(0.9))

    calculator = setup_tl_calculator(
        slab,
        rpars,
        tensor_path=_LARGE_DATA_PATH / 'Tensors' / 'Tensors_001.zip',
        phaseshifts_path=_DATA_PATH / 'PHASESHIFTS',
    )
    calculator.set_parameter_space(parameter_space)
    calculator.set_rfactor(str(rpars.R_FACTOR_TYPE))

    start_points, choleskys = [], []

    def _spy(cls):
        original = cls.__call__

        def _call(self, x0, *args, **kwargs):
            start_points.append((cls, np.array(x0, dtype=float)))
            choleskys.append(getattr(self, 'cholesky', None))
            return original(self, x0, *args, **kwargs)

        monkeypatch.setattr(cls, '__call__', _call)

    _spy(_CMAES)
    _spy(_SLSQP)

    iterator = OptimizerIterator(calculator=calculator, rpars=rpars)
    assert not iterator.scheduled_gradients
    starting_x = iterator.suggested_starting_point
    n_params = calculator.n_free_parameters
    assert starting_x.shape == (n_params,)
    assert np.all((starting_x >= 0) & (starting_x <= 1))

    results = list(iterator)

    assert [type(opt) for opt, _ in results] == [_CMAES, _SLSQP]
    assert [cls for cls, _ in start_points] == [_CMAES, _SLSQP]
    for _, result in results:
        assert result.best_x.shape == (n_params,)
        assert np.isfinite(result.best_R)
        assert 0 <= result.best_R <= 2
    # each optimizer starts from the (clipped) best point of the previous
    cmaes_result = results[0][1]
    np.testing.assert_allclose(
        start_points[1][1], np.clip(cmaes_result.best_x, 0.0, 1.0)
    )
    assert np.all((start_points[1][1] >= 0) & (start_points[1][1] <= 1))
    # SLSQP is preconditioned with the CMA-ES Cholesky factor
    assert cmaes_result.cholesky is not None
    np.testing.assert_array_equal(iterator.cholesky, cmaes_result.cholesky)
    np.testing.assert_array_equal(choleskys[1], cmaes_result.cholesky)
    assert iterator.current_x.shape == (n_params,)
    with pytest.raises(StopIteration):
        next(iterator)
