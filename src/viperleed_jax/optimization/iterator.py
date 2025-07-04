"""Module optimization.iterator."""

__authors__ = (
    'Alexander M. Imre (@amimre)',
)
__created__ = '2025-07-04'

import numpy as np
from viperleed.calc import LOGGER as logger

from viperleed_jax import optimization


class OptimizerIterator:
    """An iterator for managing multiple optimization algorithms in sequence."""

    def __init__(self, rpars, calculator, starting_x=None):
        """
        Initialize the optimizer iterator with the required parameters.

        :param rpars: The run parameters object containing configuration.
        :param calculator: The calculator object to be used for optimization.
        """
        self.rpars = rpars
        self.calculator = calculator
        self._cholesky = None # default to None
        self._upcoming_optimizers = self.rpars.VLJ_ALGO
        self._done_optimizers = []

        self._DISPATCH = {
            'CMAES': self._get_cmaes_optimizer,
            'SLSQP': self._get_slsqp_optimizer,
            'BFGS': self._get_bfgs_optimizer,
        }

        if starting_x is not None:
            self.set_x(starting_x)
        else:
            # If no starting point is provided, use the suggested starting point
            # based on the first optimizer in the sequence.
            logger.debug(
                'No starting point provided. Using suggested starting point.'
            )
            self._current_x = self.suggested_starting_point


        if not all(opt in self._DISPATCH for opt in self._upcoming_optimizers):
            msg = (
                f"Invalid optimizer(s) specified: {self._upcoming_optimizers}. "
                f"Valid options are: {list(self._DISPATCH.keys())}."
            )
            raise ValueError(msg)

    def set_x(self, x):
        """Set the parameter vector used as input for the next optimization."""
        try:
            x = np.asarray(x, dtype=float)
        except ValueError as e:
            raise ValueError(
                "The input 'x' must be convertible to a float array."
            ) from e
        if x.shape != (self.calculator.n_free_parameters,):
            msg = (
                f"The input 'x' must have shape "
                f"({self.calculator.n_free_parameters},)."
            )
            raise ValueError(msg)
        self._current_x = x

    @property
    def current_x(self):
        """Return the current point in the optimization process."""
        return self._current_x

    @property
    def cholesky(self):
        """If enabled and available, return the Cholesky factor."""
        if not self.rpars.VLJ_CONFIG['precondition']:
            return None
        return self._cholesky

    @property
    def suggested_starting_point(self):
        """Return a suggested starting point for the iterator.

        If the first optimizer uses gradients, we want to start slightly away
        from the center ([0.5, 0.5, 0.5, ...])to avoid numerical issues.
        If the first optimizer does not use gradients, return the center point.
        """
        first_optimizer =  self._DISPATCH[self.rpars.VLJ_ALGO[0]]
        center = np.array([0.5] * self.calculator.n_free_parameters)
        if isinstance(first_optimizer, optimization.optimizer.GradOptimizer):
            # Use a small offset to avoid numerical issues
            pattern = np.array([0.001, -0.001])
            offset = np.tile(
                pattern, self.calculator.n_free_parameters // 2 + 1
            )
            center += offset[: self.calculator.n_free_parameters]
        return center

    def __iter__(self):
        """Return the iterator object itself."""
        return self

    def __next__(self):
        """Run the next part of the optimizer sequence and return the result.

        Raises
        ------
            StopIteration: If there are no more optimizers to iterate over.
        """
        if not self._upcoming_optimizers:
            raise StopIteration

        # get the next optimizer from the list
        optimizer_name = self._upcoming_optimizers.pop(0)
        optimizer = self._DISPATCH[optimizer_name]()

        # run the optimizer with the current parameter vector
        logger.debug(
            f'Running optimizer: {optimizer_name} with x={self._current_x}'
        )
        result = optimizer(self._current_x)
        self._done_optimizers.append(optimizer)
        self._process_result(result)

        return optimizer, result


    def _process_result(self, result):
        """Carry over any necessary information from the previous optimizer."""
        # update the parameter vector for the next optimizer
        self._current_x = result.best_x

        last_optimizer = self._done_optimizers[-1]
        if isinstance(last_optimizer, optimization.optimizer.CMAESOptimizer):
            self._cholesky = result.cholesky
            logger.debug(
                'Carrying over Cholesky factor from CMA-ES optimization.'
            )

    def _get_cmaes_optimizer(self):
        logger.debug('Preparing CMA-ES optimizer.')
        return optimization.optimizer.CMAESOptimizer(
            fun=self.calculator.R,
            n_generations=self.rpars.vlj_algo_settings['CMAES']['max_gens'],
            pop_size=self.rpars.vlj_algo_settings['CMAES']['pop'],
            ftol=self.rpars.vlj_algo_settings['CMAES']['ftol'],
        )

    def _get_slsqp_optimizer(self):
        logger.debug('Preparing SLSQP optimizer.')
        if self.rpars.vlj_algo_settings['SLSQP']['grad']:
            return optimization.optimizer.SLSQPOptimizer(
                fun=self.calculator.R,
                grad=self.calculator.grad_R,
                cholesky=self.cholesky,
                grad_damp_factor=self.rpars.vlj_algo_settings['SLSQP'][
                    'grad_damping'
                ],
            )

        # otherwise, use the gradient-free SLSQP optimizer
        return optimization.optimizer.GradFreeSLSQPOptimizer(
            fun=self.calculator.R,
            cholesky=self.cholesky,
        )

    def _get_bfgs_optimizer(self):
        logger.debug('Preparing BFGS optimizer.')
        if self.rpars.vlj_algo_settings['BFGS']['grad']:
            return optimization.BFGSOptimizer(
                fun=self.calculator.R,
                grad=self.calculator.grad_R,
                cholesky=self.cholesky,
            )
        return optimization.optimizer.GradFreeBFGSOptimizer(
            fun=self.calculator.R,
            cholesky=self.cholesky,
        )
