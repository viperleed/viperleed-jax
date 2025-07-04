"""Module optimization.iterator."""

__authors__ = (
    'Alexander M. Imre (@amimre)',
)
__created__ = '2025-07-04'

from viperleed.calc import LOGGER as logger

from viperleed_jax import optimization


class OptimizerIterator:
    """An iterator for managing multiple optimization algorithms in sequence."""

    def __init__(self, rpars, calculator):
        """
        Initialize the optimizer iterator with the required parameters.

        :param rpars: The run parameters object containing configuration.
        :param calculator: The calculator object to be used for optimization.
        """
        self.rpars = rpars
        self.calculator = calculator
        self.current_step = 0
        self._cholesky = None # default to None
        self._upcoming_optimizers = self.rpars.VLJ_ALGO
        self._done_optimizers = []

        self._DISPATCH = {
            'CMAES': self._get_cmaes_optimizer,
            'SLSQP': self._get_slsqp_optimizer,
            'BFGS': self._get_bfgs_optimizer,
        }

        if not all(opt in self._DISPATCH for opt in self._upcoming_optimizers):
            msg = (
                f"Invalid optimizer(s) specified: {self._upcoming_optimizers}. "
                f"Valid options are: {list(self._DISPATCH.keys())}."
            )
            raise ValueError(msg)

    @property
    def cholesky(self):
        """If enabled and available, return the Cholesky factor."""
        if not self.rpars.VLJ_CONFIG['precondition']:
            return None
        return self._cholesky

    def __iter__(self):
        """Return the iterator object itself."""
        return self

    def __next__(self):
        """Return the next optimizer in the sequence.

        Raises
        ------
            StopIteration: If there are no more optimizers to iterate over.
        """
        if not self._upcoming_optimizers:
            raise StopIteration
        self._process_previous_optimizer()

        # get the next optimizer from the list
        optimizer_name = self._upcoming_optimizers.pop(0)
        next_optimizer = self._DISPATCH[optimizer_name]()
        self._done_optimizers.append(next_optimizer)
        return next_optimizer

    def _process_previous_optimizer(self):
        """Carry over any necessary information from the previous optimizer."""
        if not self._done_optimizers:
            return
        last_optimizer = self._done_optimizers[-1]
        if isinstance(last_optimizer, optimization.CMAESOptimizer):
            self._cholesky = last_optimizer.cholesky
            logger.debug(
                'Carrying over Cholesky factor from CMA-ES optimization.'
            )

    def _get_cmaes_optimizer(self):
        logger.debug('Preparing CMA-ES optimizer.')
        return optimization.CMAESOptimizer(
            fun=self.calculator.R,
            n_generations=self.rpars.vlj_algo_settings['CMAES']['max_gens'],
            pop_size=self.rpars.vlj_algo_settings['CMAES']['pop'],
            ftol=self.rpars.vlj_algo_settings['CMAES']['ftol'],
        )

    def _get_slsqp_optimizer(self):
        logger.debug('Preparing SLSQP optimizer.')
        if self.rpars.vlj_algo_settings['SLSQP']['grad']:
            return optimization.SLSQPOptimzer(
                fun=self.calculator.R,
                grad=self.calculator.grad_R,
                cholesky=self.cholesky,
                grad_damp=self.rpars.vlj_algo_settings['SLSQP']['grad_damping'],
            )

        # otherwise, use the gradient-free SLSQP optimizer
        return optimization.GradFreeSLSQPOptimizer(
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
        return optimization.GradFreeBFGSOptimizer(
            fun=self.calculator.R,
            cholesky=self.cholesky,
        )
