"""Module optimization."""

__authors__ = (
    'Paul Haidegger (@PaulHai7)',
    'Alexander M. Imre (@amimre)',
)
__created__ = '2024-11-20'

import time
from abc import ABC, abstractmethod

from .result import CMAESResult, GradOptimizerResult
from .history import GradOptimizationHistory, EvolutionOptimizationHistory

import tqdm
import numpy as np
from clinamen2.cmaes.params_and_state import (
    create_sample_from_state,
    create_update_algorithm_state,
)
from clinamen2.utils.script_functions import cma_setup
from scipy.optimize import minimize
from viperleed.calc import LOGGER as logger


class Optimizer(ABC):
    """Base class for all the optimizers.

    Parameters
    ----------
        fun: Objective function
    """

    def __init__(self, fun):
        self.fun = fun
        self.fun_history = []


    @abstractmethod
    def __call__(self):
        """Start the optimization."""


class GradOptimizer(
    Optimizer
):
    """Class for optimizers that use a gradient.

    Parameters
    ----------
        fun: Objective function to be optimized.
        grad: Gradient of the objective function.
        fun_and_grad: Function value and gradient together. This approach
            is faster, but it only makes sense, if they are always or almost
            always computed together (e.g., in L-BFGS-B). Avoid using it when
            having a lot function calls without the gradient (e.g., in SLSQP).
    """

    def __init__(self, fun, grad=None, fun_and_grad=None):
        if grad is None and fun_and_grad is None:
            raise ValueError(
                'At least one of grad or fun_and_grad must be set.'
            )
        self.fun = fun
        # Use the provided gradient or derive it from fun_and_grad
        self.grad = (
            grad if grad is not None else (lambda arg: fun_and_grad(arg)[1])
        )
        self.fun_and_grad = (
            fun_and_grad
            if fun_and_grad is not None
            else (lambda arg: (fun(arg), grad(arg)))
        )
        super().__init__(fun=fun)
        self.current_fun = 0
        self.current_grad = 0


class SciPyGradOptimizer(GradOptimizer):
    """Gradient based optimizers that wrap SciPy's optimize.minimize.

    TODO: docstring
    """
    def __init__(self, fun=None, grad=None, fun_and_grad=None, bounds=None, **kwargs):
        super().__init__(fun, grad, fun_and_grad, **kwargs)
        self.bounds = bounds
        self.options={}

    @abstractmethod
    def method(self):
        pass

    @abstractmethod
    def combined_fun_and_grad(self):
        pass

    def transform_bounds(self, x0, L):
        """Transform the bounds according to the current transformation.

        Parameters
        ----------
        x0 : ndarray
            The initial guess.
        L : ndarray
            The transformation matrix.

        Returns
        -------
        list of tuple
            Transformed bounds.
        """
        # If no bounds are set, default to [0, 1] for each dimension.
        bounds = [(0, 1)] * len(x0) if self.bounds is None else self.bounds
        x_min, x_max = np.array(bounds).T
        # Transform the bounds
        x_min_transformed = L.T @ (x_min - x0)
        x_max_transformed = L.T @ (x_max - x0)
        # Ensure lower bounds are always smaller than upper bounds
        x_min_corrected = np.minimum(x_min_transformed, x_max_transformed)
        x_max_corrected = np.maximum(x_min_transformed, x_max_transformed)
        return list(zip(x_min_corrected, x_max_corrected))


    def __call__(self, x0, L=None):
        """Run the optimization."""
        opt_history = GradOptimizationHistory()

        if L is None:
            L = np.eye(len(x0))
        L_inv = np.linalg.inv(L)

        def _fun(y):
            x = x0 + L_inv.T @ y  # Transform y back to x
            fun_val = self.fun(x)
            opt_history.append(x, R=fun_val, grad_R=None)
            return fun_val

        def _grad(y):
            x = x0 + L_inv.T @ y
            _grad_x = self.grad(x)
            opt_history.append(x, R=None, grad_R=_grad_x)
            return L_inv @ _grad_x  # Transform gradient

        def _fun_and_grad(y):
            x = x0 + L_inv.T @ y
            fun_val, grad_x = self.fun_and_grad(x)
            grad_y = L_inv @ grad_x  # Transform gradient
            opt_history.append(x, R=fun_val, grad_R=grad_y)
            return fun_val, grad_y

        # Transform initial guess
        y0 = np.zeros_like(x0)

        # get transformed bounds
        transformed_bounds = self.transform_bounds(x0, L)

        scipy_result = minimize(
            fun=_fun_and_grad if self.combined_fun_and_grad else _fun,
            x0=y0,
            method=self.method,
            jac=True if self.combined_fun_and_grad else _grad,
            bounds=transformed_bounds,
            options=self.options,
        )
        return GradOptimizerResult(scipy_result, opt_history)


class NonGradOptimizer(Optimizer):
    """Class for optimizers that do not use gradients."""

    def __init__(self, fun):
        self.fun = fun
        super().__init__(fun=fun)


class LBFGSBOptimizer(SciPyGradOptimizer):
    """Class for setting up the L-BFGS-B algorithm for local minimization.

    The BFGS algorithm uses the BFGS approximation of the Hessian, which is
    always positive definite. Gradients and Hessians (approximation) are used to
    determine the search direction. A line search is performed along this
    direction, which must satisfy the Wolfe conditions. These conditions provide
    an upper and lower limit for the step size, and one condition also ensures
    that the function value monotonically decreases for each iteration.

    Parameters
    ----------
        fun_and_grad: Function value and gradient together. This approach
            is faster, but it only makes sense, if they are always or almost
            always computed together (e.g., in L-BFGS-B). Avoid using it when
            having a lot function calls without the gradient (e.g., in SLSQP).
        bounds: Since the parameters are normalized, all bounds are by default
            set to [0,1].
        ftol: Convergence condition that sets a lower limit on the difference
            in function value between two iterations. A higher value has been
            shown to lead to a faster termination without significantly
            changeing the R factor. It can even be set to 1e-6 for a faster
            converence, but this slightly worsens the R factor.
        maxiter: Maximal number of iterations for the algorithm. Usually, the
            algorithm stops earlier due to convergence.
    """

    method='L-BFGS-B'
    combined_fun_and_grad = True

    def __init__(self, fun=None, grad=None, fun_and_grad=None, bounds=None, ftol=1e-7, maxiter=1000):
        super().__init__(fun=fun, grad=grad, fun_and_grad=fun_and_grad, bounds=bounds)
        self.options = {'maxiter': maxiter, 'ftol': ftol}


class SLSQPOptimizer(SciPyGradOptimizer):
    """Class for setting up the SLSQP algorithm for local minimization.

    The SLSQP algorithm uses a quadratic approximation of the Lagrangian to
    include equality constraints. Inequality constraints are incorporated by
    defining active and passive sets, as well as an upper limit of the step
    size. For the Hessian, the BFGS approximation is used, and the line search
    cannot satisfy the Wolfe conditions. Due to a poor Hessian approximation in
    the initial iterations, the algorithm tends to overshoot during these
    iterations. To address this issue, a damping factor (damp_fact) is used to
    reduce the gradient at the beginning. A damping factor of 0.1 has shown good
    results.

    Parameters
    ----------
        fun: Objective function.
        grad: Gradient of the objective function.
        bounds: Since the parameters are normalized, all bounds are by default
            set to [0,1].
        damp_factor: Damping factor.
        ftol: Convergence condition that sets a lower limit on the difference
            in function value between two iterations.
        maxiter: Maximal number of iterations for the algorithm. Usually, the
            algorithm stops earlier due to convergence.
    """

    method = 'SLSQP'
    combined_fun_and_grad = False

    def __init__(
        self,
        fun=None,
        grad=None,
        fun_and_grad=None,
        bounds=None,
        damp_fact=1,
        ftol=1e-6,
        maxiter=1000,
    ):
        super().__init__(fun=fun, grad=grad, fun_and_grad=fun_and_grad, bounds=bounds)
        self.bounds = bounds
        self.damp_fact = damp_fact
        self.options = {'maxiter': maxiter, 'ftol': ftol * damp_fact}

class CMAESOptimizer(NonGradOptimizer):
    """Class for setting up the CMA-ES optimizer for global exploration.

    In each evolution, a number of individuals are drawn from a distribution,
    and the distribution is updated based on the fanction values of the
    individuals. For the normalized vector, a step size of 0.5 showed very
    good results. A population size of 30 for 33 dimensions has proven
    successful. However, the population size should increase with the number
    of dimensions (not linearly, but more logarithmically). For such a large
    step size, 100-200 generations have shown great success.

    Parameters
    ----------
        fun: Objective function.
        pop_size: Number of individuals in each generation.
        n_generations: Maximum number of generations to be performed.
        step_size: The standard deviatian in the initial step and a
            parameter for how much the algorithm should focus on exploring.
            A step size of 0.5 is quite large, but showed the best results.
        ftol: Convergence condition on the standard deviation of the minimum
            function value of the last five generations.
    """

    def __init__(self, fun, pop_size, n_generations, step_size=0.5, ftol=1e-4,
                 convergence_gens=5):
        self.fun = fun
        self.step_size = step_size
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.ftol = ftol
        self.convergence_gens = convergence_gens
        super().__init__(fun=fun)

    def __call__(self, start_point):
        """Start the optimization.

        This function prints a termination message, and the return values
        are explained below.

        Parameters
        ----------
            start_point: Starting point of the algorithm. Usually it start at
                0.5 for each dimension, as this is in the middle of the bounds.

        Returns
        -------
            min_individual: Parameters of the individual with the smallest
                function value.
            message: A message indicating wether the algorithm finished due to
                convergence or reaching the maximum number of generations.
            current_generation: Number of performed generations.
            duration: Total runtime.
            fun_history: All function values of all generations stored in a
                2D array.
            step_size_history: Step size of each generation stored.
        """
        # Initialize history
        opt_history = EvolutionOptimizationHistory()

        # Set up functions for the algorithm
        parameters, initial_state = cma_setup(
            mean=start_point, step_size=self.step_size, pop_size=self.pop_size
        )
        sample_individuals = create_sample_from_state(parameters)
        update_state = create_update_algorithm_state(parameters=parameters)
        sample_and_evaluate = create_resample_and_evaluate(
            sample_individuals=sample_individuals,
            evaluate_single=self.fun,
        )
        state = initial_state

        step_size_history = []
        generation_time_history = []
        loss_min = np.full((self.convergence_gens,), fill_value=10.0)
        termination_message = 'Maximum number of generations reached'
        start_time = time.time()
        # Perform the optimization
        for g in tqdm.trange(self.n_generations):
            # Perform one generation
            generation, state, fun_value = sample_and_evaluate(
                state=state, n_samples=parameters.pop_size
            )
            opt_history.append(generation_x=generation,
                               generation_R=fun_value,
                               step_size=state.step_size)

            # To update the AlgorithmState pass in the sorted generation
            state = update_state(state, generation[np.argsort(fun_value)])
            i = g % self.convergence_gens
            loss_min[i] = fun_value.min()
            if np.std(loss_min) < self.ftol:
                termination_message = (
                    f'Evolution terminated early at generation {g}.'
                )
                break

        if (generation[fun_value.argmin()] < 0.1).any() or (
            generation[fun_value.argmin()] > 0.9
        ).any():
            logger.warning('Parameter(s) close to the bounds!')

        # Create result object
        result = CMAESResult(
            evolution_history=opt_history,
            message=termination_message,
            cholesky=state.cholesky_factor,
            convergence_generations=self.convergence_gens,
        )
        # print the minimum function value in the final generation
        logger.info(result.__repr__())
        return result


class SequentialOptimizer(Optimizer):
    """Class to run two optimizers sequentially.

    First, a global optimizer (e.g., CMA-ES) is run, followed by a local
    optimizer (e.g., SLSQP) for refining the result.

    Parameters
    ----------
        global_optimizer: Instance of a global optimizer.
        local_optimizer: Instance of a local optimizer.
    """

    def __init__(self, global_optimizer, local_optimizer):
        self.global_optimizer = global_optimizer
        self.local_optimizer = local_optimizer
        super().__init__(fun=global_optimizer.fun)

    def __call__(self, start_point):
        """Run the optimization pipeline.

        Parameters
        ----------
            start_point: Starting point for the global optimizer.

        Returns
        -------
            A dictionary containing the results of both optimizations.
        """
        logger.info('Starting global optimization (CMA-ES)...')
        # Run the global optimizer
        global_result = self.global_optimizer(start_point)

        logger.info('Global optimization finished.\n')
        logger.info('Starting local optimization')
        # Use the result of the global optimizer for the local optimizer
        local_result = self.local_optimizer(global_result.min_individual)

        logger.info('Local optimization finished.')

        # Combine results into a single dictionary
        return {
            'global_result': global_result,
            'local_result': local_result,
        }


def create_resample_and_evaluate(
    sample_individuals,
    evaluate_single,
):
    """Create function that samples a population and evaluates its function.

    Samples that are outside the bounds are partially resampled, which
    means that only the components which are outside are resampled.

    Parameters
    ----------
        sample_individuals: Function that samples a number of individuals from
            a state.
        evaluate_single: Function that returns a tuple containing the loss of
            an individual and additional information in a dictionary (at least
            exception if applicable).

    Returns
    -------
        A function to sample (with resampling) and evaluate a population from a
        state.
    """

    def resample_and_evaluate(
        state,
        n_samples,
        n_attempts=int(1e6),
    ):
        """Sample a population from a state and evaluate the objective function.

        When the resampling number reaches n_attempts an OverflowError is
        raised.

        Parameters
        ----------
            state: State of the previous CMA step.
            n_samples: Number of successfully evaluated individuals to be
                returned.
            n_attempts: Maximum number of attempts to reach n_samples.
                Default is 1e6 to avoid infinite loops.

        Returns
        -------
            tuple containing the following elements:
            - A population of individuals sampled from the AlgorithmState.
            - The new AlgorithmState.
            - An array containing the function value of all passing individuals.
        """
        population = []
        loss = []
        attempts = 0

        # resample and single evaluate individuals
        while attempts <= n_attempts and len(population) < n_samples:
            attempts += 1
            resampled_population, state = sample_individuals(state, n_samples=1)
            while (resampled_population[0] < 0.0).any() or (
                resampled_population[0] > 1.0
            ).any():
                resampled_population2, state = sample_individuals(
                    state, n_samples=1
                )
                condition = np.logical_or(
                    resampled_population[0] > 1, resampled_population[0] < 0
                )
                resampled_population = np.where(
                    condition, resampled_population2, resampled_population
                )
            population.append(resampled_population[0])
            loss.append(evaluate_single(resampled_population[0]))

        if len(population) < n_samples:
            msg = f'Evaluation attempt limit of {n_attempts} reached '
            raise OverflowError(msg)

        return (
            np.asarray(population),
            state,
            np.asarray(loss),
        )

    return resample_and_evaluate
