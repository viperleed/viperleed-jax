"""Module optimization."""

__authors__ = (
    'Paul Haidegger (@PaulHai7)',
    'Alexander M. Imre (@amimre)',
)
__created__ = '2024-11-20'

import time
from abc import ABC, abstractmethod

import numpy as np
from clinamen2.cmaes.params_and_state import (
    create_sample_from_state,
    create_update_algorithm_state,
)
from clinamen2.utils.script_functions import cma_setup
from scipy.optimize import minimize
from viperleed.calc import LOGGER as logger


class Optimizer(ABC):
    """Class for all the optimizers.

    Parameters
    ----------
        fun: Objective function
    """

    def __init__(self, fun):
        self.fun = fun

    @abstractmethod
    def __call__(self):
        """Start the optimization."""

class GradOptimizer(Optimizer):
    """Class for optimizers that use a gradient.

    Parameters
    ----------
        fun: Objective function.
        grad: Gradient of the objective function.
        fun_and_grad: Function value and gradient together. This approach
            is faster, but it only makes sense, if they are always or almost
            always computed together (e.g., in L-BFGS-B). Avoid using it when
            having a lot function calls without the gradient (e.g., in SLSQP).
    """

    def __init__(self, fun, grad, fun_and_grad):
        self.fun = fun
        self.grad = grad
        self.fun_and_grad = fun_and_grad
        super().__init__(fun=fun)


class NonGradOptimizer(Optimizer):
    """Class for optimizers that don't use a gradient."""

    def __init__(self, fun):
        self.fun = fun
        super().__init__(fun=fun)


class LBFGSBOptimizer(GradOptimizer):
    """Class for setting up the L-BFGS-B algorithm for local minimization.

    The BFGS algorithm uses the BFGS approximation of the Hessian, which is
    always positive definite. Gradient and Hessian (approximation) are used to
    determine search direction. A line search is performed along this direction,
    which must satisfy the Wolfe conditions. These conditions provide an upper
    and lower limit for the step size, and one condition also ensures that the
    function value decreases for each iteration.

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

    def __init__(self, fun_and_grad, bounds=None, ftol=1e-7, maxiter=1000):
        self.fun_and_grad = fun_and_grad
        self.bounds = bounds
        self.ftol = ftol
        self.maxiter = maxiter
        super().__init__(fun_and_grad=fun_and_grad, grad=None, fun=None)

    def __call__(self, start_point):
        """Start the optimization algorithm.

        This function prints a termination message and returns all the values
        that are also returned by the SciPy function, plus a list of the
        function values for each iteration (fun_hystory) and the
        runtime (duration).

        Parameters
        ----------
            start_point: Starting point of the algorithm.
        """
        # Setting up Callback function to save function history in fun_history
        fun_history = []
        current_fun = [None]
        current_grad = [None]

        def fun_with_storage(x):
            current_fun[0], current_grad[0] = self.fun_and_grad(x)
            return current_fun[0], current_grad[0]

        def callback_function(x):
            fun_history.append(current_fun[0])

        # Setting up the bounds
        if self.bounds is None:
            bounds = [(0, 1) for _ in range(len(start_point))]
        else:
            bounds = self.bounds

        # Performing the optimization
        start_time = time.time()
        result = minimize(
            fun_with_storage,
            x0=start_point,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            callback=callback_function,
            options={'maxiter': self.maxiter, 'ftol': self.ftol},
        )
        end_time = time.time()
        duration = end_time - start_time
        result.fun_history = fun_history
        result.duration = duration
        logger.info('Optimization Result:\n')
        logger.info(f'{str(result)}\n\n')
        return result


class SLSQPOptimizer(GradOptimizer):
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

    def __init__(
        self, fun, grad, bounds=None, damp_fact=1, ftol=1e-6, maxiter=1000
    ):
        self.fun = fun
        self.grad = grad
        self.bounds = bounds
        self.damp_fact = damp_fact
        self.ftol = ftol * damp_fact
        self.maxiter = maxiter
        super().__init__(fun_and_grad=None, grad=grad, fun=fun)

    def __call__(self, start_point):
        """Start the optimization.

        This function prints a termination message and returns all the values
        that are also returned by the SciPy function, plus a list of the
        function values for each iteration (fun_history) and the
        runtime (duration).

        Parameters
        ----------
            start_point: Starting point of the algorithm.
        """
        # Setting up Callback function to save function history in fun_history
        fun_history = []
        current_fun = [None]

        def dampened_grad(x):
            return self.damp_fact * self.grad(x)

        def dampened_fun_storage(x):
            current_fun[0] = self.fun(x)
            return current_fun[0] * self.damp_fact

        def callback_function(x):
            fun_history.append(current_fun[0])

        # Setting up the bounds
        if self.bounds is None:
            bounds = [(0, 1) for _ in range(len(start_point))]
        else:
            bounds = self.bounds

        # Performing the optimization
        start_time = time.time()
        result = minimize(
            fun=dampened_fun_storage,
            x0=start_point,
            method='L-BFGS-B',
            jac=dampened_grad,
            bounds=bounds,
            callback=callback_function,
            options={'maxiter': self.maxiter, 'ftol': self.ftol},
        )
        end_time = time.time()
        duration = end_time - start_time
        result.fun_history = fun_history
        result.duration = duration
        logger.info('Optimization Result:\n')
        logger.info(f'{str(result)}\n\n')
        return result


class CMAESOptimizer(NonGradOptimizer):
    """Class for setting up the CMA-ES optimizer for global exploration.

    In each evolution, a number of individuals are drawn from a distribution,
    and the distribution is updated based on the fanction values of the
    individuals. For the normalized vector, a step size of 0.5 showed very
    good results. A population size of 30 for 33 dimensions has proven
    successful. However, the population size should increase with the number
    of dimensions (not linearly, but more logarithmically). For such a large
    step size, 100-200 generations have showen great success.

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

    def __init__(self, fun, pop_size, n_generations, step_size=0.5, ftol=1e-4):
        self.fun = fun
        self.step_size = step_size
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.ftol = ftol
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
                convergence or reaching the maximum nuber of generations.
            current_generation: Number of performed generations.
            duration: Total runtime.
            fun_history: All function values of all generations stored in a
                2D array.
            step_size_history: Step size of each generation stored.
        """
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

        start_time = time.time()
        fun_history = []
        step_size_history = []
        loss_min = np.full((5,), fill_value=10.0)
        termination_message = 'Maximum number of generations reached'
        # Perform the optimization
        for g in range(self.n_generations):
            # Perform one generation
            generation, state, fun_value = sample_and_evaluate(
                state=state, n_samples=parameters.pop_size
            )
            fun_history.append(fun_value)
            step_size_history.append(state.step_size)
            # To update the AlgorithmState pass in the sorted generation
            state = update_state(state, generation[np.argsort(fun_value)])
            i = g % 5
            loss_min[i] = fun_value.min()
            if np.std(loss_min) < self.ftol:
                termination_message = (
                    f'Evolution terminated early at generation {g}.'
                )
                break

        end_time = time.time()
        duration = end_time - start_time
        if (generation[fun_value.argmin()] < 0.1).any() or (
            generation[fun_value.argmin()] > 0.9
        ).any():
            logger.warning('Parameter(s) close to the bounds!')
        # Create result object
        result = CMAESResult(
            min_individual=generation[fun_value.argmin()],
            fun=fun_value.min(),
            message=termination_message,
            current_generation=g,
            duration=duration,
            fun_history=fun_history,
            step_size_history=step_size_history,
        )
        # print the minimum function value in the final generation
        logger.info(
            f'Loss {fun_value.min()} for individual '
            f'{fun_value.argmin()} in generation {g}. '
            f'With Parameters: {generation[fun_value.argmin()]} \n'
            f'evaluation time: {duration} seconds'
        )
        return result


class CMAESResult:
    """Class for the output of the CMA-ES algorithm.

    Parameters
    ----------
        min_individual: Parameters of the individual with the smallest 
            function value.
        fun: Smallest function value.
        message: A message indicating wether the algorithm finished due to
            convergence or reaching the maximum nuber of generations.
        current_generation: Number of performed generations.
        duration: Total runtime.
        fun_history: All function values of all generations stored in a
            2D array.
        step_size_history: Step size of each generation stored.
    """
    def __init__(
        self,
        min_individual,
        fun,
        message,
        current_generation,
        duration,
        fun_history,
        step_size_history,
    ):
        self.min_individual = min_individual
        self.fun = fun
        self.message = message
        self.current_generation = current_generation
        self.duration = duration
        self.fun_history = fun_history
        self.step_size_history = step_size_history

    def __repr__(self):
        """Returns a string representation of the optimization result."""
        return (
            f'OptimizationResult(x = {self.min_individual}\n'
            f'fun = {self.fun}\n'
            f'message = {self.message}\n'
            f'current_generation = {self.current_generation}\n'
            f'duration = {self.duration:.2f}s)'
        )


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
        local_result = self.local_optimizer(global_result.x)

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
