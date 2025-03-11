"""Module optimization.result."""

__authors__ = (
    'Alexander M. Imre (@amimre)',
    'Paul Haidegger (@PaulHai7)',
)
__created__ = '2025-03-11'

from abc import ABC, abstractmethod

import numpy as np


class OptimizationResult(ABC):
    
    def best_R(self):
        return self.best

    def best_x(self):
        return self.min_individual

    @abstractmethod
    def __repr__(self):
        """Return a string representation of the optimization result."""

    def write_to_file(self, file_path):
        """Write the optimization result to a file."""

class CMAESResult(OptimizationResult):
    """Class for the output of the CMA-ES algorithm.

    Parameters
    ----------
        min_individual: Parameters of the individual with the smallest
            function value.
        best: Smallest function value.
        message: A message indicating wether the algorithm finished due to
            convergence or reaching the maximum number of generations.
        current_generation: Number of performed generations.
        duration: Total runtime.
        fun_history: All function values of all generations stored in a
            2D array.
        step_size_history: Step size of each generation stored.
    """

    def __init__(
        self,
        min_individual,
        best,
        message,
        current_generation,
        duration,
        fun_history,
        step_size_history,
        generation_time_history,
        cholesky,
    ):
        self.min_individual = min_individual
        self.best = best
        self.message = message
        self.current_generation = current_generation
        self.duration = duration
        self.fun_history = fun_history
        self.step_size_history = np.array(step_size_history)
        self.generation_time_history = np.array(generation_time_history)
        self.cholesky = cholesky

    def __repr__(self):
        """Return a string representation of the optimization result."""
        return (
            f'OptimizationResult(x = {self.min_individual}\n'
            f'Best R = {self.best}\n'
            f'message = {self.message}\n'
            f'current_generation = {self.current_generation}\n'
            f'duration = {self.duration:.2f}s'
        )


class GradOptimizerResult(OptimizationResult):
    def __init__(self, scipy_result, history):
        self.iterations = scipy_result.nit
        self.message = scipy_result.message
        self.history = history

    @property
    def best_R(self):
        return self.x_history[-1][1]

    @property
    def best_x(self):
        return self.x_history[-1][0]

    @property
    def duration(self):
        return self.history.duration

    def __repr__(self):
        return (
            f'Best R = {self.fun}\n'
            f'message = {self.message}\n'
            f'iterations = {self.iterations}\n'
            f'duration = {self.duration:.2f}s'
        )
