"""Module optimization.result."""

__authors__ = (
    'Alexander M. Imre (@amimre)',
    'Paul Haidegger (@PaulHai7)',
)
__created__ = '2025-03-11'

from abc import ABC, abstractmethod

import numpy as np


class OptimizationResult(ABC):
    @abstractmethod
    def best_R(self):
        return self.best

    @abstractmethod
    def best_x(self):
        return self.min_individual

    @property
    def duration(self):
        return self.history.duration

    @abstractmethod
    def __repr__(self):
        """Return a string representation of the optimization result."""

    @abstractmethod
    def write_to_file(self, file_path):
        """Write the optimization result to a file."""


class CMAESResult(OptimizationResult):
    """Class for the output of the CMA-ES algorithm.

    Parameters
    ----------

    """

    def __init__(
        self,
        evolution_history,
        message,
        cholesky,
        convergence_generations,
    ):
        self.history = evolution_history
        self.message = message
        self.convergence_generations = convergence_generations
        self.cholesky = cholesky

    @property
    def _truncated_R_history(self):
        return self.history.R_history[-self.convergence_generations:]

    @property
    def _truncated_x_history(self):
        return self.history.x_history[-self.convergence_generations:]

    @property
    def best_R(self):
        return np.min(self._truncated_R_history)

    @property
    def best_x(self):
        min_idx = np.unravel_index(np.argmin(self._truncated_R_history),
                                   self._truncated_R_history.shape)
        return self._truncated_x_history[min_idx]

    @property
    def n_generations(self):
        return len(self.history.R_history)

    def __repr__(self):
        """Return a string representation of the optimization result."""
        return (
            f'Best R:\t\t{self.best_R:.4f}\n'
            f'Message:\t{self.message}\n'
            f'Generations:\t{self.n_generations}\n'
            f'Duration:\t{self.duration:.2f}s'
        )

    def write_to_file(self, file_path):
        """Write the optimization result to a file."""
        np.savez(
            file_path,
            x_history=self.history.x_history,
            R_history=self.history.R_history,
            step_size_history=self.history.step_size_history,
            timestamp_history=self.history.timestamp_history,
        )


class GradOptimizerResult(OptimizationResult):
    def __init__(self, scipy_result, history):
        self.iterations = scipy_result.nit
        self.message = scipy_result.message
        self.history = history

    @property
    def best_R(self):
        # get last non-nan value
        non_nan_R = self.history.R_history[~np.isnan(self.history.R_history)]
        return non_nan_R[-1]

    @property
    def best_x(self):
        return self.history.x_history[-1]

    def __repr__(self):
        return (
            f'Best R:\t\t{self.best_R:.4f}\n'
            f'Message:\t{self.message}\n'
            f'Iterations:\t{self.iterations}\n'
            f'Duration:\t{self.duration:.2f}s'
        )

    def write_to_file(self, file_path):
        """Write the optimization result to a file."""
        np.savez(
            file_path,
            x_history=self.history.x_history,
            R_history=self.history.R_history,
            grad_R_history=self.history.grad_R_history,
            timestamp_history=self.history.timestamp_history,
        )
