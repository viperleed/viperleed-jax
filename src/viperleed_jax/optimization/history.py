"""Module optimization.history."""

__authors__ = (
    'Alexander M. Imre (@amimre)',
)
__created__ = '2025-03-11'

import numpy as np

class GradOptimizationHistory:
    def __init__(self):
        self._history = None

    def append(self, x, R, grad_R, timestamp):
        if self._history is None:
            self._history = np.array([x, R, grad_R, timestamp])
        else:
            self._history.append([x, R, grad_R, timestamp])

    @property
    def x_history(self):
        return self._history[:, 0]

    @property
    def R_history(self):
        return self._history[:, 1]

    @property
    def grad_R_history(self):
        return self._history[:, 2]

    @property
    def timestamp_history(self):
        return self._history[:, 3]


class EvolutionOptimizationHistory:
    def __init__(self):
        self._history = None

    def append(self, generation_x, generation_R, timestamp):
        if self._history is None:
            self._history = np.array([generation_x, generation_R, timestamp])
        else:
            self._history.append([generation_x, generation_R, timestamp])

    @property
    def x_history(self):
        return self._history[:, 0]

    @property
    def R_history(self):
        return self._history[:, 1]

    @property
    def timestamp_history(self):
        return self._history[:, 2]
