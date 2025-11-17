"""Module optimization.history."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-03-11'

import time
from abc import ABC, abstractmethod

import numpy as np


class OptimizationHistory(ABC):
    def __init__(self):
        self._history = []
        self._start_time = time.time()
        self._v0r_offsets = []
        self._geo_displacements = []
        self._vibration_amplitudes = []
        self._occupations = []

    @abstractmethod
    def append(self, *args):
        pass

    @abstractmethod
    def timestamp_history(self):
        pass

    @property
    def duration(self):
        """Return the duration of the optimization process."""
        return self.timestamp_history[-1] - self._start_time

    @property
    def v0r_offsets(self):
        return np.array(self._v0r_offsets)

    @property
    def geo_displacements(self):
        return self._geo_displacements

    @property
    def vibration_amplitudes(self):
        return self._vibration_amplitudes

    @property
    def occupations(self):
        return self._occupations

    @abstractmethod
    def expand_parameters(self, calculator):
        """Expand the parameter history using the provided calculator."""


class GradOptimizationHistory(OptimizationHistory):
    def __init__(self):
        super().__init__()

    def append(self, x, R, grad_R):
        timestamp = time.time()
        if R is None:
            R = np.nan
        if grad_R is None:
            grad_R = np.full_like(x, np.nan)
        self._history.append([x, R, grad_R, timestamp])

    @property
    def x_history(self):
        return np.array([h[0] for h in self._history])

    @property
    def R_history(self):
        return np.array([h[1] for h in self._history])

    @property
    def grad_R_history(self):
        return np.array([h[2] for h in self._history])

    @property
    def timestamp_history(self):
        return np.array([h[3] for h in self._history])

    def expand_parameters(self, calculator):
        """Expand the parameter history using the provided calculator."""
        self._v0r_offsets = []
        self._geo_displacements = []
        self._vibration_amplitudes = []
        self._occupations = []
        n_evals, _ = self.x_history.shape
        for eval in range(n_evals):
            x = self.x_history[eval]
            v0r, geo, vib, occ = calculator.expand_params(x)
            self._v0r_offsets.append(v0r)
            self._geo_displacements.append(geo)
            self._vibration_amplitudes.append(vib)
            self._occupations.append(occ)


class EvolutionOptimizationHistory(OptimizationHistory):
    def __init__(self):
        super().__init__()

    def append(self, generation_x, generation_R, step_size):
        timestamp = time.time()
        self._history.append([generation_x, generation_R, step_size, timestamp])

    @property
    def x_history(self):
        return np.array([h[0] for h in self._history])

    @property
    def R_history(self):
        return np.array([h[1] for h in self._history])

    @property
    def step_size_history(self):
        return np.array([h[2] for h in self._history])

    @property
    def timestamp_history(self):
        return np.array([h[3] for h in self._history])

    def expand_parameters(self, calculator):
        """Expand the parameter history using the provided calculator."""
        self._v0r_offsets = []
        self._geo_displacements = []
        self._vibration_amplitudes = []
        self._occupations = []
        n_evals, n_gens, _ = self.x_history.shape
        for gen in range(n_gens):
            for eval in range(n_evals):
                x = self.x_history[eval, gen]
                v0r, geo, vib, occ = calculator.expand_params(x)
                self._v0r_offsets.append(v0r)
                self._geo_displacements.append(geo)
                self._vibration_amplitudes.append(vib)
                self._occupations.append(occ)
