"""Helper functions and constants for bounded simplex tests."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-09-17'

import numpy as np

BOUNDS = [  # format: (lower, upper)
    # 2D cases
    (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
    (np.array([0.2, 0.3]), np.array([0.7, 0.8])),
    (np.array([0.0, 0.5]), np.array([0.5, 1.0])),
    (np.array([0.4, 0.4]), np.array([0.6, 0.6])),  # tight box
    # 3D cases
    (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
    (np.array([0.1, 0.1, 0.1]), np.array([0.7, 0.7, 0.7])),
    (np.array([0.2, 0.3, 0.1]), np.array([0.5, 0.6, 0.4])),
    (np.array([0.0, 0.0, 0.5]), np.array([0.5, 0.5, 0.5])),  # one fixed dim
    (np.array([0.3, 0.3, 0.3]), np.array([0.4, 0.4, 0.4])),  # tight box
    # 5D cases
    (np.array([0.1, 0.1, 0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5, 0.5, 0.5])),
    (
        np.array([0.2, 0.1, 0.1, 0.1, 0.1]),
        np.array([0.4, 0.4, 0.4, 0.1, 0.1]),  # two dims fixed
    ),
]
SAMPLING_ITERATIONS = 10000
SAMPLING_TOLERANCE = 1e-3


def uniform_vector(low, high, rng):
    """Draw a vector uniformly within [low_i, high_i] componentwise."""
    return rng.uniform(low=low, high=high).astype(np.float64)
