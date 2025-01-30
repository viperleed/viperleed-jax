"""Module displacment_range."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-21'

import numpy as np


class DisplacementRange:
    """Class representing a bound in the displacements tree.

    Used to represent the lower and upper bounds of values that can be taken on
    by the parameters represented by nodes. Bounds are assigned to leaf nodes in
    the tree. Can be propagated up and down the tree.
    """

    _EPS = 1e-6

    def __init__(self, dimension):
        self.dimension = dimension
        self._enforce_range = np.full(shape=(self.dimension,), fill_value=False)
        self._enforce_offset = np.full(
            shape=(self.dimension,), fill_value=False)
        self._lower, self._upper = np.zeros(dimension), np.zeros(dimension)
        self._offset = np.zeros(dimension)
        self.update_offset(offset=np.zeros(dimension))
        self.update_range(_range=(np.zeros(dimension), np.zeros(dimension)))

    @property
    def lower(self):
        """Return the lower bound of the displacement range."""
        return self._lower + self._offset

    @property
    def upper(self):
        """Return the upper bound of the displacement range."""
        return self._upper + self._offset

    @property
    def offset(self):  # TODO: are offsets treated correctly at the moment?
        """Return the offset of the displacement range."""
        return self._offset

    @property
    def fixed(self):
        """Check if the displacement range is fixed."""
        return abs(self.upper - self.lower) < self._EPS

    @property
    def enforce(self):
        """Return the enforce array indicating which bounds are enforced."""
        return np.logical_or(self._enforce_range, self._enforce_offset)

    def update_offset(self, offset, enforce=None):
        """Update the offset of the displacement range."""
        if offset is None:
            raise ValueError('Offset must be set.')
        if enforce is None:
            enforce = np.full(self.dimension, fill_value=False)
        offset_different = np.abs(self.offset - offset) > self._EPS
        if any(np.logical_and(self.enforce, enforce, offset_different)):
            raise ValueError('Cannot change enforced offset.')

        self._offset = np.asarray(offset).reshape(self.dimension)
        self._enforce_offset = np.logical_or(self._enforce_offset, enforce)

    def update_range(self, _range, enforce=None):
        if _range is None:
            raise ValueError('Range must be set.')
        if enforce is None:
            enforce = np.full(self.dimension, fill_value=False)
        elif isinstance(enforce, bool):
            enforce = np.full(self.dimension, enforce, dtype=bool)

        if _range is not None:
            lower, upper = _range
            lower = np.asarray(lower).reshape(self.dimension)
            upper = np.asarray(upper).reshape(self.dimension)
            lower_changed = np.abs(self.lower - self.lower) > self._EPS
            if any(np.logical_and(lower_changed, self._enforce_range, enforce)):
                raise ValueError('Cannot change enforced lower bound.')
            upper_changed = np.abs(self.upper - self.upper) > self._EPS
            if any(np.logical_and(upper_changed, self._enforce_range, enforce)):
                raise ValueError('Cannot change enforced upper bound.')
            self._lower = np.where(self._enforce_range, self._lower, lower)
            self._upper = np.where(self._enforce_range, self._upper, upper)
            # use logical_or to combine the user set flags
            self._enforce_range = np.logical_or(self._enforce_range, enforce)

    def __repr__(self):
        """Return a string representation of the DisplacementRange object."""
        return f'DisplacementRange(lower={self.lower}, upper={self.upper})'
