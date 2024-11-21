import numpy as np


class DisplacementRange:
    """Class representing a bound in the hierarchical linear tree.

    Bounds are used to represent the lower and upper bounds of values that can
    be taken by the parameters represented by nodes. Bounds are assigned to leaf
    nodes in the tree.
    Bounds can be propagated up and down the tree.
    """

    _EPS = 1e-6

    def __init__(self, dimension):
        self.dimension = dimension
        self._enforce = np.full(shape=(self.dimension,), fill_value=False)
        self._lower, self._upper = np.zeros(dimension), np.zeros(dimension)
        self._offset = np.zeros(dimension)
        self.update_range(
            _range=(np.zeros(dimension), np.zeros(dimension)),
            offset=np.zeros(dimension),
        )

    @property
    def lower(self):
        return self._lower + self._offset

    @property
    def upper(self):
        return self._upper + self._offset

    @property
    def offset(self):
        return self._offset

    @property
    def fixed(self):
        return abs(self.upper - self.lower) < self._EPS

    @property
    def enforce(self):
        return self._enforce

    def update_range(self, _range=None, offset=None, enforce=None):
        if _range is None and offset is None:
            raise ValueError('range or offset must be provided')
        if enforce is None:
            enforce = np.full(self.dimension, False)
        elif isinstance(enforce, bool):
            enforce = np.full(self.dimension, enforce, dtype=bool)
        if offset is not None:
            _offset = np.asarray(offset).reshape(self.dimension)
        else:  # offset is None
            _offset = np.zeros(self.dimension)
        if _range is not None:
            lower, upper = _range
            lower = np.asarray(lower).reshape(self.dimension) + _offset
            upper = np.asarray(upper).reshape(self.dimension) + _offset
            for idx in range(self.dimension):
                if (
                    abs(self.lower[idx] - lower[idx]) > self._EPS
                    and self.enforce[idx]
                ):
                    raise ValueError('Cannot change enforced lower bound.')
                if (
                    abs(self.upper[idx] - upper[idx]) > self._EPS
                    and self.enforce[idx]
                ):
                    raise ValueError('Cannot change enforced upper bound.')
                self._lower = lower
                self._upper = upper
                self.enforce[idx] = np.logical_or(
                    self.enforce[idx], enforce[idx]
                )

            self._lower = lower
            self._upper = upper
        if offset is not None:
            self._offset = np.asarray(offset).reshape(self.dimension)

        # mark the bounds that were user set;
        # use logical_or to combine the user set flags
        if enforce is None:
            enforce = np.full(self.dimension, False)
        _enforce = np.asarray(enforce).reshape(self.dimension)
        self._enforce = np.logical_or(self.enforce, _enforce)

    def __repr__(self):
        return f'HLBound(lower={self.lower}, upper={self.upper})'
