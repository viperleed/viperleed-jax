import numpy as np

class DeltaParam():

    @property
    def has_parent(self):
        return self.parent is not None

    @property
    def is_leaf(self):
        pass

    @property
    def has_children(self):
        return len(self.children) > 0



class BaseParam(DeltaParam):
    # every base parameter is a leaf and has to have a parent
    def __init__(self, atom_site_element):
        self.parent = None
        self.children = None

    @property
    def min(self):
        return self.parent.min()

    @property
    def max(self):
        return self.parent.max()


from abc import ABC, abstractmethod
class Params(ABC):
    param_type = None

    @property
    def terminal_params(self):
        return [param for param in self.params if param.parent is None]

    @property
    def base_params(self):
        return [param for param in self.params if isinstance(param, BaseParam)]

    @property
    def n_free_params(self):
        return sum(param.n_free_params for param in self.terminal_params)

    pass


class ConstrainedDeltaParam():
    
    def __init__(self, children):
        self.parent = None
        for child in children:
            child.parent = self

    def set_bound(self, bound):
        self._bound = bound

    @property
    def is_free(self):
        return self._free

    @property
    def min(self):
        if self._bound is None:
            return self.parent
        return self._bound.min

    @property
    def max(self):
        if self._bound is None:
            return self.parent
        return self._bound.max

class ConstrainedVibParam(ConstrainedDeltaParam):
    
    def __init__(self, children):
        super().__init__(children)

# Constrained Vibrational Parameters

class LinkVibParam(ConstrainedVibParam):
    # links vibrational amplitude changes for children
    def __init__(self, children):
        self.n_free_params = 1
        self._free = True
        if not all([child.site_element == children[0].site_element for child in children]):
            raise ValueError("All children must have the same site element")
        self.site_element = children[0].site_element
        super().__init__(children)


# Bounds

class Bound():
    def __init__(self, min, max):
        self.min = min
        self.max = max