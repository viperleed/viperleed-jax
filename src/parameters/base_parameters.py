from abc import ABC, abstractmethod
import numpy as np

class DeltaParam():
    """
    Represents a delta parameter.

    Attributes:
        parent: The parent of the delta parameter.
        children: The children of the delta parameter.
    """

    @property
    def has_parent(self):
        """
        Check if the delta parameter has a parent.

        Returns:
            bool: True if the delta parameter has a parent, False otherwise.
        """
        return self.parent is not None

    @property
    def is_leaf(self):
        """
        Check if the delta parameter is a leaf.

        Returns:
            bool: True if the delta parameter is a leaf, False otherwise.
        """
        pass

    @property
    def has_children(self):
        """
        Check if the delta parameter has children.

        Returns:
            bool: True if the delta parameter has children, False otherwise.
        """
        return len(self.children) > 0



class BaseParam(DeltaParam):
    """
    Represents a base parameter.

    Attributes:
        atom_site_element (str): The atom site element.
        site_element (str): The site element.
        parent (object): The parent object.
        children (list): The list of child objects.
    """
    def __init__(self, atom_site_element):
        self.atom_site_element = atom_site_element
        self.site_element = atom_site_element.site_element
        self.parent = None
        self.children = None

    @property
    def min(self):
        """
        Get the minimum value from the parent object.

        Returns:
            The minimum value.
        """
        return self.parent.min()

    @property
    def max(self):
        """
        Get the maximum value from the parent object.

        Returns:
            The maximum value.
        """
        return self.parent.max()



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

    @property
    def n_base_params(self):
        return len(self.base_params)

    pass


class ConstrainedDeltaParam():
    
    def __init__(self, children):
        self.parent = None
        self.children = children
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


# Bounds

class Bound():
    def __init__(self, min, max):
        self.min = min
        self.max = max