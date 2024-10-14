from copy import deepcopy
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
        base_scatterer (str): The atom site element.
        site_element (str): The site element.
        parent (object): The parent object.
        children (list): The list of child objects.
    """
    def __init__(self, base_scatterer):
        self.base_scatterer = base_scatterer
        self.site_element = base_scatterer.site_element
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


class Params():  # This is the tree
    """
    Class representing a set of parameters.

    Attributes:
        param_type: The type of the parameters.
    """
    def __init__(self):
        # symmetry and system constraints are applied during initialization,
        # so the number of free parameters at the end of __init__ is the 
        # number of symmetry constrained parameters.
        self.n_symmetry_constrained_params = deepcopy(self.n_free_params)

    param_type = None

    @property
    def terminal_params(self):
        """
        Get a list of terminal parameters.

        Returns:
            A list of terminal parameters.
        """
        return [param for param in self.params if param.parent is None]

    @property
    def base_params(self):
        """
        Get a list of base parameters.

        Returns:
            A list of base parameters.
        """
        return [param for param in self.params if isinstance(param, BaseParam)]

    @property
    def n_free_params(self):
        """
        Get the total number of free parameters.

        Returns:
            The total number of free parameters.
        """
        return sum(param.n_free_params for param in self.terminal_params)

    @property
    def n_base_params(self):
        """
        Get the number of base parameters.

        Returns:
            The number of base parameters.
        """
        return sum(param.n_free_params for param in self.base_params)

    @property
    def free_params(self):
        """
        Returns a list of terminal parameters that have more than 0 free parameters.
        """
        return [param for param in self.terminal_params
                if param.n_free_params > 0]

    @property
    def base_to_terminal_map(self):
        """
        Get the mapping from base parameters to terminal parameters.

        Returns:
            The mapping from base parameters to terminal parameters.
        """
        base_to_terminal_map = {}
        for param in self.base_params:
            top_level = param
            while top_level not in self.terminal_params:
                top_level = top_level.parent
            base_to_terminal_map[param] = top_level
        return base_to_terminal_map


class ConstrainedDeltaParam():
    """
    A class representing a constrained delta parameter.

    Parameters:
    -----------
    children : list
        A list of child parameters.

    Attributes:
    -----------
    parent : ConstrainedDeltaParam or None
        The parent parameter.
    children : list
        The list of child parameters.
    _bound : Bound or None
        The bound for the parameter.

    Properties:
    -----------
    is_free : bool
        Returns True if the parameter is free, False otherwise.
    min : ConstrainedDeltaParam or None
        Returns the minimum bound for the parameter.
    max : ConstrainedDeltaParam or None
        Returns the maximum bound for the parameter.
    """

    def __init__(self, children):
        """
        Initializes a BaseParameters object.

        Args:
            children (list): A list of child objects.

        Attributes:
            parent: The parent object of the current instance.
            children: The list of child objects.
        """
        self.parent = None
        self.children = children
        for child in children:
            child.parent = self

    def set_bound(self, bound):
        """
        Set the bound for the parameter.

        Parameters:
        -----------
        bound : Bound
            The bound to be set.
        """
        self.bound = bound

    @property
    def is_free(self):
        """
        Returns True if the parameter is free, False otherwise.
        """
        return self._free

    @property
    def min(self):
        """
        Returns the minimum bound for the parameter.
        If the bound is not set, returns the parent parameter.
        """
        if self.bound is None:
            return self.parent
        return self.bound.min

    @property
    def max(self):
        """
        Returns the maximum bound for the parameter.
        If the bound is not set, returns the parent parameter.
        """
        if self.bound is None:
            return self.parent
        return self.bound.max


# Bounds

class Bound():
    """
    Represents a range of values with a minimum and maximum bound.

    Attributes:
        min (float): The minimum value of the range.
        max (float): The maximum value of the range.
    """
    def __init__(self, min, max):
        self.min = min
        self.max = max
