
import numpy as np
from jax import numpy as jnp

from otftleed.base import LinearTransformer
from otftleed.parameters.base_parameters import BaseParam, Params, ConstrainedDeltaParam, Bound


class VibBaseParam(BaseParam):
    """Base class for vibrational parameters.

    Parameters
    ----------
    atom_site_element : AtomSiteElement
        The atom site element.

    Attributes
    ----------
    n_free_params : int
        The number of free parameters.
    ref_vib_amp : float
        The reference vibrational amplitude.
    bound : Bound
        The bound value.
    """
    def __init__(self, atom_site_element):
        self.n_free_params = 1
        element = atom_site_element.site_element.element
        self.ref_vib_amp = atom_site_element.atom.site.vibamp[element]
        self.bound = None
        super().__init__(atom_site_element)

class VibParamBound(Bound):
    """Represents the bounds for a vibration parameter.

    Parameters
    ----------
    min : float
        The minimum value of the bound.
    max : float
        The maximum value of the bound.

    Attributes
    ----------
    min : float
        The minimum value of the bound.
    max : float
        The maximum value of the bound.
    """
    def __init__(self, min, max):
        super().__init__(min, max)

    @property
    def fixed(self):
        """Checks if the bound is fixed.

        Returns
        -------
        bool
            True if the bound is fixed, False otherwise.
        """
        return abs(self.min - self.max) < 1e-6

class VibParams(Params):
    """Class to handle vibrational parameters for a slab.

    Parameters
    ----------
    delta_slab : DeltaSlab
        The delta slab for which to handle vibrational parameters.
    """
    def __init__(self, delta_slab):
        # create a base parameter for every atom-site-element, then map them
        # to the site-elements
        self.params = [VibBaseParam(ase) for ase
                            in delta_slab.atom_site_elements]
        self.vib_bounds = ()
        # TODO: if we wanted to changes vib amps for individual atoms, we would
        # have to changes this, but as is, one can easily create a new site
        for site_el in delta_slab.site_elements:
            site_el_params = [p for p in self.base_params
                              if p.site_element == site_el]
            if not site_el_params:
                continue
            self.params.append(LinkVibParam(children=site_el_params))
        super().__init__()

    def set_bounds(self, bounds):
        """Set the bounds for each parameter.

        Parameters
        ----------
        bounds : tuple or list
            The bounds for each parameter. Each element of the tuple/list should
            be a tuple containing the lower and upper bounds for a parameter.

        Returns
        -------
        None
        """
        for param, bound in zip(self.params, bounds):
            param.set_bound(bound)

    @property
    def dynamic_site_elements(self):
        """Returns a tuple of site elements for the free parameters."""
        return tuple(param.site_element for param in self.free_params)

    @property
    def static_site_elements(self):
        """Returns a tuple of site elements for the fixed parameters."""
        return tuple(param.site_element for param in self.terminal_params
                     if param.n_free_params == 0)

    @property
    def static_t_matrix_input(self):
        # inputs (site-el and vib amp) for the calculation of the static
        # t-matrices
        inputs = []
        for param in [p for p in self.terminal_params if p.n_free_params == 0]:
            if param.bound is None:
                raise ValueError("Vibrational amplitude bounds must be set.")
            if not param.bound.fixed:
                raise ValueError("Static vibrational amplitudes must be fixed.")
            inputs.append((param.site_element, param.bound.min))
        return tuple(inputs)

    @property
    def t_matrix_map(self):
        # return a tuple with the site_elements for each base parameter

        return [('static', self.static_site_elements.index(terminal.site_element))
            if terminal.site_element in self.static_site_elements else
            ('dynamic', self.dynamic_site_elements.index(terminal.site_element))
            for base, terminal in self.base_to_terminal_map.items()]

    def set_vib_bounds(self, vib_bounds):
        """Set bounds for vibrational amplitudes.

        Parameters:
        vib_bounds (iterable): Bounds for vibrational amplitudes.
            Must have same length as terminal parameters.

        Returns:
        None
        """
        self.vib_bounds = tuple(vib_bounds)
        for bound, param in zip(vib_bounds, self.terminal_params):
            param.set_bound(bound)

    def fix_site_element(self, site_element, vib_amp):
        """Fix the vibrational amplitude for a site element.

        Parameters:
        site_element (SiteEl): Site element to fix.
        vib_amp (float): Fixed vibrational amplitude.

        Returns:
        None
        """
        vib_param = next((param for param in self.terminal_params
                          if param.site_element == site_element), None)
        if vib_param is None:
            raise ValueError(f"No vibrational parameter for site element {site_element}")
        fix_param = FixVibParam(children=[vib_param])
        if vib_amp is None:
            # fix to the reference vibrational amplitude
            vib_amp = vib_param.ref_vib_amp
        fix_param.set_bound(VibParamBound(vib_amp, vib_amp))
        self.params.append(fix_param)

    def get_vib_transformer(self):
        # return a JAX function that transforms the free parameters
        # (self.n_free_params values normalized to [0, 1]) to the vibrational
        # amplitudes for the dynamic site elements (number of t-matrices)
        # that need to be calculated on-the-fly.
        # The mapping for these t-matrices to the base parameters is given by
        # the t_matrix_map property.
        if not all(param.bound is not None for param in self.terminal_params):
            raise ValueError("Not all vibrational parameters have bounds")
        # linear transformation
        # bias is ref_vib_amps, weights are given by the bounds
        weights = np.full((len(self.free_params), self.n_free_params), 0.)
        bias = np.array([param.ref_vib_amp for param in self.free_params])

        for row_id, param in enumerate(self.free_params):
            if param.bound.fixed:
                continue
            col_id = self.free_params.index(param)
            bias[row_id] += param.bound.min
            weights[row_id, col_id] = param.bound.max - param.bound.min

        return LinearTransformer(weights, bias)


class ConstrainedVibParam(ConstrainedDeltaParam):

    def __init__(self, children):
        self.bound = None
        super().__init__(children)

    def set_bound(self, bound):
        if self.ref_vib_amp + bound.min <=0 or self.ref_vib_amp + bound.max <=0:
            raise ValueError(
                f"Bounds ({bound.min}, {bound.max}) are invalid for vibrational"
                f" amplitude {self.ref_vib_amp} (range must be positive).")
        self.bound = bound


# Constrained Vibrational Parameters

class LinkVibParam(ConstrainedVibParam):
    """Represents a linked vibrational parameter that links vibrational
    amplitude changes for children.

    Args:
        children (list): A list of child vibrational parameters.

    Attributes:
        n_free_params (int): The number of free parameters.
        site_element (str): The site element of the children parameters.
        ref_vib_amp (float): The reference vibrational amplitude of the first
            child parameter.

    Raises:
        ValueError: If not all children have the same site element.
    """
    def __init__(self, children):
        self.n_free_params = 1
        if not all([child.site_element == children[0].site_element for child in children]):
            raise ValueError("All children must have the same site element")
        self.site_element = children[0].site_element
        self.ref_vib_amp = children[0].ref_vib_amp
        super().__init__(children)

    def set_bound(self, bound):
        if bound.fixed:
            raise ValueError(
                "Linking vibrational amplitudes must have a range. Assign "
                "fixed values to a FixVibParam instead.")
        super().set_bound(bound)

class FixVibParam(ConstrainedVibParam):
    """Represents a fixed vibrational parameter.

    This class sets a fixed value for the vibrational amplitude.
    """
    def __init__(self, children):
        self.n_free_params = 0
        if not all([child.site_element == children[0].site_element for child in children]):
            raise ValueError("All children must have the same site element")
        self.site_element = children[0].site_element
        self.ref_vib_amp = children[0].ref_vib_amp
        super().__init__(children)

    def set_bound(self, bound):
        if not bound.fixed:
            raise ValueError(
                "FixVibParam must be assigned a fixed value not a range.")
        super().set_bound(bound)
