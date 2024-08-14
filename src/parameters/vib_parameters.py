
import numpy as np
from jax import numpy as jnp

from src.parameters.base_parameters import BaseParam, Params, ConstrainedDeltaParam, Bound


class VibBaseParam(BaseParam):
    
    def __init__(self, atom_site_element):
        self.n_free_params = 1
        element = atom_site_element.site_element.element
        self.ref_vib_amp = atom_site_element.atom.site.vibamp[element]
        self.bound = None
        super().__init__(atom_site_element)

class VibParamBound(Bound):
    def __init__(self, min, max):
        super().__init__(min, max)

    @property
    def fixed(self):
        return abs(self.min - self.max) < 1e-6

class VibParams(Params):
    """
    Class to handle vibrational parameters for a slab.
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

    def set_bounds(self, bounds):
        for param, bound in zip(self.params, bounds):
            param.set_bound(bound)

    @property
    def free_params(self):
        return [param for param in self.terminal_params
                if param.n_free_params > 0]

    @property
    def dynamic_site_elements(self):
        return tuple(param.site_element for param in self.free_params)

    @property
    def static_site_elements(self):
        return tuple(param.site_element for param in self.terminal_params
                     if param.n_free_params == 0)

    @property
    def t_matrix_map(self):
        # return a tuple with the site_elements for each base parameter
        return tuple(param.site_element for param in self.base_params)

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
        weights = np.full((len(self.terminal_params), self.n_free_params), 0.)
        bias = np.array([param.ref_vib_amp for param in self.terminal_params])

        for row_id, param in enumerate(self.terminal_params):
            if param.bound.fixed:
                continue
            col_id = self.free_params.index(param)
            weights[row_id, col_id] = param.bound.max - param.bound.min

        def transformer(params):
            params = jnp.array(params)
            if params.shape != (self.n_free_params,):
                raise ValueError("Invalid number of free parameters")
            return weights@params + bias
        return transformer


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
    # links vibrational amplitude changes for children
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
    # sets a fixed value for the vibrational amplitude
    def __init__(self, children):
        self.n_free_params = 0
        super().__init__(children)

    def set_bound(self, bound):
        if not bound.fixed:
            raise ValueError(
                "FixVibParam must be assigned a fixed value not a range.")
        super().set_bound(bound)