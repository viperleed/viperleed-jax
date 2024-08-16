from src.parameters.base_parameters import BaseParam, DeltaParam, Params, ConstrainedDeltaParam, Bound

import jax
from jax import numpy as jnp
import numpy as np

class GeoBaseParam(BaseParam):
    def __init__(self, atom_site_element):
        self.n_free_params = 3 # x, y, z
        self.symrefm = atom_site_element.atom.symrefm
        self.layer = atom_site_element.atom.layer.num
        super().__init__(atom_site_element)

# Isotropic geometric bound; could be extended to anisotropic
class GeoParamBound(Bound):
    def __init__(self, min, max):
        super().__init__(min, max)

    @property
    def fixed(self):
        return abs(self.min - self.max) < 1e-6

class GeoParams(Params):
    def __init__(self, delta_slab):
        # Create base parameters for each non-bulk atom (x, y, z)
        self.params = [
            GeoBaseParam(atom_site_element)
            for atom_site_element in delta_slab.atom_site_elements
        ]
        # apply symmetry constraints
        for siteel in delta_slab.site_elements:
            site_el_params = [param for param in self.base_params
                                if param.site_element == siteel]
            for linklist in delta_slab.slab.linklists:
                ref_atom = linklist[0]
                params_to_link = [param for param in site_el_params
                              if param.atom_site_element.atom in linklist]
                if params_to_link:
                    self.params.append(GeoSymmetryConstraint(
                        children=params_to_link))
        super().__init__()

    @property
    def layers(self):
        return tuple(sorted(set(param.layer for param in self.base_params)))

    def constrain_layer(self, layer):
        layer_params = [param for param in self.terminal_params if param.layer == layer]
        if not layer_params:
            raise ValueError(f"No free params for layer {layer}")
        new_constraint = GeoLayerConstraint(children=layer_params)
        self.params.append(new_constraint)

    def fix_layer(self, layer, z_offset=None):
        layer_params = [param for param in self.terminal_params if param.layer == layer]
        if not layer_params:
            raise ValueError(f"No free params for layer {layer}")
        z_constraint = GeoLayerConstraint(children=layer_params)
        fix_constraint = GeoFixConstraint(children=[z_constraint])
        if z_offset is not None:
            fix_bound = GeoParamBound(z_offset, z_offset)
        fix_constraint.set_bound(fix_bound)
        self.params.append(z_constraint)
        self.params.append(fix_constraint)

    def set_geo_bounds(self, geo_bounds):
        self.geo_bounds = tuple(geo_bounds)
        for bound, param in zip(geo_bounds, self.terminal_params):
            param.set_bound(bound)

    @property
    def dynamic_propagators(self):
        return [param for param in self.free_params]

    @property
    def static_propagators(self):
        return [param for param in self.terminal_params
                if param not in self.free_params]

    @property
    def n_dynamic_propagators(self):
        return len(self.dynamic_propagators)

    @property
    def propagator_map(self):
        # map proagators to atom-site-elements

        return tuple(
            self.terminal_params.index(terminal)
            for base, terminal in self.base_to_terminal_map.items()
        )


    def get_geo_transformer(self):
        """Return a JAX function that transforms the free parameters
        (self.n_free_params values normalized to [0, 1]) to the displacements
        for the dynamic propagators ((3, self.n_dynamic_propagators) values).
        """
        if not all(param.bound is not None for param in self.terminal_params):
            raise ValueError("Not all geometric parameters have bounds.")
        # linear transformation
        # offset is 0 (# TODO: could implement that)
        # weights are given by the bounds and the constraint method
    
        weights = [param.free_param_map * (param.bound.max - param.bound.min)
                   for param in self.dynamic_propagators]
        weights = jax.scipy.linalg.block_diag(*weights)
        assert weights.shape == (3*self.n_dynamic_propagators, self.n_free_params)

        def transformer(free_params):
            if len(free_params) != self.n_free_params:
                raise ValueError("Free parameters have wrong shape")
            return (weights @ jnp.asarray(free_params)).reshape(-1, 3)
        return transformer


class GeoConstraint(ConstrainedDeltaParam):
    def __init__(self, children):
        # all children must be in the same layer
        if not all(child.layer == children[0].layer for child in children):
            raise ValueError("All children must be in the same layer")
        self.layer = children[0].layer
        self.bound = None
        super().__init__(children)


class GeoSymmetryConstraint(GeoConstraint):
    # constrains multiple atoms (x, y, z) to move together (x, y, z)
    # in a symmetry linked way
    # For n linked atom-elements, the number of free parameters is reduced
    # from 3*n to 3.
    def __init__(self, children):
        self.n_free_params = 3
        self.symmetry_operations = {
            child: child.symrefm
            for child in children
        }
        super().__init__(children)

    @property
    def free_param_map(self):
        # three free parameters are mapped to the x, y, z directions
        return np.array(([1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]))

class GeoLayerConstraint(GeoConstraint):
    # constrain multiple atoms in the same layer (x, y, z) to move together
    # (z only, since in-plane movement would break symmetries)
    def __init__(self, children):
        self.n_free_params = 1
        self.symmetry_operations = {
            child: np.array([[1., 0.],
                             [0., 1.]])
            for child in children
        }
        super().__init__(children)

    @property
    def free_param_map(self):
        # one free parameter is mapped to the z direction
        return np.array([0., 0., 1.]).T


class GeoFixConstraint(GeoConstraint):
    # constrain children to fixed values
    def __init__(self, children):
        self.n_free_params = 0
        self.symmetry_operations = {
            child: np.array([[1., 0.],
                             [0., 1.]])
            for child in children
        }
        super().__init__(children)


def geo_sym_linking(atom):
    linking = np.zeros(shape=(3,3))
    linking[1:3, 1:3] = atom.symrefm  #TODO: round off the 1e-16 contributions
    linking[0,0] = 1.0 # all symmetry linked atoms move together is z directon
    return linking
