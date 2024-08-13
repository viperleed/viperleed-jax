from src.parameters.base_parameters import BaseParam, DeltaParam, Params, ConstrainedDeltaParam


class GeoBaseParam(BaseParam):
    def __init__(self, atom_site_element):
        self.atom_site_element = atom_site_element
        self.site_element = atom_site_element.site_element
        self.n_free_params = 3 # x, y, z
        self.symrefm = atom_site_element.atom.symrefm
        self.layer = atom_site_element.atom.layer.num

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

    @property
    def n_free_params(self):
        return sum(param.n_free_params for param in self.params)

    @property
    def layers(self):
        return tuple(sorted(set(param.layer for param in self.base_params)))

    def constrain_layer(self, layer):
        layer_params = [param for param in self.terminal_params if param.layer == layer]
        if not layer_params:
            raise ValueError(f"No free params for layer {layer}")
        new_constraint = GeoLayerConstraint(children=layer_params)
        self.params.append(new_constraint)


class GeoConstraint(ConstrainedDeltaParam):
    def __init__(self, children):
        self.symmetry_operations = {
            child: child.symrefm
            for child in children
        }
        # all children must be in the same layer
        if not all(child.layer == children[0].layer for child in children):
            raise ValueError("All children must be in the same layer")
        self.layer = children[0].layer
        super().__init__(children)
        

class GeoSymmetryConstraint(GeoConstraint):
    # constrains multiple atoms (x, y, z) to move together (x, y, z)
    # in a symmetry linked way
    # For n linked atom-elements, the number of free parameters is reduced
    # from 3*n to 3.
    def __init__(self, children):
        self.free_params = 3
        super().__init__(children)

class GeoLayerConstraint(GeoConstraint):
    # constrain multiple atoms in the same layer (x, y, z) to move together
    # (z only, since in-plane movement would break symmetries)
    def __init__(self, children):
        self.free_params = 1
        super().__init__(children)


class GeoFixConstraint(GeoConstraint):
    pass

def geo_sym_linking(atom):
    linking = np.zeros(shape=(3,3))
    linking[1:3, 1:3] = atom.symrefm  #TODO: round off the 1e-16 contributions
    linking[0,0] = 1.0 # all symmetry linked atoms move together is z directon
    return linking#

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

    @property
    def layers(self):
        return tuple(sorted(set(param.layer for param in self.base_params)))

    def constrain_layer(self, layer):
        layer_params = [param for param in self.terminal_params if param.layer == layer]
        if not layer_params:
            raise ValueError(f"No free params for layer {layer}")
        new_constraint = GeoLayerConstraint(children=layer_params)
        self.params.append(new_constraint)


class GeoConstraint(ConstrainedDeltaParam):
    def __init__(self, children):
        # all children must be in the same layer
        if not all(child.layer == children[0].layer for child in children):
            raise ValueError("All children must be in the same layer")
        self.layer = children[0].layer
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


class GeoFixConstraint(GeoConstraint):
    pass

def geo_sym_linking(atom):
    linking = np.zeros(shape=(3,3))
    linking[1:3, 1:3] = atom.symrefm  #TODO: round off the 1e-16 contributions
    linking[0,0] = 1.0 # all symmetry linked atoms move together is z directon
    return linking#
