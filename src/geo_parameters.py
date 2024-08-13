from enum import Enum
from parameters import BaseParam, Params, ConstrainedDeltaParam

GeoDir = Enum('GeoDir', ('x', 'y', 'z'))


class GeoBaseParam(BaseParam):
    def __init__(self, atom_site_element):
        self.atom_site_element = atom_site_element
        self.free_dirs = [GeoDir.x, GeoDir.y, GeoDir.z]
        self.symrefm = atom_site_element.atom.symrefm
        self.layer = atom_site_element.atom.layer

class GeoParams(Params):
    
    def __init__(self, delta_slab):
        # Create base parameters for each non-bulk atom (x, y, z)
        self._base_params = [
            GeoBaseParam(atom_site_element)
            for atom_site_element in delta_slab.atom_site_elements
        ]
        # apply symmetry constraints
        self.params = []
        for siteel in delta_slab.atom_site_elements:
            site_el_params = [param for param in self._base_params
                                if param.atom_site_element == siteel]
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
        return tuple(sorted(set(param.layer for param in self._base_params)))


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
