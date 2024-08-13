class VibParam(DeltaParam):
    
    def __init__(self, atom_site_element):
        self.atom_site_element = atom_site_element
        self.site_element = atom_site_element.site_element
        pass



class VibBaseParam(VibParam, BaseParam):
    
    def __init__(self, atom_site_element):
        self.n_free_params = 1
        super().__init__(atom_site_element)
    pass

class VibParams(Params):
    """
    Class to handle vibrational parameters for a slab.
    """
    param_type = VibParam
    def __init__(self, delta_slab):
        # create a base parameter for every atom-site-element, then map them
        # to the site-elements
        self.params = [VibParam(ase) for ase
                            in delta_slab.atom_site_elements]
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
    def dynamic_site_elements(self):
        return tuple(param.site_element for param in self.terminal_params
                     if param.is_free)

    @property
    def static_site_elements(self):
        return tuple(param.site_element for param in self.terminals_params
                     if not param.is_free)

    def t_matrix_map(self):
        
        pass

    @property
    def n_free_params(self)->int:
        return sum([param.is_free for param in self.terminal_params])

    def linear_transform(self):
        # return a linear transformation matrix that maps the normalized inputs
        # (number of free parameters) to the vibrational amplitudes for the
        # dynamic site elements
        mins = [param.min for param in self.terminal_params]
        maxs = [param.max for param in self.terminal_params]
        return np.diag(mins), np.diag(maxs)

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



class FixVibParam(ConstrainedVibParam):
    # sets a fixed value for the vibrational amplitude
    def __init__(self, children):
        self._free = False
        self.n_free_params = 0
        super().__init__(children)
    
