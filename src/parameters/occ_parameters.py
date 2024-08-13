from src.parameters.base_parameters import BaseParam, Params, ConstrainedDeltaParam

class ChemBaseParam(BaseParam):
    def __init__(self, atom_site_element):
        self.atom_site_element = atom_site_element
        self.site_element = atom_site_element.site_element
        self.atom = atom_site_element.atom
        self.n_free_params = 1
        self.elements = {atom_site_element.site_element.element:None,
                         'vac':None}


class ChemParams(Params):

    def __init__(self, delta_slab):
        # initially, every atom-site-element has a free chemical weight
        # to allow for (partial) vacancies
        self.params = [ChemBaseParam(ase) for ase in delta_slab.atom_site_elements]

        # iterate over atom-site-elements and link ones from the same atom
        # since we can't have more than 100% occupancy
        nums_processed = []
        linked_params = []
        for atom in delta_slab.non_bulk_atoms:
            atom_params = [param for param in self.base_params
                           if param.atom_site_element.atom == atom]
            if not atom_params:
                continue
            linked_param = SharedOccChemConstraint(children=atom_params)
            self.params.append(linked_param)

    def remove_remaining_vacancies(self):
        # constrain all remaining free parameters, such that no vacancies are
        # left
        for param in self.terminal_params:
            if param.elements['vac'] is None:
                fixed_param = FixedOccChemConstraint(param, {'vac':0.0})
                self.params.append(fixed_param)



class ChemConstraint(ConstrainedDeltaParam):
    def __init__(self, children):
        # all children must be of the same site
        if not all(child.site_element.site == children[0].site_element.site
                   for child in children):
            raise ValueError("All children must be of the same site type.")
        self.site_element = children[0].site_element
        super().__init__(children)

    @property
    def n_free_params(self):
        return len([el for el in self.elements.values() if el is None]) - 1

class SharedOccChemConstraint(ChemConstraint):
    # link parameters of the same atom to ensure that the sum of their
    # weights is 1.0
    def __init__(self, children):
        self.elements = {}
        for child in children:
            self.elements = {**child.elements | self.elements}
        # all children must be of the same atom
        if not all(child.atom == children[0].atom for child in children):
            raise ValueError("All children must be of the same atom.")
        super().__init__(children)

class LinkedOccChemConstraint(ChemConstraint):
    # link chemical parameters
    def __init__(self, children):
        # all children must have the same elements
        if not all(child.elements == children[0].elements
                   for child in children):
            raise ValueError("All children must have the same elements.")
        self.elements = children[0].elements
        super().__init__(children)

class FixedOccChemConstraint(ChemConstraint):
    def __init__(self, child, fixed_values:dict):
        if not isinstance(fixed_values, dict):
            raise ValueError("Fixed values must be a dictionary.")
        # can only take one child
        if not isinstance(child, (ChemBaseParam, ChemConstraint)):
            raise ValueError("FixedOccChemConstraint can only take one child.")
        self.elements = child.elements
        for element, value in fixed_values.items():
            if element not in self.elements.keys():
                raise ValueError(f"Element {element} not in elements.")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Value {value} not in [0.0, 1.0].")
            self.elements[element] = value
        super().__init__([child])
