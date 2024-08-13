from src.parameters.base_parameters import BaseParam, Params, ConstrainedDeltaParam

from jax import numpy as jnp

class ChemBaseParam(BaseParam):
    def __init__(self, atom_site_element):
        self.atom = atom_site_element.atom
        self.n_free_params = 1
        self.elements = {atom_site_element.site_element.element:None,
                         'vac':None}
        super().__init__(atom_site_element)


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

    def get_weight_transformer(self):
        """Return a JAX function that transforms the free parameters
        (self.n_free_params values normalized to [0, 1]) to the chemical weights
        for every atom-site-element. If no free parameters are present, returns
        a function that returns the fixed chemical weights.
        """
        # parameters that require a transformation
        free_top_level_params = [param for param in self.terminal_params
                                 if param.n_free_params > 0]
        split_sections = jnp.cumsum(jnp.array([param.n_free_params
                                     for param in free_top_level_params]))[:-1]

        # maping of top-level parameters to atom-site-elements
        static_weights = jnp.full(len(self.base_params), 0.0)
        dynamic_indices = {}

        weights = []
        for i, param in enumerate(self.base_params):
            element = param.site_element.element
            top_level = param
            while top_level.parent is not None:
                if element in top_level.parent.elements.keys():
                    top_level = top_level.parent
            if top_level.elements[element] is None:
                dynamic_indices[i] = (free_top_level_params.index(top_level),
                                      list(top_level.elements.keys()).index(element))
            else:
                static_weights = static_weights.at[i].set(top_level.elements[element])


        def transformer(free_params):
            if len(dynamic_indices) == 0:
                return static_weights
            splits = jnp.split(free_params, split_sections)
            distribute_weights(splits[0])
            split_weights = [distribute_weights(split) for split in splits]

            weights = static_weights
            for weight_id, (split_id, split_n) in dynamic_indices.items():
                weights = weights.at[weight_id].set(split_weights[split_id][split_n])
            return weights
        return transformer


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
        # check if all but one element are fixed
        free_elements = [el for el, val in self.elements.items() if val is None]
        if len(free_elements) == 1:
            # value of last free element = 1 - sum(fixed_elements)
            self.elements[free_elements[0]] = 1 - sum([el for el in self.elements.values() if el is not None])
        super().__init__([child])

def distribute_weights(pi):
    # calculate weight as per wi = pi * min(1, 1/sum(pi))
    return pi * jnp.minimum(1, 1/jnp.sum(pi))