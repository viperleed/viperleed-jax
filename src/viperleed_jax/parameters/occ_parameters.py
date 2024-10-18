from functools import partial

from viperleed_jax.parameters.base_parameters import BaseParam, Params, ConstrainedDeltaParam
from viperleed_jax.base import LinearTransformer

import numpy as np
import jax
from jax import numpy as jnp

from .linear_transformer import LinearTransformer
from .hierarchical_linear_tree import HLLeafNode, HLConstraintNode
from .hierarchical_linear_tree import ParameterHLSubtree


class OccHLLeafNode(HLLeafNode):
    """Represents a leaf node with occupational parameters."""

    def __init__(self, base_scatterer):
        dof = 1
        self.element = base_scatterer.site_element.element
        self.site = base_scatterer.site_element.site
        self.num = base_scatterer.num
        self.site_element = base_scatterer.site_element
        self.base_scatterer = base_scatterer
        self.ref_vib_amp = base_scatterer.atom.site.vibamp[self.element]
        self.name = f"occ (At_{self.num},{self.site},{self.element})"
        super().__init__(dof=dof, name=self.name)

    def update_bounds(self, line):
        # occupational leaves are 1D, so bounds are scalars
        range = line.range
        self._bounds.update_range((range.start, range.stop), user_set=True)


class OccHLConstraintNode(HLConstraintNode):
    """Represents a constraint node for occupational parameters."""

    def __init__(self, dof, children, name, transformers=None):
        self.dof = dof

        if transformers is None:
            raise ValueError("Transformers must be provided.")
        super().__init__(dof=dof, name=name, children=children, transformers=transformers)

class OccSharedHLConstraint(OccHLConstraintNode):
    """Constraint for sharing occupation to 100%."""

    def __init__(self, children):
        name = "shared occ"
        dof = len(children)

        if any(not isinstance(child, OccHLLeafNode) for child in children):
            raise ValueError("Children must be OccHLLeaf nodes.")

        if any(child.num != children[0].num for child in children):
            raise ValueError("Children must be of the same atom.")
        # set the number of the atom
        self.num = children[0].num

        transformers = []
        for child in children:
            weights = np.full(shape=(1, dof), fill_value=-1/dof)
            weights[0, children.index(child)] = 1
            bias = np.zeros(1)
            transformers.append(LinearTransformer(weights, bias, (1,)))
        super().__init__(dof=dof, name=name, children=children, transformers=transformers)

class OccSymmetryHLConstraint(OccHLConstraintNode):
    """Constraint for enforcing symmetry in occupation."""

    def __init__(self, children, name):

        # Check that all children have the same dof
        if len(set(child.dof for child in children)) != 1:
            raise ValueError("Children must have the same dof.")

        dof = children[0].dof

        transformers = []
        for child in children:
            weights = np.identity(dof)
            bias = np.zeros(dof)
            transformers.append(LinearTransformer(weights, bias, (dof,)))
        super().__init__(dof=dof, name=name,
                         children=children, transformers=transformers)


class OccLinkedHLConstraint(OccHLConstraintNode):
    """Class for explicit links of occupational parameters."""

    def __init__(self, children, name):
        # check that all children have the same dof
        if len(set(child.dof for child in children)) != 1:
            raise ValueError("Children must have the same dof.")
        dof = children[0].dof

        # transformers can be identity
        transformers = [
            LinearTransformer(np.eye(dof), np.zeros(dof), (dof,))
            for _ in children
        ]
        super().__init__(
            dof=dof,
            children=children,
            transformers=transformers,
            name=f'CONSTRAIN "{name}"',
        )


class OccHLSubtree(ParameterHLSubtree):
    def __init__(self, base_scatterers):
        super().__init__(base_scatterers)

    @property
    def name(self):
        return "Occupational Parameters"

    @property
    def subtree_root_name(self):
        return "occ root"

    def build_subtree(self):
        # initially, every atom-site-element has a free chemical weight
        # to allow for (partial) vacancies
        occ_leaf_nodes = [OccHLLeafNode(ase) for ase in self.base_scatterers]
        self.nodes.extend(occ_leaf_nodes)

        # iterate over atom-site-elements and link ones from the same atom
        # since we can't have more than 100% occupancy
        # This does not reduce the number of free parameters, but it's a physical
        # requirement that we need to enforce
        linked_nodes = []
        for num in range(self.base_scatterers.max_atom_number+1):  # inclusive range
            atom_nodes = [node for node in self.leaves
                        if node.num == num]
            if not atom_nodes:
                continue
            linked_node = OccSharedHLConstraint(children=atom_nodes)
            self.nodes.append(linked_node)
            linked_nodes.append(linked_node)


        # occupational parameters need to fulfill symmetry constraints
        for link in self.base_scatterers.atom_number_symmetry_links:
            # put all linked atoms in the same symmetry group
            
            nodes_to_link = [node for node in linked_nodes
                                if node.num in link]
            if not nodes_to_link:
                continue
            symmetry_node = OccSymmetryHLConstraint(children=nodes_to_link,
                                                    name=f"Symmetry")
            self.nodes.append(symmetry_node)

        unlinked_site_el_nodes = [node for node in linked_nodes
                                    if node.is_root]
        for node in unlinked_site_el_nodes:
            symmetry_node = OccSymmetryHLConstraint(children=[node],
                                                    name="Symmetry")
            self.nodes.append(symmetry_node)

        # add offset nodes
        self._add_offset_nodes("occ offset (unused)")


class ChemBaseParam(BaseParam):
    def __init__(self, base_scatterer):
        self.atom = base_scatterer.atom
        self.n_free_params = 1
        self.elements = {base_scatterer.site_element.element:None,
                         'vac':None}
        super().__init__(base_scatterer)


class ChemParams(Params):

    def __init__(self, delta_slab):
        # initially, every atom-site-element has a free chemical weight
        # to allow for (partial) vacancies
        self.params = [ChemBaseParam(ase) for ase in delta_slab.base_scatterers]

        # iterate over atom-site-elements and link ones from the same atom
        # since we can't have more than 100% occupancy
        nums_processed = []
        linked_params = []
        for atom in delta_slab.non_bulk_atoms:
            atom_params = [param for param in self.base_params
                           if param.base_scatterer.atom == atom]
            if not atom_params:
                continue
            linked_param = SharedOccChemConstraint(children=atom_params)
            self.params.append(linked_param)
        super().__init__()

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
        top_level_start_indices = np.cumsum(np.array([param.n_free_params
                                     for param in free_top_level_params]))-1

        transform_biases = np.zeros((self.n_base_params))
        transform_weights = np.zeros((self.n_base_params, self.n_free_params))
        

        sum_parameters = 0
        for top_level_param in free_top_level_params:
            n_params = top_level_param.n_free_params
            block_matrix = np.full((n_params, n_params), -1/n_params)
            np.fill_diagonal(block_matrix, 1)
            sum_parameters += n_params


        weights = []
        for i, param in enumerate(self.base_params):
            element = param.site_element.element
            top_level = param
            while top_level.parent is not None:
                if element in top_level.parent.elements.keys():
                    top_level = top_level.parent
            if top_level.elements[element] is None:
                # zero bias, add weights to dynamic weights
                # for the corresponding top-level parameter, add 1 for
                # the corresponding element and -1/n for all others
                n_params = top_level.n_free_params
                top_level_index = free_top_level_params.index(top_level)
                top_level_param_slice = slice(
                    top_level_start_indices[top_level_index],
                    top_level_start_indices[top_level_index] + n_params
                )
                transform_weights[i, top_level_param_slice] = np.full((n_params,), -1/n_params)
                transform_weights[i, top_level_param_slice][list(top_level.elements.keys()).index(element)] = 1

            else:
                # keep weights as 0 and add fixed value to static weights
                transform_biases[i] = top_level.elements[element]

        transformer = LinearTransformer(transform_weights, transform_biases)
        assert transformer.n_free_params == sum_parameters

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
        return len([el for el, val in self.elements.items() if val is None and el != 'vac'])

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
