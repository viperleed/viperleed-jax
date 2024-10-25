"""Module occ_parameters."""

__authors__ = ("Alexander M. Imre (@amimre)",)
__created__ = "2024-09-08"

import numpy as np

from viperleed_jax.base import LinearTransformer

from .hierarchical_linear_tree import HLLeafNode, HLConstraintNode
from .hierarchical_linear_tree import HLTreeLayers
from .hierarchical_linear_tree import ParameterHLSubtree
from .linear_transformer import LinearTransformer


class OccHLLeafNode(HLLeafNode):
    """Represents a leaf node with occupational parameters."""

    def __init__(self, base_scatterer):
        dof = 1
        self.element = base_scatterer.site_element.element
        self.site = base_scatterer.site_element.site
        self.num = base_scatterer.num
        self.site_element = base_scatterer.site_element
        self.base_scatterer = base_scatterer
        self.name = f"occ (At_{self.num},{self.site},{self.element})"
        super().__init__(dof=dof, name=self.name)

    def _update_bounds(self, line):
        # occupational leaves are 1D, so bounds are scalars
        range = line.range
        self._bounds.update_range((range.start, range.stop), enforce=True)

    def update_offsets(self, line):
        offset = line.value
        self._bounds.update_range(_range=None, offset=offset, enforce=True)


class OccHLConstraintNode(HLConstraintNode):
    """Represents a constraint node for occupational parameters."""

    def __init__(self, dof, children, name, layer, transformers=None):
        self.dof = dof

        if transformers is None:
            raise ValueError("Transformers must be provided.")
        super().__init__(dof=dof, name=name, children=children,
                         layer=layer, transformers=transformers)

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
            bias = np.ones(1)
            transformers.append(LinearTransformer(weights, bias, (1,)))
        super().__init__(
            dof=dof,
            name=name,
            children=children,
            transformers=transformers,
            layer=HLTreeLayers.Symmetry,
        )

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
                         children=children,
                         transformers=transformers,
                         layer=HLTreeLayers.Symmetry)


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
            name=f"CONSTRAIN '{name}'",
            layer=HLTreeLayers.User_Constraints,
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

    def apply_explicit_constraint(self, constraint_line):
        # self._check_constraint_line_type(constraint_line, "occ")
        _, selected_roots = self._select_constraint(constraint_line)

        if not all(
            node.dof == selected_roots[0].dof for node in selected_roots
        ):
            raise ValueError(
                "All root nodes must have the same number of free parameters."
            )
        # create a constraint node for the selected roots
        self.nodes.append(
            OccLinkedHLConstraint(
                children=selected_roots, name=constraint_line.line
            )
        )

    @property
    def occ_weight_transformer(self):
        return self.subtree_root.collapse_transformer()
