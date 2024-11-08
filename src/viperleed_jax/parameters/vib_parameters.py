"""Module vib_parameters."""

__authors__ = ("Alexander M. Imre (@amimre)",)
__created__ = "2024-09-09"

import numpy as np

from .hierarchical_linear_tree import HLLeafNode, HLConstraintNode
from .hierarchical_linear_tree import HLTreeLayers
from .hierarchical_linear_tree import ParameterHLSubtree
from .linear_transformer import LinearTransformer


class VibHLLeafNode(HLLeafNode):
    """Represents a leaf node with vibrational parameters."""

    def __init__(self, base_scatterer):
        dof = 1
        self.element = base_scatterer.site_element.element
        self.site = base_scatterer.site_element.site
        self.num = base_scatterer.num
        self.site_element = base_scatterer.site_element
        self.ref_vib_amp = base_scatterer.atom.site.vibamp[self.element]
        self.name = f"vib (At_{self.num},{self.site},{self.element})"
        super().__init__(dof=dof, name=self.name)

        # apply reference vibrational amplitudes as non-enforced bounds
        self._bounds.update_range(
            _range=None, offset=self.ref_vib_amp, enforce=False
        )

    def _update_bounds(self, line):
        # vibrational leaves are 1D, so bounds are scalars
        range = line.range
        self._bounds.update_range(_range=(range.start, range.stop),
                                  offset=None,
                                  enforce=True)

    def update_offsets(self, line):
        offset = line.value
        self._bounds.update_range(_range=None, offset=offset, enforce=True)


class VibHLConstraintNode(HLConstraintNode):
    """Represents a constraint node for vibrational parameters."""

    def __init__(self, children, name, layer, dof=1, transformers=None):

        if dof != 1:
            raise ValueError("Vibrational constraints must have dof=1.")

        if transformers is None:
            # default to identity transformers
            transformers = [LinearTransformer(np.eye(1), np.zeros(1), (1,))
                            for _ in children]
        super().__init__(dof=dof, name=name, children=children, layer=layer,
                         transformers=transformers)


class VibLinkedHLConstraint(VibHLConstraintNode):
    """Class for explicit links of geometric parameters."""

    def __init__(self, children, name):
        # check that all children have the same dof
        if len(set(child.dof for child in children)) != 1:
            raise ValueError("Children must have the same dof.")
        dof = children[0].dof

        # ensure that children have consistent bounds
        transformers = []
        for child in children:
            # create a transformer for each child
            child.check_bounds_valid()
            mask, upper, lower = child.collapse_bounds()
            # if the child has no enforced bounds, no bounds were defined
            # this means the node would be implicitly fixed; which is
            # incompatible with explicit constraints!
            if not np.any(mask):
                raise ValueError(
                    f"Linking vibrations for {child.name} requires "
                    f"bounds to be defined."
                )
            _upper, _lower = upper[mask], lower[mask]
            # all _upper and all _lower should be the same
            if not np.all(_upper == _upper[0]) or not np.all(_lower == _lower[0]):
                raise ValueError(
                    f"Inconsistent bounds for {child.name}."
                )
            # let's create the transformer
            weights = np.array([[_upper[0] - _lower[0]]])
            biases = np.array([_lower[0]])
            transformers.append(LinearTransformer(weights, biases, (1,)))

        super().__init__(
            dof=dof,
            children=children,
            transformers=transformers,
            name=f"CONSTRAIN '{name}'",
            layer=HLTreeLayers.User_Constraints,
        )


class VibHLSiteConstraint(VibHLConstraintNode):
    """Class for linking vibrations of the same site."""

    def __init__(self, children):
        # check that all children are leaf nodes and share a site-element
        if not all(isinstance(child, VibHLLeafNode) for child in children):
            raise ValueError("Children must be VibHLLeaf nodes.")
        if not all(child.site_element == children[0].site_element
                   for child in children):
            raise ValueError("Children must have the same site-element.")
        self.ref_vib_amp = children[0].ref_vib_amp
        self.site_element = children[0].site_element
        dof = 1

        super().__init__(
            dof=dof,
            children=children,
            transformers=None,  # transformers default to identity
            name=(f"vib ({children[0].site_element.site},"
                  f"{children[0].site_element.element})"),
            layer=HLTreeLayers.Symmetry,
        )

class VibHLSubtree(ParameterHLSubtree):
    def __init__(self, base_scatterers):
        super().__init__(base_scatterers)

    @property
    def name(self):
        return "Vibrational Parameters"

    @property
    def subtree_root_name(self):
        return "vib root"

    def build_subtree(self):

        leaf_nodes = [VibHLLeafNode(ase)
                    for ase in self.base_scatterers]

        self.nodes.extend(leaf_nodes)

        # link site-elements together
        for site_el in self.site_elements:
            nodes_to_link = [
                node for node in self.leaves if node.site_element == site_el
            ]
            if not nodes_to_link:
                continue
            site_link_node = VibHLSiteConstraint(
                children=nodes_to_link,
            )
            self.nodes.append(site_link_node)

        # check that all bounds are valid
        for node in self.roots:
            node.check_bounds_valid()


    def apply_explicit_constraint(self, constraint_line):
        # self._check_constraint_line_type(constraint_line, "vib")
        *_, selected_roots = self._select_constraint(constraint_line)

        if not all(
            node.dof == selected_roots[0].dof for node in selected_roots
        ):
            raise ValueError(
                "All root nodes must have the same number of free parameters."
            )
        # create a constraint node for the selected roots
        self.nodes.append(
            VibLinkedHLConstraint(
                children=selected_roots, name=constraint_line.line
            )
        )

    ##############################
    # Vibration specific methods #
    ##############################
    def all_vib_amps_transformer(self):
        """Return a transformer that maps the free parameters to all vibrational
        amplitudes"""
        return self.subtree_root.collapse_transformer()

    def dynamic_t_matrix_transformers(self):
        """Return a transformer that maps the free parameters to the dynamic
        vibrational amplitudes."""
        dynamic_reference_nodes = {
            node.site_element:node for node
            in reversed(np.array(self.leaves)[self.leaf_is_dynamic])
        }
        return [self.subtree_root.transformer_to_descendent(node)
                for node in dynamic_reference_nodes.values()]

    @property
    def dynamic_site_elements(self):
        dynamic_site_elements = [
            node.site_element for node
            in np.array(self.leaves)[self.leaf_is_dynamic]
        ]
        # make unique
        return tuple(dict.fromkeys(dynamic_site_elements))

    @property
    def static_t_matrix_inputs(self):
        static_t_matrix_inputs = {
            node.site_element:node.ref_vib_amp for node
            in np.array(self.leaves)[~self.leaf_is_dynamic]
        }
        return static_t_matrix_inputs

    @property
    def static_site_elements(self):
        return tuple(self.static_t_matrix_inputs.keys())

    @property
    def t_matrix_map(self):
        # return a tuple with the site_elements for each base parameter
        return [
            ('static', self.static_site_elements.index(leaf.site_element))
            if leaf.site_element in self.static_site_elements else
            ('dynamic', self.dynamic_site_elements.index(leaf.site_element))
            for leaf in self.leaves
        ]

    @property
    def n_dynamic_t_matrices(self):
        return len(self.dynamic_site_elements)

    @property
    def n_static_t_matrices(self):
        return len(self.static_site_elements)
