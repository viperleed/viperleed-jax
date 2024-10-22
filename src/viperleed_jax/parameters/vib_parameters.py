import numpy as np
from jax import numpy as jnp

from .linear_transformer import LinearTransformer
from viperleed_jax.parameters.base_parameters import BaseParam, Params, ConstrainedDeltaParam, Bound

from .hierarchical_linear_tree import HLLeafNode, HLConstraintNode
from .hierarchical_linear_tree import ParameterHLSubtree
from .hierarchical_linear_tree import HLTreeLayers


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

    def update_bounds(self, line):
        # vibrational leaves are 1D, so bounds are scalars
        range = line.range
        self._bounds.update_range(range=(range.start, range.stop),
                                  offset=None,
                                  user_set=True)

    def update_offsets(self, line):
        offset = line.value
        self._bounds.update_range(range=None, offset=offset, user_set=True)


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
            site_link_node = VibHLConstraintNode(
                name=f"vib ({site_el.site},{site_el.element})",
                children=nodes_to_link,
                layer=HLTreeLayers.Symmetry,
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
