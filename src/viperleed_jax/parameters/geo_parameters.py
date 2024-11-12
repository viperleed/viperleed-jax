"""Module geo_parameters."""

__authors__ = ("Alexander M. Imre (@amimre)",)
__created__ = "2024-08-30"

from collections import deque
from itertools import zip_longest

from anytree.walker import Walker, WalkError
import numpy as np

from viperleed_jax import atomic_units
from viperleed_jax.files.displacements.lines import ConstraintLine

from .hierarchical_linear_tree import HLScattererLeafNode, HLConstraintNode
from .hierarchical_linear_tree import HLTreeLayers
from .hierarchical_linear_tree import ParameterHLSubtree
from .linear_transformer import LinearMap


class GeoHLLeafNode(HLScattererLeafNode):
    """Represents a leaf node with geometric parameters."""
    _Z_DIR_ID = 0 # TODO: unify and move to a common place

    def __init__(self, base_scatterer):
        dof = 3
        super().__init__(dof=dof, base_scatterer=base_scatterer)
        self.symrefm = base_scatterer.atom.symrefm
        self.name = f"geo (At_{self.num},{self.site},{self.element})"

    def _update_bounds(self, line):
        # geometric leaf bounds are 3D
        range = line.range
        direction = line.direction

        if direction._fractional:
            raise NotImplementedError("TODO")

        if direction.num_free_directions == 1 and direction._vectors[0][2] == 1:  # TODO: index 2 here needs to be changed to LEED convention
            # z-only movement
            start = np.array([range.start, 0., 0.])
            stop = np.array([range.stop, 0., 0.])
            user_set = [True, False, False]
        else:
            raise NotImplementedError("TODO")
        self._bounds.update_range((start, stop), enforce=user_set)

    def update_offsets(self, line):
        # geometric leaf bounds are 3D
        direction = line.direction

        if direction._fractional:
            raise NotImplementedError("TODO")

        if (
            direction.num_free_directions == 1
            and direction._vectors[0][2]
            == 1  # TODO: index 2 here needs to be changed to LEED convention
        ):
            # z-only movement
            offset = np.array([line.value, 0., 0.])
            user_set = np.array([True, False, False])
        else:
            raise NotImplementedError("TODO")
        self._bounds.update_range(_range=None, offset=offset, enforce=user_set)

    @property
    def propagator_origin(self):
        """Return the node that is the origin of the propagator for this leaf"""
        origin = self
        while origin.parent:
            if isinstance(origin.parent, GeoHLConstraintNode) and origin.parent.shared_propagator:
                origin = origin.parent
            else:
                break
        return origin

    @property
    def symmetry_operation_to_reference_propagator(self):
        """Return the symmetry operation that links this leaf to the reference
        propagator."""
        node_walker = Walker()
        target = self.propagator_origin.propagator_reference_node
        try:
            (upwards, common, downwards) = node_walker.walk(
                self, target
            )
        except WalkError as err:
            raise RuntimeError(f"Node {self} cannot be reached from "
                               f"{self.propagator_origin}.") from err
        if target is self:  # identity
            return np.eye(3)

        # sanity check
        if not common.shared_propagator:
            raise ValueError("Common node must have shared propagator")

        # traverse the tree and add up symmetry operations
        operations = deque()
        for up, down in zip_longest(upwards, reversed(downwards),
                                    fillvalue=None):
            if (up is not None and down is not None
                and up.transformer == down.transformer):
                continue
            if up is not None:
                if np.any(up.transformer.biases != 0):
                    raise ValueError("Bias must be zero")
                inverse = np.linalg.inv(up.transformer.weights)
                operations.appendleft(inverse)
            if down is not None:
                if np.any(down.transformer.biases != 0):
                    raise ValueError("Bias must be zero")
                operations.append(down.transformer.weights)
        operations.appendleft(np.eye(3))
        operations.append(np.eye(3))
        return np.linalg.multi_dot(operations)


class GeoHLConstraintNode(HLConstraintNode):
    """Base constraint node for geometric parameters."""

    def __init__(self, dof, children, transformers, layer, name="unnamed",
                 shared_propagator=False):
        self.dof = dof
        if shared_propagator:
            self.propagator_reference_node = self._check_reference_node(
                children, transformers)
            self.shared_propagator = shared_propagator

        if transformers is None:
            raise ValueError(
                "Transformers must be provided for "
                "geometric constraint nodes."
            )
        super().__init__(
            dof=dof, name=name, children=children,
            transformers=transformers, layer=layer
        )

    def _check_reference_node(self, children, transformers):
        """Checks if a shared propagator is allowed and if so, selects the
        reference node.

        For the shared propagator to be allowed, one of these conditions
        must be met:
        1. all children are leaf nodes (i.e. self is a symmetry node)
        2. all children are constraint nodes with shared propagators
           and all their transformers have 0 bias, invertible weights
           and det(weights) = 1
        3. all children are constraint nodes with shared propagators
           and their transformers are the same
        """
        # first case
        if all(isinstance(child, GeoHLLeafNode) for child in children):
            # choose first child as reference node
            return children[0]
        # check node type
        for child in children:
            if not isinstance(child, GeoHLConstraintNode):
                raise ValueError(
                    "Shared propagator nodes must have shared propagator "
                    "children."
                )
        # second case
        if all([np.any(trafo.biases == 0) and np.linalg.det(trafo.weights) == 1. # TODO: use EPS
                for trafo in transformers]):
            try:
                inverted_weights = [
                    np.linalg.inv(trafo.weights) for trafo in transformers
                ]
            except np.linalg.LinAlgError:
                raise ValueError(
                    "Shared propagator transformers must have invertible "
                    "weights."
                )
            # select the reference node of the first child
            return children[0].propagator_reference_node
        # third case
        if all([trafo == transformers[0] for trafo in transformers]):
            # select the reference node of the first child
            return children[0].propagator_reference_node


class GeoSymmetryHLConstraint(GeoHLConstraintNode):
    """Constraint node for symmetry constraints on geometric parameters.

    Symmetry constraints are the first layer of constraints applied to the
    geometric leaf nodes. They are link atoms based on symmetry relations and
    always reduce the number of free parameters to a maximum of 3.
    All transformers must therefore have in_dim=3 and the bias must be zero.
    Further properties of the transformer are determined by the symmetry and
    the allowed directions of movement.
    The viperleed.calc symmetry recognition gives atoms the attributes `symrefm`
    and `freedir`. `freedir` determines the allowed directions of movement, and
    `symrefm` is the symmetry operation that links the atoms that are symmetry
    linked."""

    def __init__(self, children):
        # transformers may not be provided, as they must be determined from the
        # symmetry operations
        if not children:
            raise ValueError("Symmetry constraints must have children")

        # make sure that all children are leaf nodes
        if not all(isinstance(child, GeoHLLeafNode) for child in children):
            raise ValueError(
                "Symmetry constraints can only be applied to "
                "geometric leaf nodes."
            )

        # check that all children are in the same linklist
        linklist = children[0].base_scatterer.atom.linklist
        if not all([child.base_scatterer.atom in linklist for child in children]):
            raise ValueError(
                "Symmetry linked atoms must be in the same " "linklist"
            )

        # irrespective of the symmetry the transformer bias so we use a map
        transformers = []

        # how to proceed is determined by the freedir attribute
        # NB: this is a weird attribute and will be changed in the future
        # Currently freedir can be either an int (specifically 0 or 1, implying
        # z-only or completely free movement) or a 1D array of shape (2,)
        # (implying 1D in-plane movement in addition to z)
        if children[0].base_scatterer.atom.freedir == 0:
            # check that all children have the same freedir
            if not all(
                child.base_scatterer.atom.freedir == 0 for child in children
            ):
                raise ValueError(
                    "All symmetry linked atoms must have the same "
                    "freedir attribute."
                )
            # z-only movement
            dof = 1
            name = "Symmetry (z-only)"
            for child in children:
                # set the symmetry linking matrix and direct transfer of z
                weights = np.array([1.0, 0., 0.]).reshape((3,1))
                transformers.append(LinearMap(weights, (3,)))

        elif children[0].base_scatterer.atom.freedir == 1:
            # check that all children have the same freedir
            if not all(
                child.base_scatterer.atom.freedir == 1 for child in children
            ):
                raise ValueError(
                    "All symmetry linked atoms must have the same "
                    "freedir attribute."
                )
            # free in-plane movement in addition to z
            dof = 3
            name = "Symmetry (free)"

            for child in children:
                # set the symmetry linking matrix and direct transfer of z
                weights = np.identity(3)
                weights[1:3, 1:3] = child.symrefm
                transformers.append(LinearMap(weights, (3,)))

        elif children[0].base_scatterer.atom.freedir.shape == (2,):
            # check that all children have the same freedir
            if not all(
                child.base_scatterer.atom.freedir.shape == (2,)
                for child in children
            ):
                raise ValueError(
                    "All symmetry linked atoms must have the same "
                    "freedir attribute."
                )
            # 1D in-plane movement in addition to z
            dof = 2
            name = "Symmetry (1D in-plane)"
            # TODO: sort this out (discuss using fractional coordinates)
            raise NotImplementedError
        else:
            raise ValueError(
                "freedir attribute must be 0, 1, or have shape (2,)"
            )

        super().__init__(
            dof=dof,
            children=children,
            transformers=transformers,
            name=name,
            layer=HLTreeLayers.Symmetry,
            shared_propagator=True,  # symmetry nodes always share propagators
        )


class GeoLinkedHLConstraint(GeoHLConstraintNode):
    """Class for explicit links of geometric parameters."""
    # TODO: if we implement linking of nodes with different dof (directional),
    # this needs to be adapted
    # TODO: this also needs to be adapted if we allow partial directional linking
    # e.g. linking z coordinates of two atoms, but not x and y
    # TODO: furthermore: linking of atoms in the same direction, but with
    #       different magnitudes would also be problematic! – check math for this
    #       case. I'm not sure if the propagators are linear with respect to the
    #       length of the displacement vector
    # If either of these is implemented, the children will not share a propagator
    def __init__(self, children, name):
        # check that all children have the same dof
        if len(set(child.dof for child in children)) != 1:
            raise ValueError("Children must have the same dof.")
        dof = children[0].dof

        # transformers can be identity
        transformers = [
            LinearMap(np.eye(dof), (dof,))
            for _ in children
        ]
        super().__init__(
            dof=dof, children=children, transformers=transformers,
            name=f"CONSTRAIN '{name}'",
            layer=HLTreeLayers.User_Constraints,
            shared_propagator=True,  # see comment above
        )

class GeoHLSubtree(ParameterHLSubtree):
    def __init__(self, base_scatterers):
        super().__init__(base_scatterers)

    @property
    def name(self):
        return "Geometric Parameters"

    @property
    def subtree_root_name(self):
        return "geo root"

    def build_subtree(self):
        # create leaf nodes
        geo_leaf_nodes = [
            GeoHLLeafNode(base_scatterer)
            for base_scatterer in self.base_scatterers
        ]
        self.nodes.extend(geo_leaf_nodes)

        # apply symmetry constraints
        for siteel in self.site_elements:
            site_el_params = [
                node for node in self.leaves if node.site_element == siteel
            ]

        for link in self.base_scatterers.symmetry_links:
            # put all linked atoms in the same symmetry group
            nodes_to_link = [
                node for node in self.leaves
                if node.base_scatterer in link
            ]
            if nodes_to_link:
                self.nodes.append(GeoSymmetryHLConstraint(children=nodes_to_link))

        unlinked_site_el_nodes = [node for node in self.leaves if node.is_root]
        for node in unlinked_site_el_nodes:
            self.nodes.append(GeoSymmetryHLConstraint(children=[node]))

    def apply_explicit_constraint(self, constraint_line):
        # self._check_constraint_line_type(constraint_line, "geo")
        *_, selected_roots = self._select_constraint(constraint_line)

        if constraint_line.direction is None:
            # complete linking; requires all root nodes to have the same dof
            if not all(node.dof == selected_roots[0].dof for node in selected_roots):
                raise ValueError(
                    "All root nodes must have the same number of free parameters."
                )
            # create a constraint node for the selected roots
            self.nodes.append(GeoLinkedHLConstraint(children=selected_roots,
                                                    name=constraint_line.line))
        else:
            raise NotImplementedError(
                "Directional geo constraints are not yet supported.")

    #############################
    # Geometry specific methods #
    #############################
    def all_displacements_transformer(self):
        """Return a transformer that gives the displacements for all base
        scatterers."""
        collapsed_transformer = self.collapsed_transformer_scatterer_order
        collapsed_transformer.out_reshape = (-1, 3)
        return collapsed_transformer

    def dynamic_displacements_transformers(self):
        """Return a list of transformers that give the reference displacements
        for the dynamic propagators."""
        return [
            self.subtree_root.transformer_to_descendent(node.propagator_reference_node)
            for node in self.dynamic_origin_nodes
        ]

    def _dynamic_origin_dict(self):
        dynamic_leaves = [leaf for leaf in np.array(self.leaves)[self.leaf_is_dynamic]]
        origin_dict = {
            leaf: leaf.propagator_origin for leaf in dynamic_leaves
        }
        return origin_dict

    @property
    def dynamic_origin_nodes(self):
        """Return nodes that are the origin of the propagator for dynamic leaves."""
        return list(dict.fromkeys(list(self._dynamic_origin_dict().values())))

    @property
    def transformers_for_dynamic_propagator_inputs(self):
        return [
            self.subtree_root.transformer_to_descendent(node)
            for node in self.dynamic_origin_nodes
        ]

    def _static_origin_dict(self):
        static_leaves = [leaf for leaf in np.array(self.leaves)[~self.leaf_is_dynamic]]
        origin_dict = {
            leaf: leaf.propagator_origin for leaf in static_leaves
        }
        return origin_dict

    @property
    def static_origin_nodes(self):
        """Return nodes that are the origin of the propagator for static leaves."""
        return list(dict.fromkeys(list(self._static_origin_dict().values())))

    @property
    def static_propagator_inputs(self):
        """Return the displacements for the static propagators."""
        static_propagator_transformers = [
            self.subtree_root.transformer_to_descendent(node)
            for node in self.static_origin_nodes
        ]
        # since the transformers are static, we can evaluate them
        return [transformer(np.full(self.subtree_root.dof, 0.5))
                for transformer in static_propagator_transformers]

    @property
    def propagator_map(self):
        """Return a mapping of base scatterers to propagators."""
        return [
            (
                (
                    "static",
                    self.static_origin_nodes.index(
                        self._static_origin_dict()[leaf]
                    ),
                )
                if not dynamic
                else (
                    "dynamic",
                    self.dynamic_origin_nodes.index(
                        self._dynamic_origin_dict()[leaf]
                    ),
                )
            )
            for leaf, dynamic in zip(self.leaves, self.leaf_is_dynamic)
        ]

    def _leaf_symmetry_operations(self):
        """Return the symmetry operations for each leaf in respect to the
        reference displacement (the one for which the propagator is calculated).
        """
        return tuple([leaf.symmetry_operation_to_reference_propagator
                      for leaf in self.leaves])

    @property
    def leaf_plane_symmetry_operations(self):
        """Return the in-plane symmetry operations for each leaf in respect to the
        reference displacement (the one for which the propagator is calculated).
        """
        for (
            sym_op
        ) in self._leaf_symmetry_operations():  # TODO: can this even happen?
            if np.any(sym_op[0,:] != np.array([1., 0, 0])):
                raise ValueError("Symmetry operation must be in-plane! "
                                 "This should not happen!")
        return tuple([sym_op[1:, 1:] for sym_op in self._leaf_symmetry_operations()])

    @property
    def n_dynamic_propagators(self):
        return len(self.dynamic_origin_nodes)

    @property
    def n_static_propagators(self):
        return len(self.static_origin_nodes)


def geo_sym_linking(atom):
    linking = np.zeros(shape=(3, 3))
    linking[1:3, 1:3] = atom.symrefm  # TODO: round off the 1e-16 contributions
    linking[0, 0] = 1.0  # all symmetry linked atoms move together is z directon
    return linking
