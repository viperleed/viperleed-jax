"""Module parameter_space."""

__authors__ = ("Alexander M. Imre (@amimre)",)
__created__ = "2024-10-07"

from abc import ABC, abstractmethod
from enum import Enum
from itertools import compress

import numpy as np
import jax.numpy as jnp
import anytree
from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter
from anytree.walker import Walker

from viperleed_jax.files.displacements.lines import ConstraintLine
from .linear_transformer import LinearTransformer, stack_transformers

# Enable checks for the anytree library – we don't deal with huge trees so this
# should not be a performance issue.
anytree.config.ASSERTIONS = True


HLTreeLayers = Enum(
    "HLTreeLayers",
    ["Base", "Symmetry", "User_Constraints", "Implicit_Constraints", "Root"],
)


class HLNode(Node):
    """Base class for hierarchical linear tree nodes."""

    separator = "/"

    def __init__(self, dof, layer, name=None, parent=None, children=None):
        self.dof = dof  # Number of degrees of freedom
        self._transformer = None  # LinearTransformer object
        self.layer = HLTreeLayers(layer)

        self.name = f"({self.dof}) {name}" if name else f"({self.dof})"
        self.parent = parent
        if self.children:
            self.children = children

    def set_transformer(self, transformer):

        # check if the transformer is valid
        if not isinstance(transformer, LinearTransformer):
            raise TypeError(
                f"Transformer must be an instance of LinearTransformer. "
                f"Invalid transformer: {transformer}"
            )

        if transformer.out_dim != self.dof:
            raise ValueError(
                f"Transformer output dimension ({transformer.out_dim}) "
                f"must match child dof ({self.dof})."
            )
        self._transformer = transformer

    @property
    def transformer(self):
        if self._transformer is None:
            raise ValueError("Node does not have a transformer.")
        return self._transformer

    def _pre_attach(self, parent):

        # check that the parent is a ConstraintNode
        if not isinstance(parent, HLConstraintNode):
            raise TypeError(
                f"Parent must be an instance of HLConstraintNode. "
                f"Invalid parent: {parent}"
            )

        # check that the parent layer is >= child layer
        if not parent.layer.value >= self.layer.value:
            raise ValueError(
                f"Parent layer ({parent.layer}) must be greater or equal "
                f"to child layer ({self.layer})."
            )

        # check that the transformer dimensions match
        if self.transformer.in_dim != parent.dof:
            raise ValueError(
                f"Transformer input dimension ({self.transformer.in_dim}) "
                f"must match parent dof ({parent.dof})."
            )

    def _pre_detach(self, parent):
        raise RuntimeError(
            "The hierarchical linear tree does not support " "detaching nodes."
        )


class HLLeafNode(HLNode):

    def __init__(self, dof, name=None, parent=None):
        # initialize bounds
        self._bounds = HLBound(dof)
        super().__init__(dof=dof, name=name, parent=parent,
                         layer=HLTreeLayers.Base)

    @abstractmethod
    def update_bounds(self, line):
        pass


class HLConstraintNode(HLNode):
    """Base class for hierarchical linear tree constraint nodes.

    Any non-leaf node in the tree is a constraint node. Constraint nodes must
    be initialized with a list of child nodes and their corresponding
    transformers."""

    def __init__(self, dof, layer, name=None, children=[], transformers=None):
        _children = list(children)
        super().__init__(dof=dof, name=name, layer=layer)  # Initialize the base class

        if len(_children) == 0:
            raise ValueError(
                "Constraint nodes must be initialized with "
                "at least one child node."
            )
        # check that children are roots
        if any(not child.is_root for child in _children):
            raise ValueError("Children must be root nodes.")

        # if no transformers are provided, check if all children already
        # have transformers
        if transformers is None:
            if any(child.transformer is None for child in _children):
                raise ValueError("All children must have transformers.")
            transformers = [child.transformer for child in _children]

        # check that the number of children and transformers match
        if len(_children) != len(transformers):
            raise ValueError(
                f"Number of children ({len(_children)}) must match "
                f"number of transformers ({len(transformers)})."
            )

        # if everything is fine, set the transformers
        for child, transformer in zip(_children, transformers):
            child.set_transformer(transformer)

        # if that went well, set the children
        for child in _children:
            child.parent = self

    def check_bounds_valid(self):
        collapsed_tansformer = self.collapse_transformer()
        user_mask, lower, upper = self.collapse_bounds()

        # if no user set bounds are provided, return True
        if not np.any(user_mask):
            return True

        # discard all non-user specified lines
        transformer = collapsed_tansformer.select_rows(user_mask)
        lower, upper = lower[user_mask], upper[user_mask]

        # All of this gives us two (lower & upper bound) systems of linear equations
        # We can check if all requirements can be statified by checking if at least
        # one solution exists. This is equivalent to checking if the rank of the
        # augmented matrix is equal to the rank of the coefficient matrix.

        coeff_rank = np.linalg.matrix_rank(transformer.weights)

        upper_bound_matrix = np.hstack(
            [transformer.weights, (upper - transformer.biases).reshape(-1, 1)]
        )
        lower_bound_matrix = np.hstack(
            [transformer.weights, (lower - transformer.biases).reshape(-1, 1)]
        )
        upper_rank = np.linalg.matrix_rank(upper_bound_matrix)
        lower_rank = np.linalg.matrix_rank(lower_bound_matrix)

        if upper_rank < coeff_rank or lower_rank < coeff_rank:
            raise ValueError(
                "Bounds are not satisfiable"
            )  # TODO: better error message
        return True

    def _stacked_transformer(self):
        """Return the stacked transformer of the children."""

        child_weights = [child.transformer.weights for child in self.children]
        child_biases = [child.transformer.biases for child in self.children]

        stacked_weights = np.vstack(child_weights)
        stacked_biases = np.hstack(child_biases)
        return LinearTransformer(stacked_weights, stacked_biases,
                                 (np.sum([c.dof for c in self.children]),))

    def transformer_to_descendent(self, node):
        """Return the transformer from this node to a descendent."""
        walker = Walker()
        try:
            (upwards, common, downwards) = walker.walk(self, node)
        except walker.WalkError as err:
            raise ValueError(f"Node {node} cannot be reached from {self}.") from err
        if upwards:
            raise ValueError(f"Node {node} is not a descendent of {self}.")
        transformers = [node.transformer for node in downwards]
        composed_transformer = transformers[0]
        for trafo in transformers[1:]:
            composed_transformer = composed_transformer.compose(trafo)
        return composed_transformer

    def down_collapse_transformers(self, stop_condition):
        if stop_condition is None:
            stop_condition = lambda node: node.is_leaf
        collapsed_transformers = []
        for child in self.children:
            if stop_condition(child):
                collapsed_transformers.append(child.transformer)
            else:
                collapsed_transformers.append(
                    child.transformer.compose(
                        stack_transformers(child.down_collapse_transformers(
                            stop_condition
                        ))
                    )
                )
        return collapsed_transformers

    def collapse_transformer(self):
        """Iterate through through all descendants, collapsing the transformers."""
        collapsed_transformers = self.down_collapse_transformers(stop_condition=None)
        return stack_transformers(collapsed_transformers)

    def collapse_bounds(self):
        """Iterate through all descendants, collapsing the bounds."""
        user_set_bounds, lower_bounds, upper_bounds = [], [], []

        for child in self.children:
            if child.is_leaf:
                user_set_bounds.append(child._bounds.user_set)
                lower_bounds.append(child._bounds.lower)
                upper_bounds.append(child._bounds.upper)
            else:
                _user_set, _lower, _upper = child.collapse_bounds()
                user_set_bounds.extend(_user_set)
                lower_bounds.extend(_lower)
                upper_bounds.extend(_upper)

        return np.hstack(user_set_bounds), np.hstack(lower_bounds), np.hstack(upper_bounds)


class ImplicitHLConstraint(HLConstraintNode):
    """
    """

    # TODO
    def __init__(self, children):
        # can only have one child
        if len(children) != 1:
            raise ValueError("Implicit constraints must have exactly one child")
        child = children[0]
        child.check_bounds_valid()

        collapsed_tansformer = child.collapse_transformer()
        user_mask, lower, upper = child.collapse_bounds()

        # if no user set bounds are provided, return True
        if not np.any(user_mask):
            dof = 0
            new_transformer = LinearTransformer(np.zeros((child.dof, 0)), np.zeros(child.dof), (child.dof,))
        else:
            dof = np.sum(user_mask)

            # discard all non-user specified lines
            transformer = collapsed_tansformer.select_rows(user_mask)
            lower, upper = lower[user_mask], upper[user_mask]

            # All of this gives us two (lower & upper bound) systems of linear equations
            # We can check if all requirements can be statified by checking if at least
            # one solution exists. This is equivalent to checking if the rank of the
            # augmented matrix is equal to the rank of the coefficient matrix.

            new_biases = np.zeros(child.dof)
            new_weights = transformer.weights.T @ np.diag(upper - lower)
            new_transformer = LinearTransformer(
                new_weights, new_biases, (child.dof,)
            )
        super().__init__(dof=dof, name=f"Implicit Constraint",
                         children=[child], transformers=[new_transformer],
                         layer=HLTreeLayers.Implicit_Constraints)


class HLBound():
    """Class representing a bound in the hierarchical linear tree.

    Bounds are used to represent the lower and upper bounds of values that can
    be taken by the parameters represented by nodes. Bounds are assigned to leaf
    nodes in the tree.
    Bounds can be propagated up and down the tree.
    """
    _EPS = 1e-6

    def __init__(self, dimension):
        self.dimension = dimension
        self._user_set = np.full(shape=(self.dimension,), fill_value=False)
        self.update_range(range=(np.zeros(dimension), np.zeros(dimension)),
                          offset=np.zeros(dimension))

    @property
    def lower(self):
        return self._lower + self._offset

    @property
    def upper(self):
        return self._upper + self._offset

    @property
    def offset(self):
        return self._offset

    @property
    def fixed(self):
        return abs(self.upper - self.lower) < self._EPS

    @property
    def user_set(self):
        return self._user_set

    def update_range(self, range=None, offset=None, user_set=None):
        if range is None and offset is None:
            raise ValueError("range or offset must be provided")
        if range is not None:
            lower, upper = range
            lower = np.asarray(lower).reshape(self.dimension)
            upper = np.asarray(upper).reshape(self.dimension)
            self._lower = lower
            self._upper = upper
        if offset is not None:
            self._offset = np.asarray(offset).reshape(self.dimension)

        # mark the bounds that were user set;
        # use logical_or to combine the user set flags
        if user_set is None:
            user_set = np.full(self.dimension, False)
        _user_set = np.asarray(user_set).reshape(self.dimension)
        self._user_set = np.logical_or(self.user_set, _user_set)


    def __repr__(self):
        return f"HLBound(lower={self.lower}, upper={self.upper})"


class HLSubtree(ABC):

    def __init__(self):
        self.nodes = []
        self._subtree_root_has_been_created = False
        self.build_subtree()

    @property
    def roots(self):
        return [node for node in self.nodes if node.is_root]

    @property
    def leaves(self):
        return [node for node in self.nodes if node.is_leaf]

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def subtree_root_name(self):
        pass

    @abstractmethod
    def build_subtree(self):
        """Method to build the subtree for the parameter group."""
        pass

    def create_subtree_root(self):
        """Create a root node that aggregates all root nodes in the subtree."""
        if self._subtree_root_has_been_created:
            raise ValueError("Subtree root has already been created.")
        self._subtree_root_has_been_created = True
        if not self.roots:
            raise ValueError("No root nodes found in subtree.")
        root_dof = sum(node.dof for node in self.roots)
        transformers = []
        cum_node_dof = 0
        for node in self.roots:
            weights = np.zeros((node.dof, root_dof))
            weights[:, cum_node_dof : cum_node_dof + node.dof] = np.identity(
                node.dof
            )
            bias = np.zeros(node.dof)
            transformers.append(LinearTransformer(weights, bias, (node.dof,)))
            cum_node_dof += node.dof
        self.subtree_root = HLConstraintNode(
            dof=root_dof,
            name=self.subtree_root_name,
            children=self.roots,
            transformers=transformers,
            layer=HLTreeLayers.Root,
        )
        self.nodes.append(self.subtree_root)

    def __repr__(self):
        if not self._subtree_root_has_been_created:
            partial_trees = [RenderTree(root).by_attr() for root in self.roots]
            trees_str = "\n".join(partial_trees)

            return f"{trees_str}"
        return RenderTree(self.subtree_root).by_attr()

    def roots_up_to_layer(self, layer):
        _layer = HLTreeLayers(layer)
        return [node for node in self.nodes
                if node.layer.value <= _layer.value and
                (node.is_root or node.parent.layer.value > _layer.value)]

    def graphical_export(self, filename):
        if not self._subtree_root_has_been_created:
            raise ValueError("Subtree root has not yet been created.")
        UniqueDotExporter(self.subtree_root).to_picture(filename)

    def collapsed_transformer(self):
        if not self._subtree_root_has_been_created:
            raise ValueError("Subtree root has not yet been created.")
        return self.subtree_root.collapse_transformer()

    @property
    def leaf_is_dynamic(self):
        boolified_transformer = self.collapsed_transformer().boolify()
        input_val = np.full(shape=(self.subtree_root.dof,), fill_value=True, dtype=bool)

        bool_outputs = np.asarray(boolified_transformer(input_val), dtype=bool)

        running_dof_count = 0
        dynamic_leafs = []
        for leaf in self.leaves:
            leaf_dof = leaf.dof
            leaf_is_dynamic = np.any(
                bool_outputs[running_dof_count:running_dof_count + leaf_dof])
            dynamic_leafs.append(leaf_is_dynamic)
            running_dof_count += leaf_dof

        return np.array(dynamic_leafs, dtype=bool)


class ParameterHLSubtree(HLSubtree):
    """Base class representing a subtree in the hierarchical linear tree.

    Subtrees are used to group nodes for a group of parameters (vib, geo, occ,
    V0r). This allows implementing constraints more easily and makes it possible
    to extract additional information from the tree (e.g. propagator
    transformations).
    """

    def __init__(self, base_scatterers):
        self.base_scatterers = base_scatterers
        self.site_elements = self.base_scatterers.site_elements

        self._offsets_have_been_added = False
        super().__init__()

    def apply_bounds(self, line):
        targets = line.targets
        _, explicitly_selected_leaves, selected_roots = self._target_nodes(
            targets
        )
        primary_leaves = self._select_primary_leaf(
            selected_roots, explicitly_selected_leaves
        )

        # apply the bound to the primary leaf only – others will be linked
        # (this is so that for e.g. geometries the bounds are not swapped
        # and violate symmetry)
        for leaf in primary_leaves.values():
            leaf.update_bounds(line)

    def apply_offsets(self, line):
        """Apply offsets to the children of the node."""
        targets = line.targets
        _, explicitly_selected_leaves, selected_roots = self._target_nodes(
            targets
        )
        primary_leaves = self._select_primary_leaf(
            selected_roots, explicitly_selected_leaves
        )

        # apply the bound to the primary leaf only – others will be linked
        # (this is so that for e.g. geometries the bounds are not swapped
        # and violate symmetry)
        for leaf in primary_leaves.values():
            leaf.update_offsets(line)

    def check_for_inconsistencies(self):
        """Check for inconsistencies in the parameter space.

        This method checks for inconsistencies in the parameter space, such as
        symmetry violations, and raises an error if any are found.
        """
        for root in self.roots:
            root.check_bounds_valid()

    def _check_constraint_line_type(self, constraint_line, constraint_type):
        if not isinstance(constraint_line, ConstraintLine):
            raise ValueError("Constraint must be a ConstraintLine.")
        if constraint_line.constraint_type != constraint_type:
            raise ValueError(f"Constraint must be a {constraint_type} constraint.")

    def _target_nodes(self, targets):
        """Takes a BSTarget and returns the corresponding leaves and roots."""
        # gets the leaves that are affected by the targets
        explicitly_selected_leaves = list(
            compress(
                self.leaves, targets.select(self.base_scatterers)
            )
        )
        if not explicitly_selected_leaves:
            raise ValueError(
                f"No leaf nodes found for target {targets}."
            )

        # get the corresponding root nodes
        selected_roots = list(
            {leaf.root: None for leaf in explicitly_selected_leaves}.keys()
        )
        # then get all unique leaves that grow form the selected roots
        affected_leaves_dict = {}
        for root in selected_roots:
            affected_leaves_dict.update({leaf: None for leaf in root.leaves})
        implicitly_selected_leaves = list(affected_leaves_dict.keys())

        return (
            implicitly_selected_leaves,
            explicitly_selected_leaves,
            selected_roots
        )

    def _select_primary_leaf(self, roots, explicit_leaves):
        # make a dict that maps the primary leaf for each root
        # go through the leaves in reverse order to assign the first leaf in
        # the list to the root
        # If the root has no leafs in the explicit list, assign the first leaf
        primary_leaves = {}
        for root in roots:
            for leaf in reversed(explicit_leaves):
                if leaf in root.leaves:
                    primary_leaves[root] = leaf
            if root not in primary_leaves:
                primary_leaves[root] = root.leaves[0]
        return primary_leaves

    def _select_constraint(self, constraint_line):
        # gets the leaves that are affected by a constraint

        # TODO: other constraints ?
        if constraint_line.value != "linked":
            raise NotImplementedError("Only linked constraints are supported.")

        targets = constraint_line.targets
        implicit_leaves, explicit_leaves, selected_roots = self._target_nodes(targets)
        return implicit_leaves, explicit_leaves, selected_roots

    def apply_implicit_constraints(self):
        for root in self.roots:
            implicit_node = ImplicitHLConstraint([root])
            self.nodes.append(implicit_node)
