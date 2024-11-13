"""Module parameter_space."""

__authors__ = ("Alexander M. Imre (@amimre)",)
__created__ = "2024-10-07"

from abc import ABC, abstractmethod
from enum import Enum
from itertools import compress

import numpy as np
import anytree
from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter
from anytree.walker import Walker

from viperleed_jax.files.displacements.lines import ConstraintLine
from .linear_transformer import LinearTransformer, LinearMap, stack_transformers

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

    def update_bounds(self, line):
        self._update_bounds(line)
        self.parent.check_bounds_valid()

    @abstractmethod
    def _update_bounds(self, line):
        pass

    @property
    def free(self):
        return ~self._bounds.fixed


class HLScattererLeafNode(HLLeafNode):

    def __init__(self, dof, base_scatterer, name=None, parent=None):
        # base scatterer based attributes
        self.base_scatterer = base_scatterer
        self.element = base_scatterer.site_element.element
        self.num = base_scatterer.num
        self.site = base_scatterer.site_element.site
        self.site_element = base_scatterer.site_element
        super().__init__(dof=dof, name=name, parent=parent)


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

        # if one child is a leaf, all children must be leaves
        if any(child.is_leaf for child in _children):
            if not all(child.is_leaf for child in _children):
                raise ValueError("If one child is a leaf node, all children "
                                 "must be leaf nodes.")

        # dof cannot be larger than the sum of the children's dofs
        if self.dof > sum(child.dof for child in _children):
            raise ValueError("Degree of freedom must be less than or equal to "
                             "the sum of the children's degrees of freedom.")

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
        """Check that the bounds of the children are valid."""
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

    def stacked_bounds(self):
        free, lower, upper = [], [], []
        for leaf in self.leaves:
            free.append(leaf.free)
            lower.append(leaf._bounds.lower)
            upper.append(leaf._bounds.upper)
        return np.hstack(free), np.hstack(lower), np.hstack(upper)

    # TODO: is this/can this be replaced by stacked_bounds()?
    def collapse_bounds(self):
        """Iterate through all descendants, collapsing the bounds."""
        enforced_bounds, lower_bounds, upper_bounds = [], [], []

        for child in self.children:
            if child.is_leaf:
                enforced_bounds.append(child._bounds.enforce)
                lower_bounds.append(child._bounds.lower)
                upper_bounds.append(child._bounds.upper)
            else:
                _enforce, _lower, _upper = child.collapse_bounds()
                enforced_bounds.extend(_enforce)
                lower_bounds.extend(_lower)
                upper_bounds.extend(_upper)

        return np.hstack(enforced_bounds), np.hstack(lower_bounds), np.hstack(upper_bounds)

    @property
    def free(self):
        partial_free = []
        for child in self.children:
            # We use the Penrose Moore pseudo inverse to see which degrees of
            # are needed to satisfy the constraints.
            # This essentially propagates the information about which implicitly
            # fixed and free parameters up the tree.
            pseudo_inverse = np.linalg.pinv(child.transformer.weights)
            # re-cast into a boolean array
            partial_free.append(np.bool_(pseudo_inverse @ child.free))
        # take the logical or of all the partial free arrays
        return np.logical_or.reduce(partial_free)

# TODO: rename to bounds constraint

class ImplicitHLConstraint(HLConstraintNode):
    """Class representing implicit constraints in the hierarchical linear tree.
    
    Implicit constraints are constraints that are not explicitly defined by the
    user, but rather by not explicitly allowing certain degrees of freedom.
    E.g. if a group of symmetry equivalent atoms is not given a displacement
    range, they are assumed to be static and will be constrained through an
    implicit constraint node.
    Implicit constraints are above user constraints in the hierarchy and are the
    last layer before the subtree root node.
    """

    def __init__(self, children):
        # can only have one child
        if len(children) != 1:
            raise ValueError("Implicit constraints must have exactly one child")
        child = children[0]
        child.check_bounds_valid()

        # using the child's free property, reduce the dof as much as possible
        # by removing fixed degrees of freedom
        dof = np.sum(child.free)
        weights = np.diag(child.free)[child.free]
        weights = (weights.astype(float)).T
        implicit_trafo = LinearMap(weights, (child.dof,))

        # TODO: does this automaticall raise if we have bound conflicts?

        # now get a transfomer that enforces the bounds
        free, lower, upper = child.stacked_bounds()
        if np.any(free):
            partial_trafo = child.collapse_transformer().select_rows(free)
            lower = lower[free]
            upper = upper[free]
        else:
            # all fixed, but may still have a default value
            partial_trafo = child.collapse_transformer()
        inverted_weights = np.linalg.pinv(partial_trafo.weights)
        norm_lower = inverted_weights @ (lower - partial_trafo.biases)
        norm_upper = inverted_weights @ (upper - partial_trafo.biases)
        weights = np.diag(norm_upper - norm_lower)
        biases = norm_lower
        range_trafo = LinearTransformer(weights, biases)

        # compose the two transformers
        composed_transformer = implicit_trafo.compose(range_trafo)

        super().__init__(dof=dof, name=f"Bounds Constraint",
                         children=[child],
                         transformers=[composed_transformer],
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
        self._enforce = np.full(shape=(self.dimension,), fill_value=False)
        self._lower, self._upper = np.zeros(dimension), np.zeros(dimension)
        self._offset = np.zeros(dimension)
        self.update_range(_range=(np.zeros(dimension), np.zeros(dimension)),
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
    def enforce(self):
        return self._enforce

    def update_range(self, _range=None, offset=None, enforce=None):
        if _range is None and offset is None:
            raise ValueError("range or offset must be provided")
        if enforce is None:
            enforce = np.full(self.dimension, False)
        elif isinstance(enforce, bool):
            enforce = np.full(self.dimension, enforce, dtype=bool)
        if offset is not None:
            _offset = np.asarray(offset).reshape(self.dimension)
        else: # offset is None
            _offset = np.zeros(self.dimension)
        if _range is not None:
            lower, upper = _range
            lower = np.asarray(lower).reshape(self.dimension) + _offset
            upper = np.asarray(upper).reshape(self.dimension) + _offset
            for idx in range(self.dimension):
                if abs(self.lower[idx] - lower[idx]) > self._EPS and self.enforce[idx]:
                    raise ValueError("Cannot change enforced lower bound.")
                if abs(self.upper[idx] - upper[idx]) > self._EPS and self.enforce[idx]:
                    raise ValueError("Cannot change enforced upper bound.")
                self._lower = lower
                self._upper = upper
                self.enforce[idx] = np.logical_or(self.enforce[idx], enforce[idx])

            self._lower = lower
            self._upper = upper
        if offset is not None:
            self._offset = np.asarray(offset).reshape(self.dimension)

        # mark the bounds that were user set;
        # use logical_or to combine the user set flags
        if enforce is None:
            enforce = np.full(self.dimension, False)
        _enforce = np.asarray(enforce).reshape(self.dimension)
        self._enforce = np.logical_or(self.enforce, _enforce)

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
        return self.subtree_root.collapse_transformer()

    @property
    def leaf_is_dynamic(self):
        """To check which of the leaves are dynamic, we use the collapsed
        transformer.
        We first take the transformer, and create a new transformer without any
        biases. We then boolify this weights transformer and feed in a vector of
        true boolean values. The resulting vector will be true for all dynamic
        leaves."""
        is_dynamic = []
        for leaf in self.leaves:
            dummy_transformer = self.subtree_root.transformer_to_descendent(leaf)
            dummy_transformer.biases = np.zeros_like(dummy_transformer.biases)
            dummy_transformer = dummy_transformer.boolify()
            input = np.full(
                dummy_transformer.in_dim, dtype=bool, fill_value=True
            )
            dummy_arr = np.asarray(dummy_transformer(input))
            is_dynamic.append(dummy_arr)
        return np.array(is_dynamic, dtype=bool)


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

    @property
    def leaves(self):
        unordered_leaves = super().leaves
        indices_by_base_scatterers = np.array([
            self.base_scatterers.scatterers.index(leaf.base_scatterer)
            for leaf in unordered_leaves
        ])
        return np.array(unordered_leaves)[indices_by_base_scatterers]

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

    @property
    def collapsed_transformer_scatterer_order(self):
        """Return the collapsed transformer in the order of the base scatterers."""
        transformers = [self.subtree_root.transformer_to_descendent(leaf)
                for leaf in self.leaves]
        return stack_transformers(transformers)

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
