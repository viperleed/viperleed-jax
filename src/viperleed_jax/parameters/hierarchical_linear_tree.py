from abc import ABC, abstractmethod
from itertools import compress

import numpy as np
import jax.numpy as jnp
import anytree
from anytree import Node, RenderTree

from viperleed_jax.files.displacements.lines import ConstraintLine
from .linear_transformer import LinearTransformer

# Enable checks for the anytree library – we don't deal with huge trees so this
# should not be a performance issue.
anytree.config.ASSERTIONS = True


class HLNode(Node):
    """Base class for hierarchical linear tree nodes."""

    separator = "/"

    def __init__(self, dof, name=None, parent=None, children=None):
        self.dof = dof  # Number of degrees of freedom
        self._transformer = None  # LinearTransformer object
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

    def __init__(self, dof, name=None, parent=None, children=[]):
        if children:
            raise ValueError("Leaf nodes cannot have children.")
        # initialize bounds
        self.bounds = HLBound(dof)
        super().__init__(dof=dof, name=name, parent=parent)


class HLConstraintNode(HLNode):
    """Base class for hierarchical linear tree constraint nodes.

    Any non-leaf node in the tree is a constraint node. Constraint nodes must
    be initialized with a list of child nodes and their corresponding
    transformers."""

    def __init__(self, dof, name=None, children=[], transformers=None):
        _children = list(children)
        super().__init__(dof=dof, name=name)  # Initialize the base class

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

    def collapse_transformer(self):
        """Iterate through through all descendants, collapsing the transformers."""
        collapsed_transformers = []
        for child in self.children:
            if child.is_leaf:
                collapsed_transformers.append(child.transformer)
            else:
                collapsed_transformers.append(
                    child.transformer.compose(child.collapse_transformer())
                )
        return stack_transformers(collapsed_transformers)

    def collapse_bounds(self):
        """Iterate through all descendants, collapsing the bounds."""
        user_set_bounds, lower_bounds, upper_bounds = [], [], []

        for child in self.children:
            if child.is_leaf:
                user_set_bounds.append(child.bounds.user_set)
                lower_bounds.append(child.bounds.lower)
                upper_bounds.append(child.bounds.upper)
            else:
                _user_set, _lower, _upper = child.collapse_bounds()
                user_set_bounds.extend(_user_set)
                lower_bounds.extend(_lower)
                upper_bounds.extend(_upper)

        return np.hstack(user_set_bounds), np.hstack(lower_bounds), np.hstack(upper_bounds)


class HLOffsetNode(HLConstraintNode):
def stack_transformers(transformers):
    """Stack a list of transformers into a single transformer."""
    weights = np.vstack([transformer.weights for transformer in transformers])
    biases = np.hstack([transformer.biases for transformer in transformers])
    return LinearTransformer(weights, biases, (np.sum([t.out_dim for t in transformers]),))

    """Node representing an offset in the hierarchical linear tree.

    Offsets are any static offset from the reference (refcalc) parameters. These
    can be user defined (in the OFFSETS block of DISPLACEMENTS) or they can be
    the optimized parameters from a previous calculation.
    Offset nodes can only have one child, and may not change the number of
    degrees of freedom (dof). The linear transformers must have identity
    weights, but the bias can be set to any value.
    """

    def __init__(self, children, offset=None, name=None):

        # if more than one child is provided, raise an error
        if len(children) != 1:
            raise ValueError("Offset nodes can only have one child.")
        child = children[0]
        dof = child.dof

        # offset must be a vector of length dof
        if offset is None:
            _offset = np.zeros(dof)
        else:
            _offset = np.asarray(offset).reshape(dof,)
        transformer = LinearTransformer(weights=np.eye(dof),
                                        biases=_offset, out_reshape=(dof,))

        super().__init__(dof=dof, name=name,
                       children=[child], transformers=[transformer])

    # TODO: discuss if updating the offsets should be allowed, or if we should
    # instead regenerate the whole tree when necessary
    def update_offset(self, offset, name=None):
        # check that the offset has the correct shape
        try:
            _offset = np.asarray(offset).reshape(self.dof,)
        except ValueError:
            raise ValueError("Offset must have the same shape as the child dof.")
        new_transformer = LinearTransformer(weights=np.eye(self.dof),
                                            bias=_offset, shape=(self.dof,))
        if name is not None:
            self.name = name
        self.children[0].set_transformer(new_transformer)


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

    def update_range(self, range=None, offset=None, user_set=False):
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
        self._user_set = user_set


    def __repr__(self):
        return f"HLBound(lower={self.lower}, upper={self.upper})"


class ExplicitHLFixConstraint(HLConstraintNode):
    """Implicit fix constraint for hierarchical linear tree.

    This constraint is used to fix the value of a node in the tree to a
    constant value. The value is fixed by setting the bias of the transformer
    of the node to the desired value.
    """

    # TODO
    def __init__(self, node, value):
        super().__init__(dof=node.dof, name=f"Fix {node.name}", children=[node])
        self.transformer.biases = value


class ImplicitHLFixConstraint(HLConstraintNode):
    """Implicit fix constraint for hierarchical linear tree.

    This constraint is used to fix the value of a node in the tree to a
    constant value. The value is fixed by setting the bias of the transformer
    of the node to the desired value.
    This class is used to create a fix constraint implicitly, when a root node
    was not assigned any bounds during the bound propagation step.
    """

    # TODO
    def __init__(self, node, value):
        self.dof = 0
        super().__init__(dof=node.dof, name=f"Fix {node.name}", children=[node])
        self.transformer.biases = value


class ParameterHLSubtree(ABC):
    """Base class representing a subtree in the hierarchical linear tree.

    Subtrees are used to group nodes for a group of parameters (vib, geo, occ,
    V0r). This allows implementing constraints more easily and makes it possible
    to extract additional information from the tree (e.g. propagator
    transformations).
    """

    def __init__(self, base_scatterers):
        self.base_scatterers = base_scatterers
        self.site_elements = self.base_scatterers.site_elements
        self.nodes = []

        self._subtree_root_has_been_created = False
        self._offsets_have_been_added = False
        self.build_subtree()

    def __repr__(self):
        if not self._subtree_root_has_been_created:
            partial_trees = [RenderTree(root).by_attr() for root in self.roots]
            trees_str = "\n".join(partial_trees)

            return (f"{trees_str}")
        return RenderTree(self.subtree_root).by_attr()

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

    def apply_bounds(self, vib_line):
        targets = vib_line.targets
        range = vib_line.range
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
            leaf.bounds.update_range(
                range=(range.start, range.stop), user_set=True
            )

    def _add_offset_nodes(self, generic_name):
        """Add offset nodes to the tree."""
        # TODO: mark the offset layer?
        if self._offsets_have_been_added:
            raise ValueError("Offset nodes have already been added.")

        for node in self.roots:
            self.nodes.append(HLOffsetNode(children=[node],
                                           name=generic_name))
        self._offsets_have_been_added = True

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
        selected_leaves, selected_roots = self._target_nodes(targets)
        return selected_leaves, selected_roots

    def create_subtree_root(self):
        """Create a root node that aggregates all root nodes in the subtree.y"""
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
            weights[:, cum_node_dof : cum_node_dof+node.dof] = np.identity(
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
        )
        self.nodes.append(self.subtree_root)
