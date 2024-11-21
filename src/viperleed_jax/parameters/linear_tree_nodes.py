"""Module linear_tree_nodes."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-21'

# TODO: rename base to atoms


from abc import abstractmethod

import numpy as np
from anytree import Node
from anytree.walker import Walker

from viperleed_jax.parameters.displacement_range import DisplacementRange
from viperleed_jax.parameters.hierarchical_linear_tree import (
    DisplacementTreeLayers,
)
from viperleed_jax.parameters.linear_transformer import (
    LinearMap,
    LinearTransformer,
    stack_transformers,
)


class TransformationTreeNode(Node):
    """Base class for nodes for transformation trees.

    Transformation trees are used to represent hierarchical and composable
    transformations. Each node in the tree represents data that is related to
    it's parent node through a transformation (the edge).
    The transformation is represented by a transformer that is stored in the
    child node.
    """

    separator = '/'

    def __init__(self, name, parent=None, children=None):
        self.name = name
        self._transformer = None
        self.parent = parent
        if children:
            self.children = children

    @property
    def transformer(self):
        """Transformer of the edge connecting this node to its parent."""
        if self._transformer is None:
            raise ValueError('Node does not have a transformer.')
        return self._transformer

    @abstractmethod
    def set_transformer(self, transformer):
        """Set the transformer describing the edge."""

    def _pre_detach(self):
        raise RuntimeError(
            'Transformation trees do not support detaching nodes.'
        )


class LinearTreeNode(TransformationTreeNode):
    """Base class for hierarchical linear tree nodes."""

    def __init__(self, dof, layer, name=None, parent=None, children=None):
        self.dof = dof  # Number of degrees of freedom
        self._transformer = None  # LinearTransformer object
        self.layer = DisplacementTreeLayers(layer)

        self.name = f'({self.dof}) {name}' if name else f'({self.dof})'
        super().__init__(name=name, parent=parent, children=children)

    def set_transformer(self, transformer):
        # check if the transformer is valid
        if not isinstance(transformer, LinearTransformer):
            msg = (
                f'Transformer must be an instance of LinearTransformer. '
                f'Invalid transformer: {transformer}'
            )
            raise TypeError(msg)

        if transformer.out_dim != self.dof:
            msg = (
                f'Transformer output dimension ({transformer.out_dim}) '
                f'must match child dof ({self.dof}).'
            )
            raise ValueError(msg)
        self._transformer = transformer

    def _pre_attach(self, parent):
        # check that the parent is a ConstraintNode
        if not isinstance(parent, LinearConstraintNode):
            msg = (
                f'Parent must be an instance of HLConstraintNode. '
                f'Invalid parent: {parent}'
            )
            raise TypeError(msg)

        # check that the parent layer is >= child layer
        if not parent.layer.value >= self.layer.value:
            msg = (
                f'Parent layer ({parent.layer}) must be greater or equal '
                f'to child layer ({self.layer}).'
            )
            raise ValueError(msg)

        # check that the transformer dimensions match
        if self.transformer.in_dim != parent.dof:
            msg = (
                f'Transformer input dimension ({self.transformer.in_dim}) '
                f'must match parent dof ({parent.dof}).'
            )
            raise ValueError(msg)


class LinearLeafNode(LinearTreeNode):
    def __init__(self, dof, name=None, parent=None):
        # initialize bounds
        self._bounds = DisplacementRange(dof)
        super().__init__(
            dof=dof, name=name, parent=parent, layer=DisplacementTreeLayers.Base
        )

    def update_bounds(self, line):
        self._update_bounds(line)
        self.parent.check_bounds_valid()

    @abstractmethod
    def _update_bounds(self, line):
        pass

    @property
    def free(self):
        return ~self._bounds.fixed


class AtomicLinearNode(LinearLeafNode):
    def __init__(self, dof, base_scatterer, name=None, parent=None):
        # base scatterer based attributes
        self.base_scatterer = base_scatterer
        self.element = base_scatterer.site_element.element
        self.num = base_scatterer.num
        self.site = base_scatterer.site_element.site
        self.site_element = base_scatterer.site_element
        super().__init__(dof=dof, name=name, parent=parent)


class LinearConstraintNode(LinearTreeNode):
    """Base class for hierarchical linear tree constraint nodes.

    Any non-leaf node in the tree is a constraint node. Constraint nodes must
    be initialized with a list of child nodes and their corresponding
    transformers."""

    def __init__(self, dof, layer, name=None, children=[], transformers=None):
        _children = list(children)
        super().__init__(
            dof=dof, name=name, layer=layer
        )  # Initialize the base class

        if len(_children) == 0:
            raise ValueError(
                'Constraint nodes must be initialized with '
                'at least one child node.'
            )
        # check that children are roots
        if any(not child.is_root for child in _children):
            raise ValueError('Children must be root nodes.')

        # if one child is a leaf, all children must be leaves
        if any(child.is_leaf for child in _children):
            if not all(child.is_leaf for child in _children):
                raise ValueError(
                    'If one child is a leaf node, all children '
                    'must be leaf nodes.'
                )

        # dof cannot be larger than the sum of the children's dofs
        if self.dof > sum(child.dof for child in _children):
            raise ValueError(
                'Degree of freedom must be less than or equal to '
                "the sum of the children's degrees of freedom."
            )

        # if no transformers are provided, check if all children already
        # have transformers
        if transformers is None:
            if any(child.transformer is None for child in _children):
                raise ValueError('All children must have transformers.')
            transformers = [child.transformer for child in _children]

        # check that the number of children and transformers match
        if len(_children) != len(transformers):
            raise ValueError(
                f'Number of children ({len(_children)}) must match '
                f'number of transformers ({len(transformers)}).'
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
                'Bounds are not satisfiable'
            )  # TODO: better error message
        return True

    def _stacked_transformer(self):
        """Return the stacked transformer of the children."""

        child_weights = [child.transformer.weights for child in self.children]
        child_biases = [child.transformer.biases for child in self.children]

        stacked_weights = np.vstack(child_weights)
        stacked_biases = np.hstack(child_biases)
        return LinearTransformer(
            stacked_weights,
            stacked_biases,
            (np.sum([c.dof for c in self.children]),),
        )

    def transformer_to_descendent(self, node):
        """Return the transformer from this node to a descendent."""
        walker = Walker()
        try:
            (upwards, common, downwards) = walker.walk(self, node)
        except walker.WalkError as err:
            raise ValueError(
                f'Node {node} cannot be reached from {self}.'
            ) from err
        if upwards:
            raise ValueError(f'Node {node} is not a descendent of {self}.')
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
                        stack_transformers(
                            child.down_collapse_transformers(stop_condition)
                        )
                    )
                )
        return collapsed_transformers

    def collapse_transformer(self):
        """Iterate through through all descendants, collapsing the transformers."""
        collapsed_transformers = self.down_collapse_transformers(
            stop_condition=None
        )
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

        return (
            np.hstack(enforced_bounds),
            np.hstack(lower_bounds),
            np.hstack(upper_bounds),
        )

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


class ImplicitLinearConstraintNode(LinearConstraintNode):
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
            raise ValueError('Implicit constraints must have exactly one child')
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

        super().__init__(
            dof=dof,
            name=f'Bounds Constraint',
            children=[child],
            transformers=[composed_transformer],
            layer=DisplacementTreeLayers.Implicit_Constraints,
        )
