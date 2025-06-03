"""Module linear_tree_nodes."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-21'

# TODO: rename base to atoms


from abc import abstractmethod

import numpy as np
from anytree import Node
from anytree.walker import Walker

from viperleed_jax.lib_math import EPS
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.errors import (
    InvalidNodeError,
    TransformationTreeError,
)
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
    LinearMap,
    stack_transformers,
)
from viperleed_jax.transformation_tree.reduced_space import (
    orthonormalize_subspace,
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
        self._name = name
        self._transformer = None
        self.parent = parent
        if children:
            self.children = children

    @property
    def transformer(self):
        """Transformer of the edge connecting this node to its parent."""
        if self._transformer is None:
            raise InvalidNodeError('Node does not have a transformer.')
        return self._transformer

    @abstractmethod
    def set_transformer(self, transformer):
        """Set the transformer describing the edge."""

    def _pre_detach(self):
        raise TransformationTreeError(
            'Transformation trees do not support detaching nodes.'
        )


class LinearTreeNode(TransformationTreeNode):
    """Base class for hierarchical linear tree nodes."""

    def __init__(self, dof, layer, name=None, parent=None, children=None):
        self.dof = dof  # Number of degrees of freedom
        self._transformer = None  # LinearTransformer object
        self.layer = DisplacementTreeLayers(layer)

        super().__init__(name=name, parent=parent, children=children)

    @property
    def name(self):
        """Return the name of the node."""
        return f'({self.dof}) {self._name}' if self._name else f'({self.dof})'

    def set_transformer(self, transformer):
        """Set transformer for the edge connecting this node to its parent."""
        # check if the transformer is valid
        if not isinstance(transformer, AffineTransformer):
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
                f'Parent must be an instance of LinearConstraintNode. '
                f'Invalid parent: {parent}'
            )
            raise TypeError(msg)

        # check that the parent layer is >= child layer
        if not parent.layer.value >= self.layer.value:
            msg = (
                f'Parent layer ({parent.layer}) must be greater or equal '
                f'to child layer ({self.layer}).'
            )
            raise InvalidNodeError(msg)

        # check that the transformer dimensions match
        if self.transformer.in_dim != parent.dof:
            msg = (
                f'Transformer input dimension ({self.transformer.in_dim}) '
                f'must match parent dof ({parent.dof}).'
            )
            raise InvalidNodeError(msg)


class LinearLeafNode(LinearTreeNode):
    """Base class for leaf nodes for the linear transformation tree."""

    def __init__(self, dof, name=None, parent=None):
        # initialize bounds
        super().__init__(
            dof=dof, name=name, parent=parent, layer=DisplacementTreeLayers.Base
        )
        # self.ref_calc_value = np.array(ref_calc_value)
        # if self.ref_calc_value.shape != (self.dof,):
        #     msg = (
        #         f'Reference calculation value shape {self.ref_calc_value.shape} '
        #         f'does not match node dof shape {self.dof}.'
        #     )
        #     raise ValueError(msg)


class AtomicLinearNode(LinearLeafNode):
    """Base class for leaf nodes representing parameters indexed by atom."""

    def __init__(self, dof, atom, name=None, parent=None):
        # base scatterer based attributes
        self.atom = atom
        self.element = atom.site_element.element
        self.num = atom.num
        self.site = atom.site_element.site
        self.site_element = atom.site_element
        super().__init__(dof=dof, name=name, parent=parent)


class LinearConstraintNode(LinearTreeNode):
    """Base class for hierarchical linear tree constraint nodes.

    Any non-leaf node in the tree is a constraint node. Constraint nodes must
    be initialized with a list of child nodes and their corresponding
    transformers.
    """

    def __init__(self, dof, layer, name=None, children=[], transformers=None):
        _children = list(children)
        super().__init__(
            dof=dof, name=name, layer=layer
        )  # Initialize the base class

        if len(_children) == 0:
            raise InvalidNodeError(
                'Constraint nodes must be initialized with '
                'at least one child node.'
            )
        # check that children are roots
        if any(not child.is_root for child in _children):
            raise InvalidNodeError('Children must be root nodes.')

        # if one child is a leaf, all children must be leaves
        if any(child.is_leaf for child in _children):
            if not all(child.is_leaf for child in _children):
                raise InvalidNodeError(
                    'If one child is a leaf node, all children '
                    'must be leaf nodes.'
                )

        # dof cannot be larger than the sum of the children's dofs
        if self.dof > sum(child.dof for child in _children):
            raise InvalidNodeError(
                'Degree of freedom must be less than or equal to '
                "the sum of the children's degrees of freedom."
            )

        # # only one child can be assigned a bound
        # if sum(child.is_bounded for child in _children) > 1:
        #     raise ValueError(
        #         'Cannot connect two or more child nodes that have been '
        #         'assigned bounds.'
        #     )

        # if no transformers are provided, check if all children already
        # have transformers
        if transformers is None:
            if any(child.transformer is None for child in _children):
                raise ValueError('All children must have transformers.')
            transformers = [child.transformer for child in _children]

        # check that the number of children and transformers match
        if len(_children) != len(transformers):
            raise InvalidNodeError(
                f'Number of children ({len(_children)}) must match '
                f'number of transformers ({len(transformers)}).'
            )

        # if everything is fine, set the transformers
        for child, transformer in zip(_children, transformers):
            child.set_transformer(transformer)

        # if that went well, set the children
        for child in _children:
            child.parent = self


    def transformer_to_descendent(self, node):
        """Return the transformer from this node to a descendent."""
        walker = Walker()
        try:
            (upwards, common, downwards) = walker.walk(self, node)
        except walker.WalkError as err:
            msg = f'Node {node} cannot be reached from {self}.'
            raise ValueError(msg) from err
        if upwards:
            msg = f'Node {node} is not a descendent of {self}.'
            raise ValueError(msg)
        transformers = [node.transformer for node in downwards]
        composed_transformer = transformers[0]
        for trafo in transformers[1:]:
            composed_transformer = composed_transformer.compose(trafo)
        return composed_transformer

    def down_collapse_transformers(self, stop_condition):
        if stop_condition is None:

            def stop_condition(node):
                return node.is_leaf

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


class LinearOffsetNode(LinearConstraintNode):
    """Class representing offsets in the hierarchical linear tree.

    Offsets are used to represent static displacements of parameters in the tree
    (that are dealt with in the tensor-LEED approximation) rather than treated
    in the reference calculation.
    Offsets may be placed in between symmetry and user constraints.
    """

    def __init__(self, children, offset_at_node, name):
        # can only have one child
        if len(children) != 1:
            raise ValueError('Offset node must have exactly one child')
        child = children[0]

        # check that no offset node is in the tree yet
        if any(isinstance(c, LinearOffsetNode) for c in child.ancestors):
            raise ValueError('Only one offset node is allowed in the tree.')

        # check that the offset is valid
        offset = np.array(offset_at_node)
        if offset.shape != (child.dof,):
            msg = (
                f'Offset shape {offset.shape} does not match child dof '
                f'shape {child.dof}.'
            )
            raise ValueError(msg)

        # create affine transformer for the offset
        offset_trafo = AffineTransformer(
            weights = np.eye(child.dof),
            biases = offset,
            out_reshape=(child.dof,)
        )
        # set the transformer for the child
        super().__init__(
            dof=child.dof,
            name=f'Offset({name})',
            children=[child],
            transformers=[offset_trafo],
            layer=DisplacementTreeLayers.Offsets,
        )


class ImplicitLinearConstraintNode(LinearConstraintNode):
    """Class representing implicit constraints in the hierarchical linear tree.

    Implicit constraints are constraints that are not explicitly defined by the
    user, but rather by not explicitly allowing certain degrees of freedom.
    If e.g. a group of symmetry equivalent atoms is not assigned a displacement
    range, they are assumed to be static and will be constrained through an
    implicit constraint node.
    Implicit constraints are above user constraints in the hierarchy and are the
    last layer before the subtree root node.

    Parameters
    ----------
    children: [LinearConstraintNode]
        A list containing the child of the node to be created. Will raise a
        ValueError if more then one child is given.
    name: str
        Label to be used by the node.
    child_basis: np.ndarray (n_basis, n_dof)
        Basis vectors for the child node. The basis vectors are used to
    child_ranges: np.ndarray (n_dof, 2)
    """

    def __init__(self, child, name, child_zonotope):

        # can only have one child
        if not isinstance(child, LinearConstraintNode):
            raise TypeError('Child must be a linear node.')

        if any(isinstance(a, ImplicitLinearConstraintNode)
               for a in [child, *child.ancestors]):
            raise ValueError('Only one implicit constraint node is allowed '
                             'per tree.')

        normalization_transformer = child_zonotope.normalize()

        super().__init__(
            dof=normalization_transformer.in_dim,
            name=name,
            children=[child],
            transformers=[normalization_transformer],
            layer=DisplacementTreeLayers.Implicit_Constraints,
        )
