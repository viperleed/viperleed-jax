"""Module derived_quantities."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-06-24'


from abc import ABC, abstractmethod

import numpy as np
from anytree.walker import Walker, WalkError

from viperleed_jax.lib.matrix import off_diagonal_frobenius
from viperleed_jax.transformation_tree.linear_transformer import LinearMap

from .tree import LinearTree


class DerivedQuantity(ABC):
    """Base class for derived quantities.

    Derived quantities are quantities that are computed based on the
    transformation tree that take an irreducible input vector as input.
    Outputs are typically either one value per leaf node or a single value for
    the entire tree.
    """

    def __init__(self, tree):
        if  not isinstance(tree, LinearTree):
            raise TypeError('tree must be an instance of LinearTree')
        if not tree.finalized:
            raise ValueError(
                'tree must be finalized before creating derived quantities')
        self.tree = tree

    @abstractmethod
    def __call__(self, params):
        pass


class PropagatedQuantity(DerivedQuantity):
    """Base class for all transformable properties.

    Tree propagated functions are functions of the quantity described in the
    transformation tree (e.g. geometrical displacements) that need to be
    evaluated for every leaf node. By using relationships in the tree, we can
    use the values of some leaf nodes to calculate the values of other leaf
    thereby reducing the number of required expensive function evaluations.
    The tree is effectively  statically analyzed and used as a type of
    computation graph for the propagated function.

    Parameters
    ----------
    name : str
        The name of the propagated property function.
    transformer_class : type, optional
        The (super)class of transformers that can be used to propagate the
        function. This can be used to restrict the types of transformations
        that can be applied to the property.
    node_requirement : callable, optional
        A callable that takes a node as an argument and returns True if the
        node can be used to propagate the property. This can be used to restrict
        the types of nodes that can be used for sharing the property.


    Returns
    -------
    dynamic_reference_nodes : (Nodes)
        Leaf nodes that serve as reference values for all dynamically calculated
        values. The expensive calculations only need to be performed for these
        nodes. Empty if no values are dynamic.
    static_reference_nodes : (Nodes)
        Leaf nodes that serve as reference values for all statically calculated
        values. The expensive calculations only need to be performed for these
        nodes. Empty if no values are static.
    ...
    _arg_transformers : Tuple(Transformer)
        A tuple of transformers, one for each leaf node. The transformers
        transform the values of the reference nodes to the values for each leaf.
    func_transformers : Tuple(Transformer)
        A tuple of transformers, one for each leaf node. The transformers
        transform the values of the reference nodes to the values for each leaf.
    """

    def __init__(self, tree, name, transformer_class=None, node_requirement=None):
        super().__init__(tree)
        self.name = name
        self.transformer_class = transformer_class
        self.node_requirement = (
            node_requirement
            if node_requirement is not None
            else lambda node: True
        )
        self._analyze_tree()

    @property
    def n_independent_values(self):
        """Return the number of independent values of the property."""
        return self.n_dynamic_values + self.n_static_values

    @property
    def n_dynamic_values(self):
        """Return the number of indep. dynamic values of the transformable."""
        return len(self.dynamic_reference_nodes)

    @property
    def n_static_values(self):
        """Return the number of indep. static values of the transformable."""
        return len(self.static_reference_nodes)

    def _analyze_tree(self):
        # Step 1) Map all leaves to their shared origin
        self.leaf_to_origin_map = {
            leaf: self._get_shared_origin(leaf) for leaf in self.tree.leaves
        }
        # Step 2) Map all origin nodes to a reference leaf node
        self.origins = list(
            dict.fromkeys(self.leaf_to_origin_map.values()).keys()
        )
        origins_to_reference_map = {
            origin: sorted(origin.leaves, key=self._node_sorting_key)[0]
            for origin in self.origins
        }
        # Step 3) Map all leaves to a reference leaf node
        self.leaf_to_reference_map = {
            leaf: origins_to_reference_map[self.leaf_to_origin_map[leaf]]
            for leaf in self.tree.leaves
        }

        self._arg_transformers = [
            self._transformation_from_to(reference, leaf)
            for leaf, reference in self.leaf_to_reference_map.items()
        ]
        self.dynamic_reference_nodes = tuple(
            leaf
            for leaf in origins_to_reference_map.values()
            if leaf in self.tree.leaves[self.tree.leaf_is_dynamic]
        )
        self.static_reference_nodes = tuple(
            leaf
            for leaf in origins_to_reference_map.values()
            if leaf in self.tree.leaves[~self.tree.leaf_is_dynamic]
        )

        transformers_to_static_reference_leaves = [
            self.tree.root.transformer_to_descendent(static_ref_node)
            for static_ref_node in self.static_reference_nodes
        ]
        self.static_reference_nodes_values = np.array(
            [
                transformer(np.full(transformer.in_dim, 0.5))
                for transformer in transformers_to_static_reference_leaves
            ]
        )

        self.static_dynamic_map = [
            (
                (
                    'static',
                    self.static_reference_nodes.index(
                        self.leaf_to_reference_map[leaf]
                    ),
                )
                if not dynamic
                else (
                    'dynamic',
                    self.dynamic_reference_nodes.index(
                        self.leaf_to_reference_map[leaf]
                    ),
                )
            )
            for leaf, dynamic in zip(self.tree.leaves, self.tree.leaf_is_dynamic)
        ]

    def _can_propagate_up(self, node):
        if not node.parent or not node.transformer:
            return False
        if not isinstance(node.transformer, self.transformer_class):
            return False
        if not node.transformer.is_injective:
            return False
        return self.node_requirement(node)

    def _get_shared_origin(self, node):
        shared_origin = node
        while self._can_propagate_up(shared_origin):
            shared_origin = shared_origin.parent
        return shared_origin

    @abstractmethod
    def _transformation_from_to(self, source, target):
        """Return the transformation from source to target."""

    @abstractmethod
    def _node_sorting_key(self, node):
        """Return a sorting key for the nodes in the tree."""


class LinearPropagatedQuantity(PropagatedQuantity):
    """Base class for transformables on linear transformation trees."""

    def __init__(
        self, tree, name, transformer_class=LinearMap, node_requirement=None
    ):
        self.walker = Walker()
        if not issubclass(transformer_class, LinearMap):
            msg = 'The transformer class must be a subclass of LinearMap.'
            raise TypeError(msg)
        super().__init__(tree, name, transformer_class, node_requirement)

    def _transformation_from_to(self, source, target):
        if not source.is_leaf or not target.is_leaf:
            raise ValueError('Both source and target must be leaf nodes.')
        if source == target:
            return LinearMap(np.eye(source.dof))
        try:
            (upwards, _, downwards) = self.walker.walk(source, target)
        except WalkError as err:
            msg = f'Node {target} cannot be reached from {source}.'
            raise RuntimeError(msg) from err

        # operations up to origin
        up_transformers = [up.transformer for up in upwards]
        down_transformers = [down.transformer for down in downwards]

        # get the symmetry operations
        up_operations = [
            np.linalg.pinv(trafo.weights) for trafo in up_transformers
        ]
        down_operations = [trafo.weights for trafo in down_transformers]

        # combine the operations
        operations = up_operations + down_operations
        # they must be applied in reverse order
        operations.reverse()
        return LinearMap(np.linalg.multi_dot(operations))

    def _node_sorting_key(self, node):
        """Return a sorting key for the nodes in the tree."""
        return off_diagonal_frobenius(node.transformer.weights)
