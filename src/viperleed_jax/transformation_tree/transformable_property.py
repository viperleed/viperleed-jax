from abc import ABC, abstractmethod

import numpy as np
from anytree.walker import Walker, WalkError

from viperleed_jax.transformation_tree.linear_transformer import LinearMap


class Transformable(ABC):
    """Base class for all transformable properties.

    Transformables are functions of the quantity described in the transformation
    tree (e.g. geometrical displacements).
    Transformables can be used to minimize the number of required evaluations of
    expensive functions by enabling sharing the results of the function
    evaluations across the tree.
    Every leaf node in the transformation tree has value for each
    transformable property. By analyzing the tree, we can determine which nodes
    share values for a given transformable property, and how they are related.

    Parameters
    ----------
    name : str
        The name of the property.
    transformer_class : type, optional
        The (super)class of transformers that can be used to propagate the
        transformable property. This can be used to restrict the types of
        transformations that can be applied to the property.
    node_requirement : callable, optional
        A callable that takes a node as an argument and returns True if the
        node can be used to propagate the property. This can be used to restrict
        the types of nodes that can be used for sharing the property.


    # TODO move to method docstrings
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

    def __init__(self, name, transformer_class=None, node_requirement=None):
        self.name = name
        self.transformer_class = transformer_class
        self.node_requirement = (
            node_requirement
            if node_requirement is not None
            else lambda node: True
        )

    # @property
    # def n_independent_values(self):
    #     """Return the number of independent values of the property."""
    #     return self.n_dynamic_values + self.n_static_values

    # @abstractmethod
    # def n_dynamic_values(self):
    #     """Return the number of indep. dynamic values of the transformable."""

    # @abstractmethod
    # def n_static_values(self):
    #     """Return the number of indep. static values of the transformable."""

    def analyze_tree(self, tree):
        # Step 1) Map all leaves to their shared origin
        leaf_to_origin_map = {
            leaf: self._get_shared_origin(leaf) for leaf in tree.leaves
        }
        # Step 2) Map all origin nodes to a reference leaf node
        origins = list(dict.fromkeys(leaf_to_origin_map.values()).keys())
        origins_to_reference_map = {
            origin: sorted(origin.leaves, key=self._node_sorting_key)[0]
            for origin in origins
        }
        # Step 3) Map all leaves to a reference leaf node
        leaf_to_reference_map = {
            leaf: origins_to_reference_map[leaf_to_origin_map[leaf]]
            for leaf in tree.leaves
        }

        transformers_to_reference = [
            self._transformation_from_to(reference, leaf)
            for leaf, reference in leaf_to_reference_map.items()
        ]
        return leaf_to_reference_map, transformers_to_reference

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


class LinearTransformable(Transformable):
    def __init__(
        self, name, transformer_class=LinearMap, node_requirement=None
    ):
        self.walker = Walker()
        if not issubclass(transformer_class, LinearMap):
            msg = 'The transformer class must be a subclass of LinearMap.'
            raise TypeError(msg)
        super().__init__(name, transformer_class, node_requirement)

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
        return np.linalg.multi_dot(operations)

    def _node_sorting_key(self, node):
        """Return a sorting key for the nodes in the tree."""
        return non_diagonality_measure(node.transformer.weights)


def non_diagonality_measure(matrix):
    """
    Compute the non-diagonality measure of a general real matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        A 2D real matrix (can be non-square).

    Returns
    -------
    float
        The Frobenius norm of the off-diagonal part of the matrix.
    """
    # Ensure input is a numpy array
    matrix = np.array(matrix)

    # Create the diagonal projection of the matrix
    diagonal_projection = np.zeros_like(matrix)
    np.fill_diagonal(diagonal_projection, np.diag(matrix))

    # Compute the Frobenius norm of the difference
    difference = matrix - diagonal_projection
    return np.linalg.norm(difference, 'fro')


class DisplacementTransformable(LinearTransformable):
    def __init__(
        self,
    ):
        super().__init__(name='displacement')
