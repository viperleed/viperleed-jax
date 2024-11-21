"""Module parameter_space."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-10-07'

from abc import ABC, abstractmethod
from enum import Enum
from itertools import compress

import anytree
import numpy as np
from anytree import RenderTree
from anytree.exporter import UniqueDotExporter

from viperleed_jax.files.displacements.lines import ConstraintLine
from viperleed_jax.parameters.linear_tree_nodes import (
    ImplicitLinearConstraintNode,
    LinearConstraintNode,
)

from .linear_transformer import LinearTransformer, stack_transformers

# Enable checks for the anytree library; we don't deal with huge trees so this
# should not be a performance issue.
anytree.config.ASSERTIONS = True


DisplacementTreeLayers = Enum(
    'HLTreeLayers',
    [
        'Base',
        'Symmetry',
        'Backend_Constraints',
        'User_Constraints',
        'Implicit_Constraints',
        'Root',
    ],
)


class DisplacementRange:
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
        self.update_range(
            _range=(np.zeros(dimension), np.zeros(dimension)),
            offset=np.zeros(dimension),
        )

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
            raise ValueError('range or offset must be provided')
        if enforce is None:
            enforce = np.full(self.dimension, False)
        elif isinstance(enforce, bool):
            enforce = np.full(self.dimension, enforce, dtype=bool)
        if offset is not None:
            _offset = np.asarray(offset).reshape(self.dimension)
        else:  # offset is None
            _offset = np.zeros(self.dimension)
        if _range is not None:
            lower, upper = _range
            lower = np.asarray(lower).reshape(self.dimension) + _offset
            upper = np.asarray(upper).reshape(self.dimension) + _offset
            for idx in range(self.dimension):
                if (
                    abs(self.lower[idx] - lower[idx]) > self._EPS
                    and self.enforce[idx]
                ):
                    raise ValueError('Cannot change enforced lower bound.')
                if (
                    abs(self.upper[idx] - upper[idx]) > self._EPS
                    and self.enforce[idx]
                ):
                    raise ValueError('Cannot change enforced upper bound.')
                self._lower = lower
                self._upper = upper
                self.enforce[idx] = np.logical_or(
                    self.enforce[idx], enforce[idx]
                )

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
        return f'HLBound(lower={self.lower}, upper={self.upper})'


# TODO: abstraction Tree -> invertible tree -> linear tree


class LinearTree(ABC):  # TODO: further abstract to a tree
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
            raise ValueError('Subtree root has already been created.')
        self._subtree_root_has_been_created = True
        if not self.roots:
            raise ValueError('No root nodes found in subtree.')
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
        self.subtree_root = LinearConstraintNode(
            dof=root_dof,
            name=self.subtree_root_name,
            children=self.roots,
            transformers=transformers,
            layer=DisplacementTreeLayers.Root,
        )
        self.nodes.append(self.subtree_root)

    def __repr__(self):
        if not self._subtree_root_has_been_created:
            partial_trees = [RenderTree(root).by_attr() for root in self.roots]
            trees_str = '\n'.join(partial_trees)

            return f'{trees_str}'
        return RenderTree(self.subtree_root).by_attr()

    def roots_up_to_layer(self, layer):
        _layer = DisplacementTreeLayers(layer)
        return [
            node
            for node in self.nodes
            if node.layer.value <= _layer.value
            and (node.is_root or node.parent.layer.value > _layer.value)
        ]

    def graphical_export(self, filename):
        if not self._subtree_root_has_been_created:
            raise ValueError('Subtree root has not yet been created.')
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
            dummy_transformer = self.subtree_root.transformer_to_descendent(
                leaf
            )
            dummy_transformer.biases = np.zeros_like(dummy_transformer.biases)
            dummy_transformer = dummy_transformer.boolify()
            input = np.full(
                dummy_transformer.in_dim, dtype=bool, fill_value=True
            )
            dummy_arr = np.asarray(dummy_transformer(input))
            is_dynamic.append(np.any(dummy_arr))
        return np.array(is_dynamic, dtype=bool)


class DisplacementTree(LinearTree):
    """Represents a tree handling displacement parameters.

    Trees are used to group nodes for a group of parameters (vib, geo, occ,
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
        indices_by_base_scatterers = np.array(
            [
                self.base_scatterers.scatterers.index(leaf.base_scatterer)
                for leaf in unordered_leaves
            ]
        )
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
        transformers = [
            self.subtree_root.transformer_to_descendent(leaf)
            for leaf in self.leaves
        ]
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
            raise ValueError('Constraint must be a ConstraintLine.')
        if constraint_line.constraint_type != constraint_type:
            raise ValueError(
                f'Constraint must be a {constraint_type} constraint.'
            )

    def _target_nodes(self, targets):
        """Takes a BSTarget and returns the corresponding leaves and roots."""
        # gets the leaves that are affected by the targets
        explicitly_selected_leaves = list(
            compress(self.leaves, targets.select(self.base_scatterers))
        )
        if not explicitly_selected_leaves:
            raise ValueError(f'No leaf nodes found for target {targets}.')

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
            selected_roots,
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
        if constraint_line.value != 'linked':
            raise NotImplementedError('Only linked constraints are supported.')

        targets = constraint_line.targets
        implicit_leaves, explicit_leaves, selected_roots = self._target_nodes(
            targets
        )
        return implicit_leaves, explicit_leaves, selected_roots

    def apply_implicit_constraints(self):
        for root in self.roots:
            implicit_node = ImplicitLinearConstraintNode([root])
            self.nodes.append(implicit_node)
