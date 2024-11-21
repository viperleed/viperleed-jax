"""Module parameter_space."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-10-07'

from abc import ABC, abstractmethod
from itertools import compress

import anytree
import numpy as np
from anytree import RenderTree
from anytree.exporter import UniqueDotExporter

from viperleed_jax.files.displacements.lines import ConstraintLine
from viperleed_jax.parameters.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.parameters.linear_tree_nodes import (
    ImplicitLinearConstraintNode,
    LinearConstraintNode,
)

from .linear_transformer import LinearTransformer, stack_transformers

# Enable checks for the anytree library; we don't deal with huge trees so this
# should not be a performance issue.
anytree.config.ASSERTIONS = True


class TransformationTree(ABC):
    """Abstract base class for a transformation tree."""

    def __init__(self):
        self.nodes = []
        self._tree_root_has_been_created = False
        self.build_tree()

    @abstractmethod
    def build_tree(self):
        """Build the tree."""

    @property
    def roots(self):
        """Return all root nodes in the tree."""
        return [node for node in self.nodes if node.is_root]

    @property
    def leaves(self):
        """Return all leaf nodes in the tree."""
        return [node for node in self.nodes if node.is_leaf]

    @property
    @abstractmethod
    def name(self):
        """Return the name of the tree."""

    @property
    @abstractmethod
    def subtree_root_name(self):
        """Return the name of the tree root node."""

    @abstractmethod
    def create_subtree_root(self):
        """Create a root node that aggregates all root nodes in the subtree."""

    def graphical_export(self, filename):
        """Create and save a graphical representation of the tree to file."""
        if not self._tree_root_has_been_created:
            raise ValueError('Subtree root has not yet been created.')
        UniqueDotExporter(self.subtree_root).to_picture(filename)


class InvertibleTransformationTree(TransformationTree):
    """Abstract base class for an invertible transformation tree."""

    def __init__(self):
        super().__init__()


class LinearTree(InvertibleTransformationTree):
    """Represents a transformation tree where all transformations are linear."""

    def __init__(self):
        super().__init__()

    def create_subtree_root(self):
        """Create a root node that aggregates all root nodes in the subtree."""
        if self._tree_root_has_been_created:
            raise ValueError('Subtree root has already been created.')
        self._tree_root_has_been_created = True
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
        """Return a string representation of the tree."""
        if not self._tree_root_has_been_created:
            partial_trees = [RenderTree(root).by_attr() for root in self.roots]
            trees_str = '\n'.join(partial_trees)

            return f'{trees_str}'
        return RenderTree(self.subtree_root).by_attr()

    def roots_up_to_layer(self, layer):
        """Return all root nodes up to a given layer."""
        _layer = DisplacementTreeLayers(layer)
        return [
            node
            for node in self.nodes
            if node.layer.value <= _layer.value
            and (node.is_root or node.parent.layer.value > _layer.value)
        ]

    def collapsed_transformer(self):
        return self.subtree_root.collapse_transformer()

    @property
    def leaf_is_dynamic(self):
        """Return an array indicating which leaves are dynamic.

        We first take the transformer, and create a new transformer without any
        biases. We then boolify this weights transformer and feed in a vector of
        true boolean values. The resulting vector will be true for all dynamic
        leaves.
        """
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
