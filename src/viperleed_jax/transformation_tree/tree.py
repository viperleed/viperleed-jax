"""Module parameter_space."""

__authors__ = ('Alexander M. Imre (@amimre)',
               'Florian Kraushofer (@fkraushofer)',
               )
__created__ = '2024-10-07'
__copyright__ = 'Copyright (c) 2023-2025 ViPErLEED developers'
__license__ = 'GPLv3+'


import logging
from abc import ABC, abstractmethod
from enum import IntEnum
from itertools import compress

import anytree
import numpy as np
from anytree import RenderTree
from anytree.exporter import UniqueDotExporter
from viperleed.calc.classes.perturbation_mode import PerturbationMode
from viperleed.calc.files.new_displacements.lines import ConstraintLine

from viperleed_jax.lib.math import EPS
from viperleed_jax.lib.matrix import closest_to_identity, xyz_matrix_to_zxy
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.nodes import (
    ImplicitLinearConstraintNode,
    LinearConstraintNode,
    LinearOffsetNode,
)

from .linear_transformer import AffineTransformer, LinearMap, stack_transformers
from .reduced_space import Zonotope, ZonotopeNotOrthogonalError

# Enable checks for the anytree library; we don't deal with huge trees so this
# should not be a performance issue.
anytree.config.ASSERTIONS = True

logger = logging.getLogger(__name__)


class ConstructionOrder(IntEnum):
    """Enum for the construction order of transformation trees."""

    SYMMETRY = 0
    EXPLICIT_CONSTRAINT = 1
    OFFSET = 2
    BOUNDS = 3
    IMPLICIT_CONSTRAINT = 4
    ROOT = 5


class ZonotopeAccumulator:
    def __init__(self):
        self._accumulated = {}

    def add(self, key, zonotope, name):
        if key not in self._accumulated:
            self._accumulated[key] = (zonotope, name)
            return
        # otherwise add the zonotopes and names
        existing_zonotope, existing_name = self._accumulated[key]
        try:
            combined_zonotope = existing_zonotope.add_orthogonal_same_center(
                zonotope
            )
        except ZonotopeNotOrthogonalError as e:
            msg = (
                f'Cannot combine range zonotope from "{name}" with existing '
                f'zonotope from "{existing_name}": {e}. Ensure that the '
                'specified ranges are orthogonal to each other.'
            )
            raise ValueError(msg) from e
        combined_name = f'{existing_name}; {name}'
        self._accumulated[key] = (combined_zonotope, combined_name)

    def __iter__(self):
        """Iterate over the accumulated zonotopes."""
        return iter(self._accumulated.items())


class TransformationTree(ABC):
    """Abstract base class for a transformation tree."""

    def __init__(self, name, root_node_name):
        self.nodes = []
        self.name = name
        self.root_node_name = root_node_name
        self._current_construction_order = ConstructionOrder.SYMMETRY
        self._initialize_tree()

    @abstractmethod
    def _initialize_tree(self):
        """Set up the tree."""

    def _check_construction_order(self, order):
        """Check if the tree is in the correct construction order."""
        if self._current_construction_order == ConstructionOrder.ROOT:
            raise ValueError(
                'Tree has already been finalized. No further modifications '
                'are allowed.'
            )
        if order < self._current_construction_order:
            msg = (
                f'Cannot apply {order.name} after '
                f'{self._current_construction_order.name}.'
            )
            raise ValueError(msg)
        if order > self._current_construction_order:
            self._current_construction_order = order
            logger.debug(f'{self.name}: Construction order {order.name}.')

    @abstractmethod
    def apply_explicit_constraint(self):
        """Apply an explicit constraint to the tree."""
        self._check_construction_order(ConstructionOrder.EXPLICIT_CONSTRAINT)

    @abstractmethod
    def apply_offsets(self):
        """Apply offsets to the tree."""
        self._check_construction_order(ConstructionOrder.OFFSET)

    @abstractmethod
    def apply_bounds(self):
        """Apply bounds to the tree."""
        self._check_construction_order(ConstructionOrder.BOUNDS)

    @abstractmethod
    def apply_implicit_constraints(self):
        """Apply implicit constraints to the tree."""
        self._check_construction_order(ConstructionOrder.IMPLICIT_CONSTRAINT)

    @property
    def finalized(self):
        """Return whether the tree has been finalized."""
        return self._current_construction_order == ConstructionOrder.ROOT

    @abstractmethod
    def finalize_tree(self):
        """Finish setting up the tree.

        Called after all user constraints have been applied. Applies implicit
        constraints and creates the root node.
        """
        self._create_root()

    @property
    def roots(self):
        """Return all root nodes in the tree."""
        return [node for node in self.nodes if node.is_root]

    @property
    def leaves(self):
        """Return all leaf nodes in the tree."""
        return [node for node in self.nodes if node.is_leaf]

    @abstractmethod
    def _create_root(self):
        """Create a root node that aggregates all root nodes in the subtree."""
        self._check_construction_order(ConstructionOrder.ROOT)

    def graphical_export(self, filename):
        """Create and save a graphical representation of the tree to file."""
        if not self.finalized:
            raise ValueError('Subtree root has not yet been created.')
        # Left-to-right orientation looks better for broad trees like we have
        UniqueDotExporter(self.root, options=['rankdir=LR']).to_picture(
            filename,
        )


class InvertibleTransformationTree(TransformationTree):
    """Abstract base class for an invertible transformation tree."""

    def __init__(self, name, root_node_name):
        super().__init__(name, root_node_name)


class LinearTree(InvertibleTransformationTree):
    """Represents a transformation tree where all transformations are linear."""

    def __init__(self, name, root_node_name):
        super().__init__(name, root_node_name)

    def finalize_tree(self):
        """Finish setting up the tree."""
        super().finalize_tree()

    def _create_root(self):
        """Create a root node that aggregates all root nodes in the subtree."""
        if self.finalized:
            raise ValueError('Subtree root has already been created.')
        if not self.roots:
            raise ValueError('No root nodes found in subtree.')
        for root in self.roots:
            all_ancestors = [root, *root.ancestors]
            if not any(
                isinstance(node, ImplicitLinearConstraintNode)
                for node in all_ancestors
            ):
                msg = (
                    'Implicit constraints have not been applied for all roots.'
                )
                raise ValueError(msg)

        root_dof = sum(node.dof for node in self.roots)
        transformers = []
        cum_node_dof = 0
        for node in self.roots:
            weights = np.zeros((node.dof, root_dof))
            weights[:, cum_node_dof : cum_node_dof + node.dof] = np.identity(
                node.dof
            )
            bias = np.zeros(node.dof)
            transformers.append(AffineTransformer(weights, bias, (node.dof,)))
            cum_node_dof += node.dof
        self.root = LinearConstraintNode(
            dof=root_dof,
            name=self.root_node_name,
            children=self.roots,
            transformers=transformers,
            layer=DisplacementTreeLayers.Root,
        )
        self.nodes.append(self.root)
        super()._create_root()

    def __repr__(self):
        """Return a string representation of the tree."""
        if not self.finalized:
            partial_trees = [RenderTree(root).by_attr() for root in self.roots]
            trees_str = '\n'.join(partial_trees)

            return f'{trees_str}'
        return RenderTree(self.root).by_attr()

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
        return self.root.collapse_transformer()

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
            dummy_transformer = self.root.transformer_to_descendent(leaf)
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

    def __init__(self, atom_basis, name, root_node_name, perturbation_type):
        self.atom_basis = atom_basis
        self.site_elements = self.atom_basis.site_elements

        self._offsets_have_been_added = False
        self.perturbation_mode = PerturbationMode(perturbation_type)

        self._zonotope_accumulator = ZonotopeAccumulator()
        super().__init__(name, root_node_name)

    @property
    def leaves(self):
        """Nodes that are leaves, i.e., that have no children."""
        unordered_leaves = super().leaves
        indices_by_atom_basis = np.array(
            [
                self.atom_basis.scatterers.index(leaf.atom)
                for leaf in unordered_leaves
            ]
        )
        return np.array(unordered_leaves)[indices_by_atom_basis]

    def apply_implicit_constraints(self):
        """Apply implicit constraints to the tree."""
        super().apply_implicit_constraints()
        for root in self.roots:
            all_ancestors = [root, *root.ancestors]
            # check if the root has an implicit constraint
            if any(
                isinstance(node, ImplicitLinearConstraintNode)
                for node in all_ancestors
            ):
                continue

            # if not, top it off with an implicit fixed constraint
            implicit_fixed_zonotope = Zonotope(
                basis=np.array([]).reshape(root.dof, 0),
                ranges=np.array([]).reshape(2, 0),
            )
            implicit_fixed_constraint = ImplicitLinearConstraintNode(
                child=root,
                name='Implicit fixed',
                child_zonotope=implicit_fixed_zonotope,
            )
            self.nodes.append(implicit_fixed_constraint)

    def _initialize_tree(self):
        # create leaf nodes
        geo_leaf_nodes = [self._leaf_node(atom) for atom in self.atom_basis]
        self.nodes.extend(geo_leaf_nodes)

        # apply symmetry constraints
        for siteel in self.site_elements:
            site_el_leaves = [
                node for node in self.leaves if node.site_element == siteel
            ]
            for link in self.atom_basis.symmetry_links:
                # put all linked atoms in the same symmetry group
                nodes_to_link = [
                    node for node in site_el_leaves if node.atom in link
                ]
                if not nodes_to_link:
                    continue
                symmetry_node = self._symmetry_node(children=nodes_to_link)
                self.nodes.append(symmetry_node)

        unlinked_site_el_nodes = [node for node in self.leaves if node.is_root]
        for node in unlinked_site_el_nodes:
            self.nodes.append(self._symmetry_node(children=[node]))

    @property
    def _raw_leaf_transformers(self):
        """Return the raw leaf transformers in the order of the base scatterers."""
        transformers = [
            self.root.transformer_to_descendent(leaf) for leaf in self.leaves
        ]
        return stack_transformers(transformers)

    def __call__(self, reduced_params):
        """Apply the transformation tree to the given parameters."""
        if not self.finalized:
            raise ValueError('Subtree root has not yet been created.')
        # Post-process the values to apply the transformations
        return self._post_process_values(
            self._raw_leaf_transformers(reduced_params)
        )

    def _post_process_values(self, raw_values):
        return raw_values

    def _target_nodes(self, targets):
        """Take a BSTarget and returns the corresponding leaves and roots."""
        # gets the leaves that are affected by the targets
        explicitly_selected_leaves = list(
            compress(self.leaves, targets.select(self.atom_basis))
        )
        if not explicitly_selected_leaves:
            msg = f'No leaf nodes found for target {targets}.'
            raise ValueError(msg)

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

    def _get_leaves_and_roots(self, targets):
        """Return the leaves, roots and primary_leaves for the given targets."""
        all_target_leaves = self.leaves[
            self.atom_basis.target_selection_mask(targets)
        ]
        target_roots = [leaf.root for leaf in all_target_leaves]
        target_roots = {
            root: _select_primary_leaf(root, all_target_leaves)
            for root in target_roots
        }
        return all_target_leaves, target_roots

    def apply_offsets(self, offset_line):
        """Apply offsets to the children of the node."""
        offset = np.array(offset_line.offset.offset)

        # TODO: allow off axis offsets; multi dim offsets?

        # if there is a direction do processing
        if offset_line.direction is not None:
            if len(offset_line.direction.vectors_zxy) != 1:
                msg = f'Cannot interpret offset for line "{offset_line}".'
                raise ValueError(msg)
            vector = offset_line.direction.vectors_zxy[0]
            offset = np.array(vector) * offset

        # check construction order
        super().apply_offsets()
        # get roots targeted by the offset
        _, target_roots_primary_leaves = self._get_leaves_and_roots(
            offset_line.targets
        )

        # iterate over the roots and apply the offset
        for root, primary_leaf in target_roots_primary_leaves.items():
            if root.dof != offset.size:
                msg = (
                    f'Offset line "{offset_line}" has a size of {offset.size} '
                    f'but the target has {root.dof} DOFs. The offset must be of '
                    f'shape ({root.dof},).'
                )
                raise ValueError(msg)

            # check that offset is not yet defined
            if any(
                isinstance(node, LinearOffsetNode)
                for node in [root, *root.ancestors]
            ):
                msg = (
                    f'Offset line "{offset_line}" is already defined for '
                    f'{root.name}.'
                )
                raise ValueError(msg)

            # get the transformation from the primary leaf to the root
            inv_trafo = root.transformer_to_descendent(
                primary_leaf
            ).pseudo_inverse()
            # get the offset in the root coordinates
            offset_at_root = inv_trafo(offset)

            # create the offset node
            offset_node = LinearOffsetNode(
                children=[root],
                offset_at_node=offset_at_root,
                name=offset_line.raw_line,
            )
            self.nodes.append(offset_node)

    def apply_explicit_constraint(self, constraint_line):
        r"""Apply an explicit constraint to the tree.

        This method applies explicit, user defined constraints to the tree.
        """
        # check construction order
        self._check_construction_order(ConstructionOrder.EXPLICIT_CONSTRAINT)

        # check if the constraint line is of the correct type
        _check_constraint_line_type(constraint_line, self.perturbation_mode)

        # resolve the reference (rhs of constraint) into a mask
        link_target_mask = self.atom_basis.target_selection_mask(
            (constraint_line.link_target,)
        )
        # if multiple atoms are targeted, we need to select the first one
        link_target = self.leaves[link_target_mask][0]
        link_target_root = link_target.root

        # select which atoms to link by interpreting the target token
        to_link_mask = self.atom_basis.target_selection_mask(
            constraint_line.targets
        )
        leaves_to_link = self.leaves[to_link_mask]
        # get the roots of the leaves to link
        roots_to_link = [leaf.root for leaf in leaves_to_link]
        # remove the link_target from the list of roots to link
        roots_to_link = [
            root for root in roots_to_link if root != link_target_root
        ]
        # remove duplicates
        roots_to_link = list({root: None for root in roots_to_link}.keys())

        # if there are no roots to link, complain
        if len(roots_to_link) == 0:
            msg = (
                'All targets of CONSTRAIN block line '
                f'"{constraint_line.raw_line}" are already linked. It is '
                'likely redundant.'
            )
            raise ValueError(msg)

        # check that the roots all have the same number of DOFs
        if not all(root.dof == link_target_root.dof for root in roots_to_link):
            msg = (
                f'CONSTRAIN block line "{constraint_line.raw_line}" links '
                'atoms with different DOFs. This means it either violates '
                'symmetry or is in contradiction with other constraints.'
            )
            raise ValueError(msg)

        user_arr = constraint_line.linear_operation.arr

        # TODO: If we allow for non (3x3) transformations, we need to check
        # casting here.
        # check that the linear transformation given by the user is valid
        if user_arr.size == 1:
            # scalar case
            user_trafo = float(user_arr.flatten()[0]) * np.eye(link_target.dof)
        elif user_arr.shape == (link_target.dof, link_target.dof):
            user_trafo = user_arr
            # 3-dimensional linear operations from user input have coordinates
            # order xyz, transform them to zxy order:
            if link_target.dof == 3:
                user_trafo = LinearMap(xyz_matrix_to_zxy(user_trafo))
        else:
            msg = (
                f'Constraint line "{constraint_line}" has a linear '
                f'transformation of shape {user_arr.shape} but the target has '
                f'{link_target.dof} DOFs. The transformation must be of shape '
                f'({link_target.dof}, {link_target.dof}).'
            )
            raise ValueError(msg)

        transformations = []

        # iterate over the roots to link and determine the transformations
        for root in roots_to_link:
            # get primary leaf of the root
            primary_target_leaf = _select_primary_leaf(root, leaves_to_link)
            # map transformation between the leaves to the root
            transformations.append(
                _map_transformation_from_leaf_to_root(
                    primary_target_leaf, link_target, user_trafo
                )
            )

        transformations = [LinearMap(np.eye(link_target.dof)), *transformations]
        children = [link_target_root, *roots_to_link]

        constraint_node = LinearConstraintNode(
            dof=link_target.dof,
            name=constraint_line.raw_line,
            children=children,
            transformers=transformations,
            layer=DisplacementTreeLayers.User_Constraints,
        )
        self.nodes.append(constraint_node)

    @abstractmethod
    def _zonotope_from_bounds_line(self, bounds_line, primary_leaf):
        """Create a zonotope for the leaf node from a bounds line."""

    def apply_bounds_line(self, bounds_line):
        # resolve targets
        _, target_roots_and_primary_leaves = self._get_leaves_and_roots(
            bounds_line.targets
        )

        for root, primary_leaf in target_roots_and_primary_leaves.items():
            leaf_zonotope = self._zonotope_from_bounds_line(
                bounds_line, primary_leaf
            )
            # get the transformation from the primary leaf to the root
            root_to_leaf_transformer = root.transformer_to_descendent(
                primary_leaf
            )
            leaf_to_root_transformer = root_to_leaf_transformer.pseudo_inverse()
            root_range_zonotope = leaf_zonotope.apply_affine(
                leaf_to_root_transformer
            )
            self._zonotope_accumulator.add(
                key=root,
                zonotope=root_range_zonotope,
                name=bounds_line.raw_line,
            )

    def apply_bounds(self):
        """Apply bounds to the children of the node."""
        super().apply_bounds()

        # apply the accumulated zonotopes
        for root, (zonotope, name) in self._zonotope_accumulator:
            # double check that no implicit constraint exists yet
            root_and_ancestors = [root, *root.ancestors]
            for ancestor in root_and_ancestors:
                if isinstance(ancestor, ImplicitLinearConstraintNode):
                    msg = (
                        f'{self.perturbation_mode} implicit constraint/'
                        f'boundary "{name}" is in conflict with '
                        f'"{ancestor.name}". Only one displacement range '
                        'may be defined per set of linked parameters.'
                    )
                    raise TypeError(msg)

            # from the zonotope, create and apply an implicit constraint node
            implicit_constraint_node = ImplicitLinearConstraintNode(
                child=root,
                name=name,
                child_zonotope=zonotope,
            )
            self.nodes.append(implicit_constraint_node)

    @abstractmethod
    def is_centered(self):
        """Check if the tree is centered."""
        if not self.finalized:
            raise ValueError(
                'Tree must be finalized before checking if it is centered.'
            )

    @property
    @abstractmethod
    def ref_calc_values(self):
        """Return the reference values for the tree."""
        if not self.finalized:
            raise ValueError(
                'Tree must be finalized before getting reference values.'
            )

    def get_parameter_names(self):
        if not self.finalized:
            raise ValueError(
                'Tree must be finalized before generating parameter names.'
            )

        leaf_transformers = [
            self.root.transformer_to_descendent(leaf) for leaf in self.leaves
        ]
        # use weights only, since biases are irrelevant for the connection
        leaf_transformers_weights = [
            transformer.weights for transformer in leaf_transformers
        ]
        parameter_names = []
        for root_id in range(self.root.dof):
            test_array = np.zeros((self.root.dof,))
            test_array[root_id] = 1.0
            mask = [
                np.any(transformer_weights @ test_array != 0.0)
                for transformer_weights in leaf_transformers_weights
            ]
            root_param_leaves = list(compress(self.leaves, mask))
            primary_leaf = _select_primary_leaf(self.root, root_param_leaves)
            parameter_names.append(
                self._parameter_name(primary_leaf, test_array)
            )
        return parameter_names

    @abstractmethod
    def _parameter_name(self, leaf, test_array):
        if not self.finalized:
            raise ValueError(
                'Tree must be finalized before generating parameter names.'
            )
        if leaf not in self.leaves:
            raise ValueError('Leaf not found in this tree')

    def reference_parameters(self):
        """Return the parameter values that recreate the reference state.

        The reference state is combination of parameters that recreate the
        reference structure of the system (as close as possible).
        """
        if not self.finalized:
            raise ValueError(
                'Tree must be finalized before getting reference parameters.'
            )

        # get the raw leaf transformers and invert
        raw_trafo = self._raw_leaf_transformers
        if raw_trafo.in_dim == 0:
            # if there are no free parameters, return empty array
            return np.array([])

        pseudo_inverse = raw_trafo.pseudo_inverse()

        # get closest to the reference occupations
        ref_parameters = pseudo_inverse(self.ref_calc_values)
        leaf_values = self.__call__(ref_parameters)

        if np.sum(abs(leaf_values - self.ref_calc_values)) > EPS:
            raise ValueError(
                'Reference parameters do not recreate the reference values. '
            )
        return ref_parameters


def _map_transformation_from_leaf_to_root(
    primary_leaf, secondary_leaf, transformation
):
    r"""Map a transformation from leaf level to root level.

    Map a given linear transformation from one leaf to another into a
    transformation between their root nodes. In many cases, we need to determine
    what the transformation between two leaves is in terms of the root
    transformations. We are given two leafs with values p and p' and the
    transformation between them is given by the transformation matrix A such
    that p'=Ap. We need to find the transformations B and B' between the root
    nodes of the two leaves as shown in the figure below where T and T' are
    intermediate transformations.

                 ( )
             B   / \   B'
                /   \
              ( )   ( )
           T  / \   / \  T'
             /         \
           ( )         ( )
            p           p'

    Since the transformations between the leaves are equivalent, we find
    L = T^{+} B^{+} B' T'.
    We can choose the transformation on the left side (the primary leaf) to be
    the identity matrix. Inserting this, we find
    B' = T L T'^{+}.
    """
    primary_root = primary_leaf.root
    secondary_root = secondary_leaf.root

    T = primary_root.transformer_to_descendent(primary_leaf)
    T_dash = secondary_root.transformer_to_descendent(secondary_leaf)
    return T @ LinearMap(transformation) @ T_dash.pseudo_inverse()


def _select_primary_leaf(root, leaves):
    root_leaves = [leaf for leaf in root.leaves if leaf in leaves]
    if not root_leaves:
        # if no leaf is found, raise an error
        msg = f'None of the provided leaves matches {root}.'
        raise ValueError(msg)

    primary_leaf_id = closest_to_identity(
        [leaf.transformer.weights for leaf in root_leaves]
    )
    return root_leaves[primary_leaf_id]


def _check_constraint_line_type(constraint_line, expected_perturbation_mode):
    """Check if the constraint line is of the expected type."""
    if not isinstance(constraint_line, ConstraintLine):
        msg = 'Constraint must be a ConstraintLine.'
        raise TypeError(msg)
    if constraint_line.mode_token.mode != expected_perturbation_mode:
        msg = (
            f'Wrong constraint type for {expected_perturbation_mode} '
            f'parameter: {constraint_line.mode_token}.'
        )
        raise ValueError(msg)
