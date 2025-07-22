"""Module parameter_space."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-09-02'

import logging
import numpy as np
from viperleed.calc.classes.perturbation_mode import PerturbationMode

logger = logging.getLogger(__name__)

from viperleed_jax.transformation_tree import (
    geo_parameters,
    meta_parameters,
    occ_parameters,
    vib_parameters,
)
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)


class ParameterSpace:
    """Parameter space for the LEED calculator.

    TODO

    Notes
    -----
    Since the transformations in the tree are linear, and the user has to
    specify bounds for all parameters, the center of the parameter space may not
    be 'centered' around the point of the reference calculation (three points
    are not necessarily on a straight line!). This is can be true even if no
    explicit offsets are given.
    For an example, consider an alloy that (in the) reference calculation has a
    25/75% occupation of two elements. If the user specifies bounds of 0-100%
    for the two elements, the center of the parameter space will be at 50%
    occupation of both elements, which is not the same as the reference
    calculation!"""

    def __init__(self, atom_basis, rpars):
        self._displacements_applied = False
        self.atom_basis = atom_basis

        # create the meta parameter subtree
        self.meta_tree = meta_parameters.MetaTree()
        # set V0r bounds from Rparams object
        self.meta_tree.apply_bounds(rpars)
        # apply roots – TODO: move to meta tree class methods
        self.meta_tree._create_root()

        # create the structural trees - this automatically sets up all the
        # symmetry constraints
        self.vib_tree = vib_parameters.VibTree(atom_basis)
        self.geo_tree = geo_parameters.GeoTree(atom_basis)
        self.occ_tree = occ_parameters.OccTree(atom_basis)

        self.trees = (
            self.meta_tree,
            self.geo_tree,
            self.vib_tree,
            self.occ_tree,
        )

    def apply_search_segment(self, search_block):
        """
        Parse the search block from the displacements file.

        Parameters
        ----------
        search_block
        """
        if self._displacements_applied:
            raise ValueError('Displacements have already been applied.')

        # first, parse the constraints
        self._parse_constraints(search_block)

        # parse and set the bounds & check for symmetry violations
        self._parse_bounds(search_block)

        # apply the implicit constraints & create the subtree root
        for subtree in (self.geo_tree, self.vib_tree, self.occ_tree):
            subtree.apply_implicit_constraints()
            subtree.finalize_tree()

        self._displacements_applied = True

        # check if subtrees are centered
        for subtree in (self.geo_tree, self.vib_tree, self.occ_tree):
            if not subtree.is_centered():
                logger.warning('Parameter space is not centered.')

    def apply_offsets(self, offsets_block):
        """
        Parse the offsets block from the DISPLACEMENTS file.

        Parameters
        ----------
        offsets_block
        """
        for line in offsets_block.lines:
            if line.offset_type == 'geo':
                self.geo_tree.apply_offsets(line)
            elif line.offset_type == 'vib':
                self.vib_tree.apply_offsets(line)
            elif line.offset_type == 'occ':
                self.occ_tree.apply_offsets(line)
            else:
                msg = f'Unknown offset mode: {line.offset_type}'
                raise ValueError(msg)

    def _parse_bounds(self, search_block):
        """Parse bounds from the DISPLACEMENTS file."""
        # Geometric bounds
        for line in search_block.geo_delta_lines:
            self.geo_tree.apply_bounds(line)
        # Vibration bounds
        for line in search_block.vib_delta_lines:
            self.vib_tree.apply_bounds(line)
        # Occupation bounds
        for line in search_block.occ_delta_lines:
            self.occ_tree.apply_bounds(line)

    def _parse_constraints(self, search_block):
        """Parse constraints from the DISPLACEMENTS file."""
        for constraint_line in search_block.explicit_constraint_lines:
            if constraint_line.mode.mode is PerturbationMode.GEO:
                self.geo_tree.apply_explicit_constraint(constraint_line)
            elif constraint_line.mode.mode is PerturbationMode.VIB:
                self.vib_tree.apply_explicit_constraint(constraint_line)
            elif constraint_line.mode.mode is PerturbationMode.OCC:
                self.occ_tree.apply_explicit_constraint(constraint_line)
            elif constraint_line.mode.mode is PerturbationMode.DOM:
                raise NotImplementedError(
                    'Domain constraints are not supported yet.'
                )
            else:
                msg = f'Unknown constraint mode: {constraint_line.mode}'
                raise ValueError(msg)

    def _free_params_up_to_layer(self, layer):
        """Return the number of free parameters in all trees up to a layer."""
        free_params = []
        for subtree in self.trees:
            layer_roots = subtree.roots_up_to_layer(layer)
            free_params.append(int(sum(node.dof for node in layer_roots)))
        return free_params

    @property
    def dynamic_displacements_transformers(self):
        return self.geo_tree.dynamic_displacements_transformers()

    @property
    def v0r_transformer(self):
        return self.meta_tree.collapsed_transformer

    @property
    def n_free_params(self):
        """Returns the total number of free parameters in the parameter space."""
        if not self._displacements_applied:
            raise ValueError(
                'Displacements must be applied before counting '
                'free parameters.'
            )
        return sum(self._free_params_up_to_layer(DisplacementTreeLayers.Root))

    @property
    def n_user_constrained_params(self):
        """Returns the total number of free parameters in the parameter space."""
        if not self._displacements_applied:
            raise ValueError(
                'Displacements must be applied before counting '
                'user constrained parameters.'
            )
        return sum(
            self._free_params_up_to_layer(
                DisplacementTreeLayers.User_Constraints
            )
        )

    @property
    def n_symmetry_constrained_params(self):
        """Returns the total number of symmetry constrained parameters.

        This method calculates the total number of symmetry constrained
        parameters by summing up the number of symmetry constrained
        parameters from different parameter subtrees.

        Returns
        -------
            int: The total number of symmetry constrained parameters.
        """
        return sum(
            self._free_params_up_to_layer(DisplacementTreeLayers.Symmetry)
        )

    @property
    def n_base_params(self):
        """Returns the total number of base parameters.

        This method calculates the sum of the number of base parameters from
        different parameter subtrees.

        Returns
        -------
            int: The total number of base parameters.
        """
        return sum(self._free_params_up_to_layer(DisplacementTreeLayers.Base))

    @property
    def n_atom_basis(self):
        return len(self.atom_basis)

    @property
    def propagator_plane_symmetry_operations(self):
        return self.geo_tree.leaf_plane_symmetry_operations

    @property
    def site_elements(self):
        return self.atom_basis.site_elements

    @property
    def is_dynamic_scatterer(self):
        return np.logical_or(
            self.is_dynamic_t_matrix, self.is_dynamic_propagator
        )

    @property
    def dynamic_scatterer_id(self):
        return np.arange(self.n_atom_basis)[self.is_dynamic_scatterer]

    @property
    def static_scatterer_id(self):
        return np.arange(self.n_atom_basis)[~self.is_dynamic_scatterer]

    @property
    def dynamic_scatterer_propagator_id(self):
        return self.propagator_id[self.is_dynamic_scatterer]

    @property
    def dynamic_scatterer_t_matrix_id(self):
        return self.t_matrix_id[self.is_dynamic_scatterer]

    @property
    def n_dynamic_scatterers(self):
        return np.sum(self.is_dynamic_scatterer)

    @property
    def n_static_scatterers(self):
        return np.sum(~self.is_dynamic_scatterer)

    @property
    def n_param_split(self):
        return tuple(
            sum(root.dof for root in tree.roots) for tree in self.trees
        )

    @property
    def info(self):
        """
        Return a string containing information about the parameter space size.

        Returns a string containing information about the number of free
        parameters, user constrained parameters, symmetry constrained
        parameters, and total parameters.

        Returns
        -------
            str: Information about the parameters.
        """
        n_root_params = self._free_params_up_to_layer(
            DisplacementTreeLayers.Root
        )
        n_user_params = self._free_params_up_to_layer(
            DisplacementTreeLayers.User_Constraints
        )
        n_sym_params = self._free_params_up_to_layer(
            DisplacementTreeLayers.Symmetry
        )
        n_base_params = self._free_params_up_to_layer(
            DisplacementTreeLayers.Base
        )

        def format(n_params):
            return (
                f'({n_params[0]} V0r, '
                f'{n_params[1]} geo, '
                f'{n_params[2]} vib, '
                f'{n_params[3]} occ)'
            )

        return (
            'Free parameters (implicit constraints):'
            f'\n{self.n_free_params}\t{format(n_root_params)}\n'
            'User constrained parameters:'
            f'\n{self.n_user_constrained_params}\t{format(n_user_params)}\n'
            'Symmetry constrained parameters:'
            f'\n{self.n_symmetry_constrained_params}\t{format(n_sym_params)}\n'
            'Total parameters:\n'
            f'{self.n_base_params}\t{format(n_base_params)}\n'
        )

    def split_free_params(self):
        """Return a function to split free parameters into categories.

        Returns
        -------
        fn: Callable
            A function free_params -> (v0r_params, vib_params, geo_params, occ_params)
        """
        n_split = self.n_param_split
        n_total = self.n_free_params

        def compute(free_params):
            if len(free_params) != n_total:
                raise ValueError('Number of free parameters does not match.')
            i0 = 0
            i1 = n_split[0]
            i2 = i1 + n_split[1]
            i3 = i2 + n_split[2]
            v0r_params = free_params[i0:i1]
            vib_params = free_params[i1:i2]
            geo_params = free_params[i2:i3]
            occ_params = free_params[i3:]
            return v0r_params, vib_params, geo_params, occ_params

        return compute
