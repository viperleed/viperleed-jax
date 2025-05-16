"""Module parameter_space."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-09-02'

import numpy as np

from .constants import ATOM_Z_DIR_ID
from .files.displacements.reader import DisplacementFileSections
from .transformation_tree import (
    geo_parameters,
    meta_parameters,
    occ_parameters,
    vib_parameters,
)
from .transformation_tree.displacement_tree_layers import DisplacementTreeLayers


class ParameterSpace:
    def __init__(self, atom_basis, rpars):
        self._displacements_applied = False
        self.atom_basis = atom_basis

        # create the meta parameter subtree
        self.meta_param_subtree = meta_parameters.MetaParameterSubtree()
        # read the meta parameters from the rpars
        self.meta_param_subtree.read_from_rpars(rpars)

        # create the parameter subtrees - this automatically sets up all the
        # symmetry constraints
        self.vib_tree = vib_parameters.VibTree(atom_basis)
        self.geo_tree = geo_parameters.GeoTree(atom_basis)
        self.occ_tree = occ_parameters.OccTree(atom_basis)

        self.subtrees = (
            self.meta_param_subtree,
            self.geo_tree,
            self.vib_tree,
            self.occ_tree,
        )

    def apply_displacements(self, offset_block=None, search_block=None):
        """
        Parse the search block from the displacements file.

        Parameters
        ----------
        search_block
        """
        if self._displacements_applied:
            raise ValueError('Displacements have already been applied.')

        if offset_block is None and search_block is None:
            raise ValueError(
                'Either offset_block or search_block must be provided.'
            )

        if offset_block is not None:
            # parse and set the offsets
            self._parse_offsets(offset_block)

        if search_block is not None:
            # parse and set the bounds & check for symmetry violations
            self._parse_bounds(search_block)

            # first, parse the constraints
            self._parse_constraints(search_block)

        # apply the implicit constraints & create the subtree root
        for subtree in (self.geo_tree, self.vib_tree, self.occ_tree):
            subtree.apply_implicit_constraints()
            subtree.finalize_tree()

        self._displacements_applied = True

    def check_for_inconsistencies(self):
        """
        Check for inconsistencies in the parameter space.

        This method checks for inconsistencies in the parameter space, such as
        symmetry violations, and raises a ValueError if any are found.
        """
        for subtree in (self.geo_tree, self.vib_tree, self.occ_tree):
            subtree.check_for_inconsistencies()

    def _parse_offsets(self, offsets_block):
        """
        Parse the offsets block from the displacements file.

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
                msg = f'Unknown offset type: {line.offset_type}'
                raise ValueError(msg)
        self.check_for_inconsistencies()

    def _parse_bounds(self, search_block):
        # parse geo, vib and occ bounds
        geo_block = search_block.sections[DisplacementFileSections.GEO_DELTA]
        vib_block = search_block.sections[DisplacementFileSections.VIB_DELTA]
        occ_block = search_block.sections[DisplacementFileSections.OCC_DELTA]

        for subtree, block in zip(
            (self.geo_tree, self.vib_tree, self.occ_tree),
            (geo_block, vib_block, occ_block),
        ):
            for line in block:
                # apply and check for symmetry violations
                subtree.apply_bounds(line)
        self.check_for_inconsistencies()

    def _parse_constraints(self, search_block):
        """
        Parse constraints from the displacements file.

        Parameters
        ----------
        constrain_block
        """
        constraints_block = search_block.sections[
            DisplacementFileSections.CONSTRAIN
        ]
        for constraint in constraints_block:
            if constraint.constraint_type == 'geo':  # TODO: make into Enum
                self.geo_tree.apply_explicit_constraint(constraint)
            elif constraint.constraint_type == 'vib':
                self.vib_tree.apply_explicit_constraint(constraint)
            elif constraint.constraint_type == 'occ':
                self.occ_tree.apply_explicit_constraint(constraint)
            else:
                raise ValueError(
                    f'Unknown constraint type: {constraint.constraint_type}'
                )
        self.check_for_inconsistencies()

    def freeze(self):
        if not self._displacements_applied:
            raise ValueError('Displacements must be applied before freezing.')
        return FrozenParameterSpace(self)

    def _free_params_up_to_layer(self, layer):
        """Return the number of free parameters in all trees up to a layer."""
        free_params = []
        for subtree in self.subtrees:
            layer_roots = subtree.roots_up_to_layer(layer)
            free_params.append(int(sum(node.dof for node in layer_roots)))
        return free_params

    @property
    def atoms_ref_z_position(self):
        """Return the reference z positions of the atoms in the atom basis."""
        return np.array(
                [bs.atom.cartpos[ATOM_Z_DIR_ID] for bs in self.atom_basis]
            )

    @property
    def all_displacements_transformer(self):
        return self.geo_tree.all_displacements_transformer

    @property
    def dynamic_displacements_transformers(self):
        return self.geo_tree.dynamic_displacements_transformers()

    @property
    def all_vib_amps_transformer(self):
        return self.vib_tree.all_vib_amps_transformer

    @property
    def dynamic_t_matrix_transformers(self):
        return self.vib_tree.dynamic_t_matrix_transformers()

    def occ_weight_transformer(self):
        return self.occ_tree.collapsed_transformer_scatterer_order

    @property
    def v0r_transformer(self):
        return self.meta_param_subtree.collapsed_transformer

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

        Returns:
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

        Returns:
            int: The total number of base parameters.
        """
        return sum(self._free_params_up_to_layer(DisplacementTreeLayers.Base))

    @property
    def n_atom_basis(self):
        return len(self.atom_basis)

    @property
    def n_dynamic_propagators(self):
        return self.geo_tree.n_dynamic_propagators

    @property
    def n_static_propagators(self):
        return self.geo_tree.n_static_propagators

    @property
    def propagator_map(self):
        return self.geo_tree.propagator_map

    @property
    def propagator_plane_symmetry_operations(self):
        return self.geo_tree.leaf_plane_symmetry_operations

    @property
    def site_elements(self):
        return self.atom_basis.site_elements

    @property
    def static_propagator_inputs(self):
        return self.geo_tree.static_propagator_inputs

    @property
    def n_dynamic_t_matrices(self):
        return self.vib_tree.n_dynamic_t_matrices

    @property
    def n_static_t_matrices(self):
        return self.vib_tree.n_static_t_matrices

    @property
    def static_t_matrix_inputs(self):
        return self.vib_tree.static_t_matrix_inputs

    @property
    def dynamic_t_matrix_site_elements(self):
        return self.vib_tree.dynamic_site_elements

    @property
    def t_matrix_map(self):
        return self.vib_tree.t_matrix_map

    @property
    def is_dynamic_t_matrix(self):
        return np.array([val == 'dynamic' for (val, id) in self.t_matrix_map])

    @property
    def is_dynamic_propagator(self):
        return np.array([val == 'dynamic' for (val, id) in self.propagator_map])

    @property
    def is_dynamic_scatterer(self):
        return np.logical_or(
            self.is_dynamic_t_matrix, self.is_dynamic_propagator
        )

    @property
    def t_matrix_id(self):
        return np.array([id for (val, id) in self.t_matrix_map])

    @property
    def propagator_id(self):
        return np.array([id for (val, id) in self.propagator_map])

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
        return np.array(
            [
                self.meta_param_subtree.root.dof,
                self.vib_tree.root.dof,
                self.geo_tree.root.dof,
                self.occ_tree.root.dof,
            ]
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

    def expand_params(self, free_params):
        _free_params = np.asarray(free_params)
        splitter = self.split_free_params()
        v0r_params, vib_params, geo_params, occ_params = splitter(
            np.asarray(_free_params)
        )
        v0r_shift = self.v0r_transformer()(v0r_params)
        vib_amps = self.all_vib_amps_transformer()(vib_params)
        displacements = self.all_displacements_transformer()(geo_params)
        weights = self.occ_weight_transformer()(occ_params)
        return v0r_shift, vib_amps, displacements, weights

    def reference_displacements(self, n_batch_atoms):
        """Return a function that computes displacements for ref. propagators.

        Parameters
        ----------
        n_batch_atoms: int
            Batch size for lax.map.

        Returns
        -------
        fn: Callable
            A function geo_free_params -> displacements.
        """

        def compute(geo_free_params):
            return [
                trafo(geo_free_params)
                for trafo in self.dynamic_displacements_transformers
            ]

        return compute

    def reference_vib_amps(self, n_batch_atoms):
        """Return a function that computes vibrational amplitudes for t-matrices.

        Parameters
        ----------
        n_batch_atoms: int
            Batch size for lax.map.

        Returns
        -------
        fn: Callable
            A function vib_free_params -> vib_amps.
        """

        def compute(vib_free_params):
            return [
                trafo(vib_free_params)
                for trafo in self.dynamic_t_matrix_transformers
            ]

        return compute

    def occ_weights(self):
        """Calculate the occupation weights for all scatters.

        Returns
        -------
        weights: The occupation weights for all scatterers.
        """
        return self.occ_weight_transformer
