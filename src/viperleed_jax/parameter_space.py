"""Module parameter_space."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-09-02'

from collections import namedtuple
from copy import deepcopy

import jax
import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from .files.displacements.reader import DisplacementFileSections
from .transformation_tree import (
    geo_parameters,
    meta_parameters,
    occ_parameters,
    vib_parameters,
)
from .transformation_tree.displacement_tree_layers import DisplacementTreeLayers

_ATOM_Z_DIR_ID = 2
_DISP_Z_DIR_ID = 0


class ParameterSpace:
    def __init__(self, base_scatterers, rpars):
        self._displacements_applied = False
        self.base_scatterers = base_scatterers

        # create the meta parameter subtree
        self.meta_param_subtree = meta_parameters.MetaParameterSubtree()
        # read the meta parameters from the rpars
        self.meta_param_subtree.read_from_rpars(rpars)

        # create the parameter subtrees - this automatically sets up all the
        # symmetry constraints
        self.vib_tree = vib_parameters.VibTree(base_scatterers)
        self.geo_tree = geo_parameters.GeoTree(base_scatterers)
        self.occ_tree = occ_parameters.OccTree(base_scatterers)

        self.subtrees = (
            self.meta_param_subtree,
            self.geo_tree,
            self.vib_tree,
            self.occ_tree,
        )

        # atom-site-element reference z positions
        self._ats_ref_z_pos = jnp.array(
            [bs.atom.cartpos[_ATOM_Z_DIR_ID] for bs in self.base_scatterers]
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
                'Either offset_block or search_block must be ' 'provided.'
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
            subtree.create_root()

        self._displacements_applied = True
        self.geo_tree.get_functionals()

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
                raise ValueError('Unknown offset type: ' f'{line.offset_type}')
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
                    'Unknown constraint type: ' f'{constraint.constraint_type}'
                )
        self.check_for_inconsistencies()

    def freeze(self):
        if not self._displacements_applied:
            raise ValueError('Displacements must be applied before freezing.')
        return FrozenParameterSpace(self)

    def _free_params_up_to_layer(self, layer):
        """Return the number of free parameters in all subtrees up to a given
        layer."""
        free_params = []
        for subtree in self.subtrees:
            layer_roots = subtree.roots_up_to_layer(layer)
            free_params.append(int(sum(node.dof for node in layer_roots)))
        return free_params

    @property
    def all_displacements_transformer(self):
        return self.geo_tree.all_displacements_transformer

    @property
    def dynamic_displacements_transformers(self):
        return self.geo_tree.dynamic_displacements_transformers()

    @property
    def all_vib_amps_transformer(self):
        return self.vib_tree.all_vib_amps_transformer()

    @property
    def dynamic_t_matrix_transformers(self):
        return self.vib_tree.dynamic_t_matrix_transformers()

    @property
    def occ_weight_transformer(self):
        return self.occ_tree.collapsed_transformer_scatterer_order

    @property
    def v0r_transformer(self):
        return self.meta_param_subtree.collapsed_transformer()

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
    def n_base_scatterers(self):
        return len(self.base_scatterers)

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
        return self.base_scatterers.site_elements

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
        return np.arange(self.n_base_scatterers)[self.is_dynamic_scatterer]

    @property
    def static_scatterer_id(self):
        return np.arange(self.n_base_scatterers)[~self.is_dynamic_scatterer]

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


@register_pytree_node_class
class FrozenParameterSpace:
    frozen_attributes = (
        '_ats_ref_z_pos',
        'all_displacements_transformer',
        'all_vib_amps_transformer',
        'dynamic_displacements_transformers',
        'dynamic_scatterer_id',
        'dynamic_scatterer_id',
        'dynamic_scatterer_propagator_id',
        'dynamic_scatterer_t_matrix_id',
        'dynamic_t_matrix_site_elements',
        'dynamic_t_matrix_transformers',
        'info',
        'is_dynamic_scatterer',
        'is_dynamic_propagator',
        'is_dynamic_t_matrix',
        'n_base_params',
        'n_base_scatterers',
        'n_dynamic_scatterers',
        'n_dynamic_propagators',
        'n_dynamic_t_matrices',
        'n_free_params',
        'n_param_split',
        'n_static_scatterers',
        'n_static_propagators',
        'n_static_t_matrices',
        'n_symmetry_constrained_params',
        'occ_weight_transformer',
        'propagator_id',
        'propagator_map',
        'propagator_plane_symmetry_operations',
        'site_elements',
        'static_scatterer_id',
        'static_propagator_inputs',
        'static_t_matrix_inputs',
        't_matrix_id',
        't_matrix_map',
        'v0r_transformer',
    )

    def __init__(self, parameter_space):
        # take all the information from the parameter space and
        # convert it to an immutable object.
        for attr in self.frozen_attributes:
            copied_attr = deepcopy(getattr(parameter_space, attr))
            # if the attribute is a numpy array, convert it to a jax array
            if isinstance(copied_attr, jnp.ndarray):
                copied_attr = jnp.asarray(copied_attr)
            setattr(self, attr, copied_attr)

    def split_free_params(self, free_params):
        if len(free_params) != self.n_free_params:
            raise ValueError('Number of free parameters does not match.')
        v0r_params = free_params[: self.n_param_split[0]]
        vib_params = free_params[
            self.n_param_split[0] : sum(self.n_param_split[:2])
        ]
        geo_params = free_params[
            sum(self.n_param_split[:2]) : sum(self.n_param_split[:3])
        ]
        occ_params = free_params[sum(self.n_param_split[:3]) :]
        return v0r_params, vib_params, geo_params, occ_params

    def expand_params(self, free_params):
        v0r_params, vib_params, geo_params, occ_params = self.split_free_params(
            free_params
        )
        v0r_shift = self.v0r_transformer(v0r_params)
        vib_amps = self.all_vib_amps(vib_params)
        displacements = self.all_displacements(geo_params)
        weights = self.occ_weight_transformer(occ_params)
        return v0r_shift, vib_amps, displacements, weights

    def reference_displacements(self, geo_free_params):
        """Calculate the displacements for all reference propagators.

        Parameters
        ----------
        geo_free_params: The geometric free parameters.

        Returns
        -------
        displacements: The displacements for all reference propagators.
        """
        return [
            trafo(geo_free_params)
            for trafo in self.dynamic_displacements_transformers
        ]

    def reference_vib_amps(self, vib_free_params):
        """Calculate the vibrational amplitudes for all reference t-matrices.

        Parameters
        ----------
        vib_free_params: The vibrational free parameters.

        Returns
        -------
        vib_amps: The vibrational amplitudes for all reference t-matrices.
        """
        return [
            trafo(vib_free_params)
            for trafo in self.dynamic_t_matrix_transformers
        ]

    def occ_weights(self, occ_free_params):
        """Calculate the occupation weights for all scatters.

        Parameters
        ----------
        occ_free_params: The occupation free parameters.

        Returns
        -------
        weights: The occupation weights for all scatterers.
        """
        return self.occ_weight_transformer(occ_free_params)

    def all_displacements(self, geo_free_params):
        """Calculate the displacements for all propagators.

        Parameters
        ----------
        geo_free_params: The geometric free parameters.

        Returns
        -------
        displacements: The displacements for all propagators.
        """
        return self.all_displacements_transformer()(geo_free_params)

    def all_vib_amps(self, vib_free_params):
        """Calculate the vibrational amplitudes for all t-matrices.

        Parameters
        ----------
        vib_free_params: The vibrational free parameters.

        Returns
        -------
        vib_amps: The vibrational amplitudes for all t-matrices.
        """
        return self.all_vib_amps_transformer(vib_free_params)

    def potential_onset_height_change(self, geo_free_params):
        """Calculate the change in the highest atom z position.

        This is needed because the onset height of the inner potential is
        defined as the z position of the highest atom in the slab.
        Therefore, changes to this height may change refraction of the incoming
        electron wave.

        Parameters
        ----------
        geo_free_params: The geometric free parameters.

        Returns
        -------
        float: The difference between the new highest atom z position and the
            highest reference z position.
        """
        z_changes = self.all_displacements(geo_free_params)[:, _DISP_Z_DIR_ID]
        new_z_pos = self._ats_ref_z_pos + z_changes
        # find the difference between the new highest atom z position and the
        # highest reference z position
        return jnp.max(new_z_pos) - jnp.max(self._ats_ref_z_pos)

    def tree_flatten(self):
        aux_data = {
            attr: getattr(self, attr) for attr in self.frozen_attributes
        }
        children = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, children, aux_data):
        frozen_parameter_space = cls.__new__(cls)
        for kw, value in aux_data.items():
            setattr(frozen_parameter_space, kw, value)
        return frozen_parameter_space
