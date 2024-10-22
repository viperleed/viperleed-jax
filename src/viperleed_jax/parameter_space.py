from collections import namedtuple
from copy import deepcopy

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from viperleed_jax.parameters import occ_parameters
from viperleed_jax.parameters import vib_parameters
from viperleed_jax.parameters import geo_parameters
from viperleed_jax.base_scatterers import BaseScatterers
from viperleed_jax.parameters import meta_parameters
from viperleed_jax.files.displacements.reader import DisplacementFileSections
from viperleed_jax.parameters.hierarchical_linear_tree import HLTreeLayers


_ATOM_Z_DIR_ID = 2
_DISP_Z_DIR_ID = 0


class ParameterSpace():

    def __init__(self, base_scatterers, rpars):
        self._displacements_applied = False
        self.base_scatterers = base_scatterers

        # create the meta parameter subtree
        self.meta_param_subtree = meta_parameters.MetaParameterSubtree()
        # read the meta parameters from the rpars
        self.meta_param_subtree.read_from_rpars(rpars)

        # create the parameter subtrees - this automatically sets up all the
        # symmetry constraints
        self.vib_subtree = vib_parameters.VibHLSubtree(base_scatterers)
        self.geo_subtree = geo_parameters.GeoHLSubtree(base_scatterers)
        self.occ_subtree = occ_parameters.OccHLSubtree(base_scatterers)

        self.subtrees = (
            self.meta_param_subtree,
            self.geo_subtree,
            self.vib_subtree,
            self.occ_subtree,
        )

        # atom-site-element reference z positions
        self._ats_ref_z_pos = jnp.array(
            [bs.atom.cartpos[_ATOM_Z_DIR_ID]
             for bs in self.base_scatterers]
        )

    def apply_displacements(self, offset_block=None, search_block=None):
        """
        Parse the search block from the displacements file.

        Parameters
        ----------
        search_block
        """
        if self._displacements_applied:
            raise ValueError("Displacements have already been applied.")

        if offset_block is None and search_block is None:
            raise ValueError("Either offset_block or search_block must be "
                             "provided.")

        if offset_block is not None:
            # parse and set the offsets
            self._parse_offsets(offset_block)

        if search_block is not None:
            # parse and set the bounds & check for symmetry violations
            self._parse_bounds(search_block)

            # first, parse the constraints
            self._parse_constraints(search_block)

        # apply the implicit constraints & create the subtree root
        for subtree in (self.geo_subtree, self.vib_subtree, self.occ_subtree):
            subtree.apply_implicit_constraints()
            subtree.create_subtree_root()

        self._displacements_applied = True

    def check_for_inconsistencies(self):
        """
        Check for inconsistencies in the parameter space.

        This method checks for inconsistencies in the parameter space, such as
        symmetry violations, and raises a ValueError if any are found.
        """
        for subtree in (self.geo_subtree, self.vib_subtree, self.occ_subtree):
            subtree.check_for_inconsistencies()

    def _parse_offsets(self, offsets_block):
        """
        Parse the offsets block from the displacements file.

        Parameters
        ----------
        offsets_block
        """
        for line in offsets_block.lines:
            if line.offset_type == "geo":
                self.geo_subtree.apply_offsets(line)
            elif line.offset_type == "vib":
                self.vib_subtree.apply_offsets(line)
            elif line.offset_type == "occ":
                self.occ_subtree.apply_offsets(line)
            else:
                raise ValueError("Unknown offset type: "
                                 f"{line.offset_type}")
        self.check_for_inconsistencies()

    def _parse_bounds(self, search_block):
        # parse geo, vib and occ bounds
        geo_block = search_block.sections[DisplacementFileSections.GEO_DELTA]
        vib_block = search_block.sections[DisplacementFileSections.VIB_DELTA]
        occ_block = search_block.sections[DisplacementFileSections.OCC_DELTA]

        for subtree, block in zip(
            (self.geo_subtree, self.vib_subtree, self.occ_subtree),
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
        constraints_block = search_block.sections[DisplacementFileSections.CONSTRAIN]
        for constraint in constraints_block:
            if constraint.constraint_type == "geo": # TODO: make into Enum
                self.geo_subtree.apply_explicit_constraint(constraint)
            elif constraint.constraint_type == "vib":
                self.vib_subtree.apply_explicit_constraint(constraint)
            elif constraint.constraint_type == "occ":
                self.occ_subtree.apply_explicit_constraint(constraint)
            else:
                raise ValueError("Unknown constraint type: "
                                 f"{constraint.constraint_type}")
        self.check_for_inconsistencies()

    def freeze(self):
        if not self._displacements_applied:
            raise ValueError("Displacements must be applied before freezing.")
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
    def n_free_params(self):
        """Returns the total number of free parameters in the parameter space.
        """
        if not self._displacements_applied:
            raise ValueError("Displacements must be applied before counting "
                             "free parameters.")
        return sum(self._free_params_up_to_layer(HLTreeLayers.Root))

    @property
    def n_symmetry_constrained_params(self):
        """Returns the total number of symmetry constrained parameters.

        This method calculates the total number of symmetry constrained
        parameters by summing up the number of symmetry constrained
        parameters from different parameter subtrees.

        Returns:
            int: The total number of symmetry constrained parameters.
        """
        return sum(self._free_params_up_to_layer(HLTreeLayers.Symmetry))

    @property
    def n_base_params(self):
        """Returns the total number of base parameters.

        This method calculates the sum of the number of base parameters from
        different parameter subtrees.

        Returns:
            int: The total number of base parameters.
        """
        return sum(self._free_params_up_to_layer(HLTreeLayers.Base))

    @property
    def n_base_scatterers(self):


    @property
    def geo_transformer(self):
        return self.geo_params.get_geo_transformer()

    @property
    def vib_transformer(self):
        return self.vib_params.get_vib_transformer()

    @property
    def occ_weight_transformer(self):
        return self.occ_params.get_weight_transformer()

    @property
    def v0r_transformer(self):
        return self.v0r_param.get_v0r_transformer()

    @property
    def n_dynamic_t_matrices(self):
        return len(self.vib_params.dynamic_site_elements)

    @property
    def n_static_t_matrices(self):
        return len(self.vib_params.static_site_elements)

    @property
    def n_dynamic_propagators(self):
        return len(self.geo_params.dynamic_propagators)

    @property
    def n_static_propagators(self):
        return len(self.geo_params.static_propagators)

    @property
    def static_t_matrix_inputs(self):
        return self.vib_params.static_t_matrix_input

    @property
    def static_propagator_inputs(self):
        return self.geo_params.static_propagator_inputs

    @property
    def dynamic_t_matrix_site_elements(self):
        return self.vib_params.dynamic_site_elements

    @property
    def t_matrix_map(self):
        return self.vib_params.t_matrix_map

    @property
    def propagator_map(self):
        return self.geo_params.propagator_map

    @property
    def propagator_plane_symmetry_operations(self):
        return jnp.array(self.geo_params.symmetry_operations)

    @property
    def is_dynamic_t_matrix(self):
        return jnp.array([val=='dynamic' for (val, id) in self.t_matrix_map])

    @property
    def is_dynamic_propagator(self):
        return jnp.array([val=='dynamic' for (val, id) in self.propagator_map])

    @property
    def is_dynamic_ase(self):
        return jnp.logical_or(
            self.is_dynamic_t_matrix, self.is_dynamic_propagator
        )

    @property
    def t_matrix_id(self):
        return jnp.array([id for (val, id) in self.t_matrix_map])

    @property
    def propagator_id(self):
        return jnp.array([id for (val, id) in self.propagator_map])

    @property
    def dynamic_ase_id(self):
        return jnp.arange(self.n_base_scatterers)[self.is_dynamic_ase]

    @property
    def static_ase_id(self):
        return jnp.arange(self.n_base_scatterers)[~self.is_dynamic_ase]

    @property
    def dynamic_ase_propagator_id(self):
        return self.propagator_id[self.is_dynamic_ase]

    @property
    def dynamic_ase_t_matrix_id(self):
        return self.t_matrix_id[self.is_dynamic_ase]

    @property
    def n_dynamic_ase(self):
        return jnp.sum(self.is_dynamic_ase)

    @property
    def n_static_ase(self):
        return jnp.sum(~self.is_dynamic_ase)

    @property
    def n_param_split(self):
        return (
            self.v0r_param.n_free_params,
            self.vib_params.n_free_params,
            self.geo_params.n_free_params,
            self.occ_params.n_free_params,
        )

    @property
    def info(self):
        """
        Returns a string containing information about the free parameters,
        symmetry constrained parameters, and total parameters.

        Returns:
            str: Information about the parameters.
        """
        return (
            "Free parameters:\n"
            f"{self.n_free_params}\t"
            f"({self.v0r_param.n_free_params} V0r, "
            f"{self.geo_params.n_free_params} geo, "
            f"{self.vib_params.n_free_params} vib, "
            f"{self.occ_params.n_free_params} occ)\n"

            "Symmetry constrained parameters:\n"
            f"{self.n_symmetry_constrained_params}\t"
            f"({self.v0r_param.n_symmetry_constrained_params} V0r, "
            f"{self.geo_params.n_symmetry_constrained_params} geo, "
            f"{self.vib_params.n_symmetry_constrained_params} vib, "
            f"{self.occ_params.n_symmetry_constrained_params} occ)\n"

            "Total parameters:\n"
            f"{self.n_base_params}\t"
            f"({self.v0r_param.n_base_params} V0r, "
            f"{self.geo_params.n_base_params} geo, "
            f"{self.vib_params.n_base_params} vib, "
            f"{self.occ_params.n_base_params} occ)\n"
        )


@register_pytree_node_class
class FrozenParameterSpace():
    frozen_attributes = (
        'dynamic_t_matrix_site_elements',
        'geo_transformer',
        'n_base_scatterers',
        'n_base_params',
        'n_dynamic_propagators',
        'n_dynamic_t_matrices',
        'n_free_params',
        'n_param_split',
        'n_static_propagators',
        'n_static_t_matrices',
        'n_symmetry_constrained_params',
        'occ_weight_transformer',
        'site_elements',
        'static_t_matrix_inputs',
        'static_propagator_inputs',
        'vib_transformer',
        'v0r_transformer',
        't_matrix_map',
        'propagator_map',
        'propagator_plane_symmetry_operations',
        '_ats_ref_z_pos',
        'is_dynamic_t_matrix',
        'is_dynamic_propagator',
        'is_dynamic_ase',
        'dynamic_ase_id',
        'propagator_id',
        'dynamic_ase_id',
        'static_ase_id',
        'dynamic_ase_propagator_id',
        'dynamic_ase_t_matrix_id',
        't_matrix_id',
        'n_dynamic_ase',
        'n_static_ase',
    )

    def split_free_params(self, free_params):
        if len(free_params) != self.n_free_params:
            raise ValueError("Number of free parameters does not match.")
        v0r_params = free_params[:self.n_param_split[0]]
        vib_params = free_params[self.n_param_split[0]:sum(self.n_param_split[:2])]
        geo_params = free_params[sum(self.n_param_split[:2]):sum(self.n_param_split[:3])]
        occ_params = free_params[sum(self.n_param_split[:3]):]
        return v0r_params, vib_params, geo_params, occ_params

    def __init__(self, parameter_space):
        for attr in self.frozen_attributes:
            setattr(self, attr, deepcopy(getattr(parameter_space, attr)))

    def tree_flatten(self):
        aux_data = {attr: getattr(self, attr)
                    for attr in self.frozen_attributes}
        children = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, children, aux_data):
        frozen_parameter_space = cls.__new__(cls)
        for kw, value in aux_data.items():
            setattr(frozen_parameter_space, kw, value)
        return frozen_parameter_space

    def expand_params(self, free_params):
        v0r_params, vib_params, geo_params, occ_params = self.split_free_params(free_params)
        v0r_shift = self.v0r_transformer(v0r_params)
        vib_amps = self.all_vib_amps(vib_params)
        displacements = self.all_displacements(geo_params)
        weights = self.occ_weight_transformer(occ_params)
        return v0r_shift, vib_amps, displacements, weights

    def all_displacements(self, geo_free_params):
        """Calculate the displacements for all propagators.

        Parameters
        ----------
        geo_free_params: The geometric free parameters.

        Returns
        -------
        displacements: The displacements for all propagators.
        """
        dynamic_displacements = self.geo_transformer(geo_free_params)
        static_displacements = jnp.array(self.static_propagator_inputs)
        if len(static_displacements) == 0:
            static_displacements = jnp.array([jnp.nan, jnp.nan, jnp.nan])

        mapped_dynamic_disp = [dynamic_displacements[id] for id in jnp.array(self.propagator_id)]
        mapped_static_disp = [static_displacements[id] for id in jnp.array(self.propagator_id)]
        vmap_where = jax.vmap(jnp.where, in_axes=(0, 0, 0))
        return vmap_where(self.is_dynamic_propagator,
                          jnp.array(mapped_dynamic_disp),
                          jnp.array(mapped_static_disp))

    def all_vib_amps(self, vib_free_params):
        """Calculate the vibrational amplitudes for all t-matrices.

        Parameters
        ----------
        vib_free_params: The vibrational free parameters.

        Returns
        -------
        vib_amps: The vibrational amplitudes for all t-matrices.
        """
        dynamic_vib_amps = self.vib_transformer(vib_free_params)
        static_vib_amps = jnp.array([va for se, va
                                     in self.static_t_matrix_inputs])
        if len(static_vib_amps) == 0:
            static_vib_amps = jnp.array([jnp.nan])

        mapped_dynamic_vib_amps = [dynamic_vib_amps[id] for id in self.t_matrix_id]
        mapped_static_vib_amps = [static_vib_amps[id] for id in self.t_matrix_id]
        return jnp.where(self.is_dynamic_t_matrix,
                          jnp.array(mapped_dynamic_vib_amps),
                          jnp.array(mapped_static_vib_amps))

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
