from collections import namedtuple
from copy import deepcopy

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from viperleed_jax.parameters.occ_parameters import ChemParams
from viperleed_jax.parameters.vib_parameters import VibParams
from viperleed_jax.parameters.geo_parameters import GeoParams
from viperleed_jax.parameters.v0r_parameters import V0rParam

_ATOM_Z_DIR_ID = 2
_DISP_Z_DIR_ID = 0

SiteEl = namedtuple('SiteEl', ['site', 'element'])
AtomSiteElement = namedtuple('AtomSiteElement', ['atom', 'site_element'])


def get_site_elements(slab):
    """Get the site elements from the slab.

    Parameters:
    slab (Slab): The slab object for which to retrieve the site-elements.

    Returns:
    tuple: A tuple of SiteEl namedtuples representing the site elements.
    """
    site_elements = []
    for site in slab.sitelist:
        if site.mixedEls:
            site_elements.extend([SiteEl(site.label, el) for el in site.mixedEls])
        else:
            site_elements.append(SiteEl(site.label, site.el))
    site_elements = tuple(site_elements) # read only from here on out
    return site_elements


def get_atom_site_elements(slab):
    """Get the atom site elements for a given slab.

    Parameters:
    slab (Slab): The slab object for which to retrieve the
        atom-site-elements.

    Returns:
    tuple: A tuple of AtomSiteElement objects representing the atom site
        elements.
    """
    atom_site_elements = []
    non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
    for at in non_bulk_atoms:
        for siteel in get_site_elements(slab):
            if siteel.site == at.site.label:
                atom_site_elements.append(AtomSiteElement(at, siteel))
    return tuple(atom_site_elements) # read only from here on out


class ParameterSpace():

    def __init__(self, slab):
        self.slab = slab
        self.non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
        self.site_elements = get_site_elements(slab)
        self.atom_site_elements = get_atom_site_elements(slab)

        # TODO: handle "trees" in a more general manner that allows further parameters inter-linking (e.g. domains, incidence, linking geometry&vib, geo&occ etc.)
        # apply base parameters
        self.vib_params = VibParams(self)
        self.geo_params = GeoParams(self)
        self.occ_params = ChemParams(self)
        self.v0r_param = V0rParam(self)

        # atom-site-element reference z positions
        self._ats_ref_z_pos = jnp.array(
            [  ase.atom.cartpos[_ATOM_Z_DIR_ID]
             for ase in self.atom_site_elements]
        )

    def freeze(self):
        return FrozenParameterSpace(self)

    @property
    def n_free_params(self):
        """
        Returns the total number of free parameters in the DeltaSlab object.
        This includes the number of free parameters in the vibrational, geometric,
        occupancy, and v0r parameters.
        """
        return (
            self.vib_params.n_free_params
            + self.geo_params.n_free_params
            + self.occ_params.n_free_params
            + self.v0r_param.n_free_params
        )

    @property
    def n_atom_site_elements(self):
        return len(self.atom_site_elements)

    @property
    def n_base_params(self):
        """
        Returns the total number of base parameters.

        This method calculates the sum of the number of base parameters from different parameter objects,
        including `vib_params`, `geo_params`, `occ_params`, and `v0r_param`.

        Returns:
            int: The total number of base parameters.
        """
        return (
            self.vib_params.n_base_params
            + self.geo_params.n_base_params
            + self.occ_params.n_base_params
            + self.v0r_param.n_base_params
        )

    @property
    def n_symmetry_constrained_params(self):
        """
        Returns the total number of symmetry constrained parameters.
        
        This method calculates the total number of symmetry constrained
        parameters by summing up the number of symmetry constrained
        parameters from different parameter groups.

        Returns:
            int: The total number of symmetry constrained parameters.
        """
        return (
            self.vib_params.n_symmetry_constrained_params
            + self.geo_params.n_symmetry_constrained_params
            + self.occ_params.n_symmetry_constrained_params
            + self.v0r_param.n_symmetry_constrained_params
        )

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
    def propagator_rotation_angles(self):
        return jnp.array([jnp.arccos(arr[0,0]) for arr in
                          self.geo_params.symmetry_operations])

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
        return jnp.arange(self.n_atom_site_elements)[self.is_dynamic_ase]

    @property
    def static_ase_id(self):
        return jnp.arange(self.n_atom_site_elements)[~self.is_dynamic_ase]

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
        'n_atom_site_elements',
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
        'propagator_rotation_angles',
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

    def __init__(self, delta_slab):
        for attr in self.frozen_attributes:
            setattr(self, attr, deepcopy(getattr(delta_slab, attr)))

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

        mapped_dynamic_disp = [dynamic_displacements[id] for id in self.propagator_id]
        mapped_static_disp = [static_displacements[id] for id in self.propagator_id]
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
