from collections import namedtuple

from jax.tree_util import register_pytree_node_class

from src.parameters.occ_parameters import ChemParams
from src.parameters.vib_parameters import VibParams
from src.parameters.geo_parameters import GeoParams


SiteEl = namedtuple('SiteEl', ['site', 'element'])
AtomSiteElement = namedtuple('AtomSiteElement', ['atom', 'site_element'])


def get_site_elements(slab):
    site_elements = []
    for site in slab.sitelist:
        if site.mixedEls:
            site_elements.extend([SiteEl(site.label, el) for el in site.mixedEls])
        else:
            site_elements.append(SiteEl(site.label, site.el))
    site_elements = tuple(site_elements) # read only from here on out
    return site_elements


def get_atom_site_elements(slab):
    atom_site_elements = []
    non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
    for at in non_bulk_atoms:
        for siteel in get_site_elements(slab):
            if siteel.site == at.site.label:
                atom_site_elements.append(AtomSiteElement(at, siteel))
    return tuple(atom_site_elements) # read only from here on out

class V0rParam():
    def __init__(self, delta_slab):
        # TODO
        self.n_free_params = 1
        self.n_base_params = 1
        self.n_symmetry_constrained_params = 1

class DeltaSlab():

    def __init__(self, slab):
        self.slab = slab
        self.non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
        self.site_elements = get_site_elements(slab)
        self.atom_site_elements = get_atom_site_elements(slab)

        # apply base parameters
        self.vib_params = VibParams(self)
        self.geo_params = GeoParams(self)
        self.occ_params = ChemParams(self)
        self.v0r_param = V0rParam(self)

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
        return self.geo_params.get_transformer()

    @property
    def vib_transformer(self):
        return self.vib_params.get_transformer()

    @property
    def occ_weights(self):
        return self.occ_params.get_transformer()

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
            f"({self.geo_params.n_free_params} geo, "
            f"{self.vib_params.n_free_params} vib, "
            f"{self.occ_params.n_free_params} occ, "
            f"{self.v0r_param.n_free_params} V0r)\n"

            "Symmetry constrained parameters:\n"
            f"{self.n_symmetry_constrained_params}\t"
            f"({self.geo_params.n_symmetry_constrained_params} geo, "
            f"{self.vib_params.n_symmetry_constrained_params} vib, "
            f"{self.occ_params.n_symmetry_constrained_params} occ, "
            f"{self.v0r_param.n_symmetry_constrained_params} V0r)\n"

            "Total parameters:\n"
            f"{self.n_base_params}\t"
            f"({self.geo_params.n_base_params} geo, "
            f"{self.vib_params.n_base_params} vib, "
            f"{self.occ_params.n_base_params} occ, "
            f"{self.v0r_param.n_base_params} V0r)\n"
        )


@register_pytree_node_class
class FrozenParameterSpace():
    frozen_attributes = (
        'site_elements',
        'n_free_params',
        'n_base_params',
        'n_symmetry_constrained_params',
    )

    def __init__(self, delta_slab):
        for attr in self.frozen_attributes:
            setattr(self, attr, getattr(delta_slab, attr))

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
