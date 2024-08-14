from collections import namedtuple

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
        self.n_free_params = 1

class DeltaSlab():

    def __init__(self, slab):
        self.slab = slab
        self.non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
        self.site_elements = get_site_elements(slab)
        self.atom_site_elements = get_atom_site_elements(slab)

        # apply base parameters
        self.vib_params = VibParams(self)
        self.geo_params = GeoParams(self)
        self.chem_params = ChemParams(self)
        self.v0r_param = V0rParam(self)

    @property
    def n_free_parameters(self):
        return (
            self.vib_params.n_free_params
            + self.geo_params.n_free_params
            + self.chem_params.n_free_params
            + self.v0r_param.n_free_params
        )
