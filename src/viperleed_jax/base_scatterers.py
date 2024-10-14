"""Module base_scatterers of viperleed_jax."""

__authors__ = ("Alexander M. Imre (@amimre)",)
__created__ = "2024-10-14"

from collections import namedtuple

SiteEl = namedtuple("SiteEl", ["site", "element"])

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
            site_elements.extend(
                [SiteEl(site.label, el) for el in site.mixedEls]
            )
        else:
            site_elements.append(SiteEl(site.label, site.el))
    site_elements = tuple(site_elements)  # read only from here on out
    return site_elements

# TODO: To discuss
# Currently, the internal numbering is a bit inconsitent.
# Atom numbers start at 1, but layer numbers start at 0.

class BaseScatterer:
    def __init__(self, atom, site_element):
        self.atom = atom
        self.site_element = site_element
        self.num = atom.num
        self.layer = atom.layer.num

    @property
    def site(self):
        return self.site_element.site

    @property
    def element(self):
        return self.site_element.element

    def __repr__(self):
        return (
            f"BaseScatterer(num={self.num}, layer={self.layer}, "
            f"site={self.site}, element={self.element})"
        )


def get_base_scatterers(slab):
    """Get the atom site elements for a given slab.

    Parameters:
    slab (Slab): The slab object for which to retrieve the
        atom-site-elements.

    Returns:
    tuple: A tuple of BaseScatterer objects representing the atom site
        elements.
    """
    base_scatterers = []
    non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
    for at in non_bulk_atoms:
        for siteel in get_site_elements(slab):
            if siteel.site == at.site.label:
                base_scatterers.append(BaseScatterer(at, siteel))
    return tuple(base_scatterers)  # read only from here on out
