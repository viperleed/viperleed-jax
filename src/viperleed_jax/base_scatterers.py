"""Module base_scatterers of viperleed_jax."""

__authors__ = ("Alexander M. Imre (@amimre)",)
__created__ = "2024-10-14"

from collections import namedtuple

SiteEl = namedtuple("SiteEl", ["site", "element"])

# TODO: To discuss
# Currently, the internal numbering is a bit inconsitent.
# Atom numbers start at 1, but layer numbers start at 0.

class BaseScatterer:
    def __init__(self, atom, site_element):
        self.atom = atom
        self.site_element = site_element
        self.num = atom.num
        self.layer = atom.layer.num
        self.symrefm = atom.symrefm

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

class BaseScatterers:
    def __init__(self, slab):
        self.site_elements = self._get_site_elements(slab)
        self.scatterers = self._get_base_scatterers(slab)
        symmetry_links, atom_number_symmetry_links = [], []
        for linklist in slab.linklists:
            link_numbers = [at.num for at in linklist]
            atom_number_symmetry_links.append(link_numbers)
            reference_atom_number = link_numbers[0]
            reference_scatterers = [scatterer for scatterer in self.scatterers
                                    if scatterer.num == reference_atom_number]
            site_elements = [scat.site_element for scat in reference_scatterers]
            for site_element in site_elements:
                symmetry_links.append([
                    scatterer for scatterer in self.scatterers
                    if scatterer.site_element == site_element
                    and scatterer.num in link_numbers
                ])
        self.atom_number_symmetry_links = tuple(atom_number_symmetry_links)
        self.symmetry_links = tuple(symmetry_links)

    def __iter__(self):
        return iter(self.scatterers)

    def __len__(self):
        return len(self.scatterers)

    @property
    def max_atom_number(self):
        return max(scat.num for scat in self)

    @staticmethod
    def _get_site_elements(slab):
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
        return tuple(site_elements)  # read only from here on out

    def _get_base_scatterers(self, slab):
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
            for site in slab.sitelist:
                if site.label == at.site.label:
                    break
            for siteel in self.site_elements:
                if siteel.site == at.site.label:
                    base_scatterers.append(BaseScatterer(at, siteel))
        return tuple(base_scatterers)  # read only from here on out
