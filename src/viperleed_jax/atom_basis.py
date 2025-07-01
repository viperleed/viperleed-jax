"""Module atom_basis of viperleed_jax."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-10-14'

from collections import namedtuple

import numpy as np
from viperleed.calc.files.new_displacements.tokens.target import TargetToken

SiteEl = namedtuple('SiteEl', ['site', 'element'])

# Note: The naming of these clasess may still need to be discussed.
# They are different from the atom class in base viperleed calc. It may make
# sense to merge the AtomBasis into viperleed.calc's Slab class.


# Currently, the internal numbering is a bit inconsitent.
# Atom numbers start at 1, but layer numbers start at 0.


class TargetSelectionError(ValueError):
    """Error raised if target selection goes wrong."""


class Atom:
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
            f'BaseScatterer(num={self.num}, layer={self.layer}, '
            f'site={self.site}, element={self.element})'
        )


class AtomBasis:
    def __init__(self, slab):
        self.site_elements = self._get_site_elements(slab)
        self.scatterers = self._get_atom_basis(slab)
        symmetry_links, atom_number_symmetry_links = [], []
        for linklist in slab.linklists:
            link_numbers = [at.num for at in linklist]
            atom_number_symmetry_links.append(link_numbers)
            reference_atom_number = link_numbers[0]
            reference_scatterers = [
                scatterer
                for scatterer in self.scatterers
                if scatterer.num == reference_atom_number
            ]
            site_elements = [scat.site_element for scat in reference_scatterers]
            for site_element in site_elements:
                symmetry_links.append(
                    [
                        scatterer
                        for scatterer in self.scatterers
                        if scatterer.site_element == site_element
                        and scatterer.num in link_numbers
                    ]
                )
        self.atom_number_symmetry_links = tuple(atom_number_symmetry_links)
        self.symmetry_links = tuple(symmetry_links)

    def __iter__(self):
        return iter(self.scatterers)

    def __getitem__(self, index):
        return self.scatterers[index]

    def __len__(self):
        return len(self.scatterers)

    @property
    def max_atom_number(self):
        return max(scat.num for scat in self)

    @staticmethod
    def _get_site_elements(slab):
        """Get the site elements from the slab.

        Parameters
        ----------
        slab (Slab): The slab object for which to retrieve the site-elements.

        Returns
        -------
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

    def _get_atom_basis(self, slab):
        """Get the atom site elements for a given slab.

        Parameters
        ----------
        slab (Slab): The slab object for which to retrieve the
            atom-site-elements.

        Returns
        -------
        tuple: A tuple of BaseScatterer objects representing the atom site
            elements.
        """
        atom_basis = []
        non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]

        for at in non_bulk_atoms:
            for site in slab.sitelist:
                if site.label == at.site.label:
                    break
            for siteel in self.site_elements:
                if siteel.site == at.site.label:
                    atom_basis.append(Atom(at, siteel))
        return tuple(atom_basis)  # read only from here on out

    def _selection_mask_from_target_token(self, target_token):
        """Select base scatterers that match the target specification."""
        mask = np.full(len(self), fill_value=True)

        # mask based on the site
        matches = np.array(
            [target_token.regex.match(bs.site) is not None for bs in self]
        )
        mask = mask & matches

        # mask based on the labels
        label_mask = mask.copy()

        # If nums are specified, apply the selection based on nums
        if target_token.nums is not None:
            # check range for nums
            if any(num < 1 or num > len(self) for num in target_token.nums):
                msg = (
                    f'Invalid atom number for target: {target_token.target_str}'
                )
                raise TargetSelectionError(msg)
            num_mask = np.array([bs.num in target_token.nums for bs in self])
            # check if any of the given nums have the wrong label
            wrong_label = np.logical_and(num_mask, ~label_mask)
            if np.any(wrong_label):
                msg = (
                    'Atom numbers do not match label for target: '
                    f'{target_token.target_str}'
                )
                raise TargetSelectionError(msg)
            mask = mask & num_mask

        # If layers are specified, apply the selection based on layers
        if target_token.layers is not None:
            mask = mask & np.array(
                # TODO: layer counting from 1; can we unify this somewhere?
                [bs.layer + 1 in target_token.layers for bs in self]
            )

        if mask.sum() == 0:
            msg = f'No atoms selected for TargetToken: {target_token}'
            raise TargetSelectionError(msg)

        return mask

    def selection_mask(self, targets):
        """Take the 'or' of all targets, combining masks."""
        if not all(isinstance(target, TargetToken) for target in targets):
            raise TargetSelectionError(
                'All supplied targets must be of TargetToken type.'
            )

        combined_mask = np.full(len(self), fill_value=False)
        for target in targets:
            combined_mask = (
                combined_mask | self._selection_mask_from_target_token(target)
            )
        return combined_mask
