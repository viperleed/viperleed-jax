"""Module phaseshifts.

This module is based on the files/phaseshifts and psgen modules from
viperleed.calc.
"""

__authors__ = (
    'Alexander M. Imre (@amimre)',
    'Florian Kraushofer (@fkraushofer)',
)
__created__ = '2024-08-29'

import logging

import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from viperleed_jax.atom_basis import SiteEl

logger = logging.getLogger(__name__)


# TODO: probably this should move into the main viperleed package
def phaseshift_site_el_order(slab, rpars):
    # this reproduces the order of blocks contained in PHASESHIFTS:
    ps_site_el_order = []
    for el in slab.elements:
        if el in rpars.ELEMENT_MIX:
            chem_el_list = rpars.ELEMENT_MIX[el]
        elif el in rpars.ELEMENT_RENAME:
            chem_el_list = [rpars.ELEMENT_RENAME[el]]
        else:
            chem_el_list = [el]
        ps_site_el_order.extend(
            [
                SiteEl(site=s.label, element=cel)
                for cel in chem_el_list
                for s in slab.sitelist
                if s.el == el
            ]
        )
    # make into a dict that maps site element to int index
    ps_site_el_map = {site_el: i for i, site_el in enumerate(ps_site_el_order)}
    return ps_site_el_map


def ps_list_to_array(ps_list):
    n_energies = len(ps_list)
    ps_energy_values = np.array([ps_list[ii][0] for ii in range(n_energies)])

    n_species = len(ps_list[0][1])
    l_max = len(ps_list[0][1][0])

    phaseshifts = np.full(
        shape=(n_species, n_energies, l_max), fill_value=np.nan
    )
    for en in range(n_energies):
        for elem_id in range(n_species):
            phaseshifts[elem_id, en, :] = ps_list[en][1][elem_id]

    return ps_energy_values, phaseshifts


# could easily be vectorized


def interpolate_phaseshift(phaseshifts, ps_energies, interp_energy, el_id, l):
    return np.interp(interp_energy, ps_energies, phaseshifts[el_id, :, l])


def regrid_phaseshifts(old_grid, new_grid, phaseshifts):
    n_elem, n_en, n_l = phaseshifts.shape
    new_phaseshifts = np.full(
        shape=(n_elem, len(new_grid), n_l), fill_value=np.nan
    )
    for l in range(n_l):
        for el in range(n_elem):
            for en_id in range(len(new_grid)):
                new_phaseshifts[el, en_id, l] = interpolate_phaseshift(
                    phaseshifts, old_grid, en_id, el, l
                )
    return new_phaseshifts


@register_pytree_node_class
class Phaseshifts:
    def __init__(self, raw_phaseshifts, energies, l_max, phaseshift_map):
        """Class to handle phaseshifts.

        Parameters:
        -----------
        raw_phaseshifts : numpy.ndarray
            The raw phaseshifts data.
        energies : numpy.ndarray
            The energies data.
        l_max : int
            The maximum value of l.
        phaseshift_map : dict
            A dictionary that maps the SiteElement namedtuple to an int index.
        """

        self.l_max = l_max
        self.phaseshift_map = phaseshift_map
        self._phaseshifts = self.interpolate(raw_phaseshifts, energies)
        self._phaseshifts = jnp.asarray(self._phaseshifts)
        # move site indices to the front
        self._phaseshifts = self._phaseshifts.swapaxes(0, 1)

    def __getitem__(self, site_el):
        index = self.phaseshift_map[site_el]
        return self._phaseshifts[index, ...]

    # TODO: We should consider a spline interpolation instead of a linear
    def interpolate(self, raw_phaseshifts, energies):
        """Interpolate phaseshifts for a given site and energy."""
        stored_phaseshift_energies = [entry[0] for entry in raw_phaseshifts]
        stored_phaseshift_energies = np.array(stored_phaseshift_energies)

        stored_phaseshifts = [entry[1] for entry in raw_phaseshifts]
        # covert to numpy array, indexed as [energy][site][l]
        stored_phaseshifts = np.array(stored_phaseshifts)

        if min(energies) < min(stored_phaseshift_energies) or max(
            energies
        ) > max(stored_phaseshift_energies):
            raise ValueError(
                'Requested energies are out of range the range '
                'for the loaded phaseshifts.'
            )

        n_sites = stored_phaseshifts.shape[1]
        # interpolate over energies for each l and site
        interpolated = np.empty(
            shape=(len(energies), n_sites, self.l_max + 1), dtype=np.float64
        )

        for l in range(self.l_max + 1):
            for site in range(n_sites):
                interpolated[:, site, l] = np.interp(
                    energies,
                    stored_phaseshift_energies,
                    stored_phaseshifts[:, site, l],
                )
        return interpolated

    # TODO: methods for easier access
