"""Module data_structures."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-04-29'

from dataclasses import dataclass
from functools import partial

import jax
import numpy as np
import tqdm
from jax import numpy as jnp

from viperleed_jax import atomic_units
from viperleed_jax.constants import HARTREE
from viperleed_jax.dense_quantum_numbers import (
    DENSE_L,
    DENSE_M,
    MINUS_ONE_POW_M,
)

# TODO: keep everything in atomic units (Hartree, Bohr) internally
# TODO: maybe make property to print into eV, Angstroms, etc.


@partial(
    jax.tree_util.register_dataclass,
    # Mark all fields as meta fields
    data_fields=[],
    meta_fields=['energies', 'v0i', 'v0r',
                 'kx_in', 'ky_in', 'lmax',
                 'n_tensors', 'n_beams']
)
@dataclass
class RefCalcParams:
    energies: np.ndarray
    v0i: float
    v0r: np.ndarray
    kx_in: np.ndarray
    ky_in: np.ndarray
    lmax: np.ndarray
    n_tensors: int
    n_beams: int

    @property
    def n_energies(self):
        """Number of energies used in the reference calculation."""
        return self.energies.size

    @property
    def min_energy_per_beam(self):
        # TODO?
        pass

    @property
    def min_energy_index_per_beam(self):
        # TODO?
        pass

    @property
    def needed_lmax(self):
        return sorted(set(self.lmax))

    @property
    def energy_sorting(self):
        """Sorting to return the energies to the original order from lmax."""
        energy_ids = []
        for lmax in self.needed_lmax:
            energy_ids.append(self.energy_ids_for_lmax[lmax])
        energy_ids = np.concatenate(energy_ids)
        return np.argsort(energy_ids)

    def energy_ids_for_lmax(self, l):
        return np.where(self.lmax == l)[0]

    @property
    def incident_energy_ev(self):
        return (self.energies - self.v0r) * HARTREE

    @property
    def kappa(self):
        """Return wave vector kappa (atomic units)."""
        return atomic_units.kappa(self.energies, self.v0i)

    @property
    def max_lmax(self):
        return max(self.lmax)

@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        'ref_amps',
        't_matrices',
        'in_amps',
        'out_amps',
        ],
    meta_fields=[
    ],
)
@dataclass
class RefCalcOutput:
    ref_amps: jax.Array
    t_matrices: jax.Array
    in_amps: jax.Array
    out_amps: jax.Array

    @property
    def size_in_memory(self):
        """Estimate size in memory.

        This may be useful for debugging and selective optimization.

        Returns
        -------
        int
            Total size in bytes of all arrays.
        """
        return (
            self.ref_amps.nbytes
            + self.t_matrices.nbytes
            + self.in_amps.nbytes
            + self.out_amps.nbytes
        )


def process_tensors(tensors, fix_lmax=False):
    """TODO.

    Parameters
    ----------
    tensors : tuple of TensorFileData
        _description_
    fix_lmax : bool or int, optional
        Fixes LMAX to a constant (energy-independent) value. If an integer,
        uses this value for LMAX. If True, uses the maximum LMAX of the
        tensor files. By default False.

    Raises
    ------
    ValueError
        If the consistency checks for the tensors fails.
    """
    # Check consistency of tensor files
    for comp_tensor in tensors[1:]:
        if not tensors[0].is_consistent(comp_tensor):
            raise ValueError('Inconsistent tensor files')

    # Now we can assume that all tensors are consistent, thus we can
    # extract shared data from the first tensor file.

    # Energy Independent Data
    # we don't yet support energy dependent v0i
    v0i_per_energy = tensors[0].v0i_substrate
    if not np.all(v0i_per_energy == v0i_per_energy[0]):
        raise ValueError('Energy dependent v0i not supported')

    ref_amps = np.asarray(tensors[0].ref_amps)
    energies = np.asarray(tensors[0].e_kin)

    # energy dependent LMAX – NB: 1 smaller than number of phaseshifts
    dynamic_lmax = tensors[0].n_phaseshifts_per_energy - 1
    if fix_lmax:
        if isinstance(fix_lmax, int):
            lmax = [
                fix_lmax,
            ] * len(energies)
        else:
            lmax = [
                int(max(dynamic_lmax)),
            ] * len(energies)
    else:
        lmax = dynamic_lmax


    calc_params = RefCalcParams(
        energies=energies,  # in Hartree
        v0r=np.asarray(tensors[0].v0r),  # in Hartree
        v0i=v0i_per_energy[0],  # in Hartree
        # Note: kx and ky maybe could be simplified as well
        # TODO: pack into a single array (maybe already in read_tensor)
        kx_in=np.asarray(tensors[0].kx_in),
        ky_in=np.asarray(tensors[0].ky_in),
        lmax=lmax,
        n_tensors=len(tensors),
        n_beams=ref_amps.shape[1],
    )

    # LMAX dependent quantities - crop to max needed shape
    ref_t_matrix = np.full(
        shape=(
            calc_params.n_energies,
            calc_params.n_tensors,
            calc_params.max_lmax + 1
        ),
        dtype=np.complex128,
        fill_value=np.nan,
    )
    tensor_amps_in = np.full(
        shape=(
            calc_params.n_energies,
            calc_params.n_tensors,
            (calc_params.max_lmax + 1) ** 2,
        ),
        dtype=np.complex128,
        fill_value=np.nan,
    )
    tensor_amps_out = np.full(
        shape=(
            calc_params.n_energies,
            calc_params.n_tensors,
            calc_params.n_beams,
            (calc_params.max_lmax + 1) ** 2,
        ),
        dtype=np.complex128,
        fill_value=np.nan,
    )
    for en_id, _ in enumerate(
        tqdm.tqdm(lmax, desc='Processing tensors for energies...')
    ):
        # crop to the first lmax+1 elements (l indexed)
        tmp_ref_t_matrix = [
            t.t_matrix[en_id, : calc_params.max_lmax + 1] for t in tensors
        ]
        # crops to the first (lmax+1)^2 elements (lm indexed)
        tmp_tensor_amps_in = [
            t.tensor_amps_in[en_id, : (calc_params.max_lmax + 1) ** 2]
            for t in tensors
        ]
        # transpose to swap lm, and beams axis for tensor_amps_out
        tmp_tensor_amps_out = [
            t.tensor_amps_out[en_id, :, : (calc_params.max_lmax + 1) ** 2].T
            for t in tensors
        ]

        # One more conversion for usage in the delta amplitude calculation:
        # tensor_amps_out is for outgoing beams, so we need to swap indices
        # m -> -m. To do this in the dense representation, we do the
        # following:
        tmp_tensor_amps_out = [
            amps[
                (DENSE_L[calc_params.max_lmax] + 1) ** 2
                - DENSE_L[calc_params.max_lmax]
                - DENSE_M[calc_params.max_lmax]
                - 1
            ]
            for amps in tmp_tensor_amps_out
        ]

        # apply (-1)^m to tensor_amps_out - this factor is needed
        # in the calculation of the amplitude differences
        tmp_tensor_amps_out = [
            np.einsum('l,lb->bl', MINUS_ONE_POW_M[calc_params.max_lmax], amps)
            for amps in tmp_tensor_amps_out
        ]

        # write to arrays
        ref_t_matrix[en_id, :, :] = np.asarray(tmp_ref_t_matrix)
        tensor_amps_in[en_id, ...] = np.asarray(tmp_tensor_amps_in)
        tensor_amps_out[en_id, ...] = np.asarray(tmp_tensor_amps_out)

    ref_calc_output = RefCalcOutput(
        ref_amps=jnp.array(ref_amps),
        t_matrices=jnp.array(ref_t_matrix),
        in_amps=jnp.array(tensor_amps_in),
        out_amps=jnp.array(tensor_amps_out),
    )
    return calc_params, ref_calc_output
