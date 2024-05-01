from dataclasses import dataclass

import sys

import numpy as np

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from src.hashable_array import HashableArray
from src.constants import HARTREE


#@register_pytree_node_class


# TODO: keep everything in atomic units (Hartree, Bohr) internally
# TODO: maybe make property to print into eV, Angstroms, etc.
# TODO: delete tensor file data at the end

class ReferenceData:
    """Holds the data from the reference calculation

    The reference calculation produces a set of tensor files, one for each
    non-bulk atom. However, not every piece of information is atom-specific and
    many arrays are redundant between the tensor files. This class applies all
    possible simplifications to the data and stores it in a single object.

    We convert all energies to electron volts and all lengths to atomic units
    at this point to avoid repeated conversions.

    Attributes
    ----------
    energies : np.ndarray
        Kinetic energies of the reference calculation. Converted to electron
        volts. Parameter E in TensErLEED.
    n_energies : int
        Number of energies used in the reference calculation.
    v0i: float
        Imaginary part of the inner potential. Parameter VPIS in TensErLEED.
        This value is read out from the tensor file as the substrate inner
        potential. ViPErLEED and modern TensErLEED do not support a different
        inner potential for the overlayer.
    v0r: np.ndarray
        Real part of the inner potential. Parameter VV in TensErLEED.
    ref_amps: np.ndarray
        Reference scattering amplitudes. Parameter XIST in TensErLEED.
    kx_in : np.ndarray
        (Negative) absolute lateral momentum of Tensor LEED beams
        (for use as incident beams in time-reversed LEED calculation).
        PARAMETER AK2M in TensErLEED.
    ky_in : np.ndarray
        Same as kx_in, but for the y-component.
        PARAMETER AK3M in TensErLEED.
    lmax: np.ndarray
        Maximum angular momentum used for each energy. Parameter LMAX & L1DAT in
        TensErLEED.
    tensor_amps_out: np.ndarray
    tensor_amps_in: np.ndarray
    """
    expected_kws = (
            'energies',
            'v0r',
            'ref_amps',
            'kx_in',
            'ky_in',
            'lmax',
            'ref_t_matrix'
            'tensor_amps_in',
            'tensor_amps_out'
        )

    def __init__(self, tensors, fix_lmax=False):
        """TODO

        Parameters
        ----------
        tensors : tuple of TensorFileData
            _description_
        fix_lmax : bool, optional
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
                raise ValueError("Inconsistent tensor files")

        self.n_tensors = len(tensors)

        # Now we can assume that all tensors are consistent, thus we can
        # extract shared data from the first tensor file.

        # Energy Independent Data
        # we don't yet support energy dependent v0i
        v0i_per_energy = tensors[0].v0i_substrate
        if not np.all(v0i_per_energy == v0i_per_energy[0]):
            raise ValueError("Energy dependent v0i not supported")
        self.v0i = v0i_per_energy[0]     # in Hartree

        # Energy Dependent Data
        self.energies = tensors[0].e_kin # in Hartree
        self.v0r = tensors[0].v0r        # in Hartree


        self.ref_amps = tensors[0].ref_amps

        # Note: kx and ky maybe could be simplified as well
        self.kx_in = tensors[0].kx_in
        self.ky_in = tensors[0].ky_in

        # energy dependent LMAX – NB: 1 smaller than number of phaseshifts
        self.lmax = tensors[0].n_phaseshifts_per_energy - 1

        # LMAX dependent quantities – crop to max needed shape
        self.ref_t_matrix = []
        self.tensor_amps_in = []
        self.tensor_amps_out = []
        for en_id, lmax in enumerate(self.lmax):
            # crop to the first lmax+1 elements (l indexed)
            tmp_ref_t_matrix = [t.t_matrix[en_id, :lmax+1]
                                for t in tensors]
            # crops to the first (lmax+1)^2 elements (lm indexed)
            tmp_tensor_amps_in = [t.tensor_amps_in[en_id, :(lmax+1)**2]
                                  for t in tensors]
            # transpose to swap lm, and beams axis for tensor_amps_out
            tmp_tensor_amps_out = [t.tensor_amps_out[en_id, :(lmax+1)**2, :].T
                                   for t in tensors]
            # convert to arrays
            self.ref_t_matrix.append(jnp.asarray(tmp_ref_t_matrix))
            self.tensor_amps_in.append(jnp.asarray(tmp_tensor_amps_in))
            self.tensor_amps_out.append(jnp.asarray(tmp_tensor_amps_out))


    def _rebuild_from_dict(self, kw_dict):
        if not all(kw in kw_dict.keys() for kw in expected_kws):
            raise ValueError("Not all keywords supplied to rebuild from dict.")
        for kw in expected_kws:
            setattr(self, kw, kw_dict[kw])

        # TODO: rearange and apply prefactors to tensor_amps_out

    def _build_from_tensors(tensors):
        pass

    def tree_flatten(self):
        # build kw_dict to rebuild the object
        kw_dict = {}
        for kw in self.expected_kws:
            kw_dict[kw] = getattr(self, kw)
        return kw_dict

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(tensors=None, kw_dict=aux_data)

    @property
    def n_energies(self):
        return self.energies.size

    @property
    def n_beams(self):
        return self.ref_amps.shape[1]

    @property
    def min_energy_per_beam(self):
        # TODO
        pass

    @property
    def min_energy_index_per_beam(self):
        # TODO
        pass

    @property
    def size_in_memory(self):
        """Estimate size in memory.

        This may be useful for debugging and selective optimization.

        Returns
        -------
        int
            Total size in bytes of all arrays."""
        total_size_in_bytes = 0
        # simple arrays
        for attr in ["energies", "v0i", "v0r", "ref_amps", "kx_in", "ky_in"]:
            total_size_in_bytes += sys.getsizeof(getattr(self, attr))

        # lists of arrays
        for attr in ["tensor_amps_in", "tensor_amps_out"]:
            for array in getattr(self, attr):
                total_size_in_bytes += sys.getsizeof(array)

        return total_size_in_bytes

