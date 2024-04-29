from dataclasses import dataclass

import numpy as np

from jax.tree_util import register_pytree_node_class

from src.hashable_array import HashableArray
from src.constants import HARTREE


@register_pytree_node_class
@dataclass
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

    def __init__(tensors, fix_lmax=False):
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
            If the consitency checks for the tensors fails.
        """
        # Check consistency of tensor files


        # Extract shared data from first tensor file


        # set energy dependent LMAX


        # crop other arrays to the max needed shape

        # rearange and apply prefactors to tensor_amps_out

    @property
    def n_atoms(self):
        pass

    @property
    def n_energies(self):
        return self.energies.size

    @property
    def n_beams(self):           # NTO
        return self.ref_amps.shape[1]

    @property
    def min_energy_per_beam(self):
        pass

    @property
    def min_energy_index_per_beam(self):
        pass

