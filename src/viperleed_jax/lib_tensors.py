from dataclasses import dataclass

import numpy as np
import fortranformat as ff

from viperleed_jax.hashable_array import HashableArray
from viperleed_jax.constants import HARTREE

FF_READER_5E16_12 = ff.FortranRecordReader("5E16.12")
FF_READER_4E16_12 = ff.FortranRecordReader("4E16.12")
FF_READER_5E12_6 = ff.FortranRecordReader("5E12.6")
FF_READER_4E12_6 = ff.FortranRecordReader("4E12.6")
FF_READER_I5 = ff.FortranRecordReader("I5")

"""General Information on the data layout

We use C-style indexing, i.e. the first index is the slowest varying one.
This is different from TensErLEED, which is Fortran-style indexed since it is
written in Fortran (though this is not always done consistently).

We thus always index quantities (skipping unncessary indices) as follows
from outermost (slowest) to innermost (fastest):
    1) energy
        Order of ~100-300 values. We cannot (efficiently) vectorize over
        energies, since the dynamic LMAX and thus the size of the tensors
        are energy-dependent.
    2) tensors / atom&site
        Order of ~10-100 values. May or may not be vectorized over. (TBD)
    3) beams
        Order of ~10-1000s values. Should be vectorized over. Only few
        calculations are beam-dependent.
    4) l & m (quantum numbers)
        Should be vectorized over. Almost every calculation is l & m dependent
        and these should always be handled in a batched manner.
"""


# TODO: implement consistency checks for the tensor files
@dataclass
class TensorFileData:
    """Holds the data read from a tensor file

    Tensor files are atom specific and generated in the reference-calculation
    for every non-bulk atom. They contain the atomic t-matrices and the
    scattering amplitudes for the reference structure.
    While the t-matrices are unique, every tensor file contains the same
    reference-amplitudes.

    Attributes
    ----------
    e_kin : np.ndarray
        Kinetic energies of the reference calculation.
        Parameter E in TensErLEED.
    v0i_substrate : np.ndarray
        Imaginary part of the inner potential (substrate).
        Parameter VPIS in TensErLEED.
    v0i_overlayer : np.ndarray
        Imaginary part of the inner potential (overlayer).                      # TODO: can we get rid of this?
        Parameter VPIO in TensErLEED.
    v0r : np.ndarray
        Real part of the inner potential.
        Parameter VV in TensErLEED.
    n_phaseshifts_per_energy : np.ndarray
        Number of phaseshifts used for each energy
        Parameter L1DAT in TensErLEED.
    ref_amps : np.ndarray
        Reference scattering amplitudes.
        Parameter XIST in TensErLEED.
    t_matrix : np.ndarray
        Atomic t-matrix of current site as used in reference calculation.
        Parameter CAF in TensErLEED.
    tensor_amps_in : np.ndarray
        Spherical wave amplitudes incident on current atomic site in
        the reference calculation.
        Parameter ALM in TensErLEED.
    tensor_amps_out : np.ndarray
        Spherical wave amplitudes incident from exit beam NEXIT in "time-reversed"
        LEED experiment (or rather, all terms of Born series immediately after
        scattering on current atom).
        Parameter EXLM in TensErLEED.
    kx_in : np.ndarray
        (Negative) absolute lateral momentum of Tensor LEED beams
        (for use as incident beams in time-reversed LEED calculation).
        PARAMETER AK2M in TensErLEED.
    ky_in : np.ndarray
        Same as kx_in, but for the y-component.
        PARAMETER AK3M in TensErLEED.
    """
    e_kin: np.ndarray
    v0i_substrate: np.ndarray
    v0i_overlayer: np.ndarray
    v0r: np.ndarray
    n_phaseshifts_per_energy: np.ndarray
    ref_amps: np.ndarray
    t_matrix: np.ndarray
    tensor_amps_in: np.ndarray
    tensor_amps_out: np.ndarray
    kx_in: np.ndarray
    ky_in: np.ndarray

    def __hash__(self):
        return sum(hash(HashableArray(arr)) for arr in
                   (self.e_kin, self.ref_amps, self.t_matrix,))

    @property
    def n_energies(self):
        return self.e_kin.size

    @property
    def n_beams(self):           # NTO
        return self.ref_amps.shape[1]

    def is_consistent(self, other):
        """Check if two tensor files are consistent

        Consistency is checked by comparing all data that is
        atom/site-independent. This includes the kinetic energies,

        Parameters
        ----------
        other : TensorFileData
            The other tensor file to compare with.

        Returns
        -------
        bool
            True if the two tensor files are consistent, False otherwise.
        """
        return (
            np.allclose(self.e_kin, other.e_kin)
            and np.all(self.n_phaseshifts_per_energy ==
                       other.n_phaseshifts_per_energy)
            and np.all(self.n_beams == other.n_beams)
            and np.allclose(self.v0i_substrate, other.v0i_substrate)
            and np.allclose(self.v0i_overlayer, other.v0i_overlayer)
            and np.allclose(self.v0r, other.v0r)
            and np.allclose(self.ref_amps, other.ref_amps)
            and np.allclose(self.kx_in, other.kx_in)
            and np.allclose(self.ky_in, other.ky_in)
        )


def read_tensor(filename, n_beams=9, n_energies=100, l_max = 11, compare_with=None):
    # Reading in the data of a file
    try:
        with open(filename, mode="r") as file:
            content = file.readlines()
    except Exception:
        raise RuntimeError(f"Unable to read Tensor file: {filename}")
    # make into an iterator
    file_lines = iter(content)

    # fortran format readers
    e_kin = np.full((n_energies,) ,dtype=np.float64, fill_value = np.nan)
    v0i_substrate = np.full((n_energies,) ,dtype=np.float64, fill_value = np.nan)
    v0i_overlayer = np.full((n_energies,), dtype=np.float64, fill_value=np.nan)
    v0r = np.full((n_energies,), dtype=np.float64, fill_value=np.nan)
    n_phaseshifts_per_energy = np.full(
        (n_energies,), dtype=np.int_, fill_value=0)

    # t_matrix indexed by: energy, l
    t_matrix = np.full((n_energies, l_max) ,dtype=np.complex128, fill_value = 0)
    ref_amps = np.full((n_energies, n_beams),
                       dtype=np.complex128, fill_value=np.nan)
    # may want to rethink how to index this array. Would it be better to do it the natural way, ie T[l,m,l',m']?
    tensor_amps_in = np.full((n_energies, (l_max+1)**2),
                             dtype=np.complex128, fill_value=0.0)
    tensor_amps_out = np.full((n_energies, n_beams,(l_max+1)**2),
                          dtype=np.complex128, fill_value=0.0)

    delta_kx = np.full((n_energies, n_beams) ,dtype=np.float64, fill_value = 0)
    delta_ky = np.full((n_energies, n_beams) ,dtype=np.float64, fill_value = 0)
    kx_in = np.full((n_energies, n_beams) ,dtype=np.float64, fill_value = 1.0E+10) #fill with very large numbers
    ky_in = np.full((n_energies, n_beams) ,dtype=np.float64, fill_value = 1.0E+10)

    for e_step in range(n_energies):

        # energy values - version 1.71 and up compatible only
        line = next(file_lines)
        e_kin[e_step], v0i_substrate[e_step], v0i_overlayer[e_step], v0r[e_step] = FF_READER_4E16_12.read(
            line)


        # number of phaseshifts used - important for size of arrays
        line = next(file_lines)
        n_phaseshifts_per_energy[e_step] = FF_READER_I5.read(line)[0]
        
        # atomic T matrices
        # 2*(lmax +1) numbers
        t_matrix_as_real = read_block(FF_READER_5E16_12, file_lines, shape=(
            n_phaseshifts_per_energy[e_step], 2,))
        t_matrix_as_complex = t_matrix_as_real.view(dtype=np.complex128)
        t_matrix[e_step, :n_phaseshifts_per_energy[e_step]] = t_matrix_as_complex.flatten()
        
        # complex amplitudes of outgoing beams
        # 2*n_beams numbers
        amps_as_real = read_block(FF_READER_5E16_12, file_lines, shape=(n_beams, 2))
        amps_as_complex = amps_as_real.view(dtype=np.complex128)
        ref_amps[e_step, :] = amps_as_complex.flatten()
        
        # block for incident beam
        line = next(file_lines)
        n_beam = FF_READER_I5.read(line)[0]
        assert(n_beam == 0)
        line = next(file_lines) # discard relative momenta vs. incident beam (==0)

        tens_amps_as_real = read_block(FF_READER_5E12_6, file_lines, shape=(
            n_phaseshifts_per_energy[e_step]**2, 2))
        tens_amps_as_complex = tens_amps_as_real.view(np.complex128)[..., 0]
        tensor_amps_in[e_step, :n_phaseshifts_per_energy[e_step]**2] = tens_amps_as_complex

        # blocks for exiting beams
        while True:
            # number of beam
            line = next(file_lines)
            n_beam = FF_READER_I5.read(line)[0]
            if n_beam< 0:
                break
            # delta k and k
            line = next(file_lines)
            delta_kx[e_step, n_beam - 1], delta_ky[e_step, n_beam - 1], kx_in[e_step, n_beam - 1], ky_in[e_step, n_beam - 1] = FF_READER_4E12_6.read(
                line)

            # complex Tensor amplitudes
            tens_amps_as_real = read_block(FF_READER_5E12_6, file_lines, shape=(
                n_phaseshifts_per_energy[e_step]**2, 2))
            tens_amps_as_complex = tens_amps_as_real.view(np.complex128)[..., 0]
            tensor_amps_out[e_step, n_beam-1, :n_phaseshifts_per_energy[e_step]**2] = tens_amps_as_complex


    tensor_amps_out = np.transpose(tensor_amps_out, axes=(0, 2, 1))

    # crop Tensor amplitudes to the LMAX used - TODO: check if we can do this earlier
    tensor_amps_in = tensor_amps_in[:, :(l_max+1-1)**2] 
    tensor_amps_out = tensor_amps_out[:, :(l_max+1-1)**2, :]

    return TensorFileData(
        e_kin=e_kin,
        v0i_overlayer=v0i_overlayer,
        v0i_substrate=v0i_substrate,
        v0r=v0r,
        ref_amps=ref_amps,
        t_matrix=t_matrix,
        n_phaseshifts_per_energy=n_phaseshifts_per_energy,
        tensor_amps_in=tensor_amps_in,
        tensor_amps_out=tensor_amps_out,
        kx_in=kx_in,
        ky_in=ky_in,
        )


def read_block(reader, lines, shape, dtype=np.float64): 
    llist = []
    len_lim = np.prod(shape)
    for line in lines:
        llist.extend((v for v in reader.read(line) if v is not None))
        if len(llist) >= len_lim:
            break
    return np.array(llist, dtype=dtype).reshape(shape)
