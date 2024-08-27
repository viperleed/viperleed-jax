from dataclasses import dataclass

import numpy as np
import fortranformat as ff
import zipfile

ENERGY_BLOCK_SEPARATOR = '\n   -1\n'
FF_READER_4E12_6 = ff.FortranRecordReader("4E12.6")
FF_READER_4E16_12 = ff.FortranRecordReader("4E16.12")
NUMBERS_PER_LINE = 5

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


@dataclass
class TensorFileData:
    """Holds the data read from a tensor file

    Tensor files are atom specific and generated in the reference-calculation
    for every non-bulk atom. They contain the atomic t-matrices and the
    scattering amplitudes for the reference structure. While the t-matrices are
    unique, every tensor file contains the same reference-amplitudes.

    Attributes
    ----------
    raw_file_hash: int
        Hash of the original string representation of the tensor file.
    e_kin : np.ndarray
        Kinetic energies of the reference calculation.
        Parameter E in TensErLEED.
    v0i_substrate : np.ndarray
        Imaginary part of the inner potential (substrate).
        Parameter VPIS in TensErLEED.
    v0i_overlayer : np.ndarray
        Imaginary part of the inner potential (overlayer).
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
    raw_file_hash: int
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


def read_tensor_zip(tensor_path, lmax, n_beams, n_energies):
    """
    Reads and interprets the contents of a tensor zip file.
    
    Due to the layout of the tensor files, the interpretation requires knowledge
    of the maximum angular momentum quantum number, the number of beams, and the
    number of energies. The interpretation is done in a way that is consistent
    with the Fortran code that generates/reads the tensor files.

    Args:
        tensor_path (str): The path to the tensor zip file.
        lmax (int): The maximum angular momentum quantum number.
        n_beams (int): The number of beams.
        n_energies (int): The number of energies.

    Returns:
        dict: A dictionary containing the interpreted tensor data.
            The keys are the filenames of the tensor files and the values are
            instances of TensorFileData.
    """

    # set up number of expected floats and lines
    n_t_matrix_floats = 2*(lmax-1)
    n_ref_amps_floats = 2*n_beams
    n_outgoing_tensor_amps_floats = 2*(lmax-1)**2

    n_t_matrix_lines = n_t_matrix_floats // NUMBERS_PER_LINE + 1
    n_ref_amps_lines = n_ref_amps_floats // NUMBERS_PER_LINE + 1
    n_outgoing_tensor_amp_block_lines = 2 + n_outgoing_tensor_amps_floats // NUMBERS_PER_LINE + 1
    n_outgoing_tensor_amp_lines = (n_outgoing_tensor_amp_block_lines+1)*n_beams

    # set up Fortran record readers
    FF_T_MATRIX_READER = ff.FortranRecordReader(f"{n_t_matrix_floats}E16.12")
    FF_REF_AMPS_READER = ff.FortranRecordReader(f"{n_ref_amps_floats}E16.12")
    FF_OUTGOING_TENSOR_AMPS_READER = ff.FortranRecordReader(f"{n_outgoing_tensor_amps_floats}E12.6")

    # set up subblock interpretation functions
    def interpret_exit_amps_subblock(split_subblock):
        try:
            beam_id, momentum_line, *amp_lines = split_subblock  # no extra .split()
        except ValueError:
            return None, None, None, None
        _, _, k_in_x, k_in_y = FF_READER_4E12_6.read(momentum_line)
        amps = FF_OUTGOING_TENSOR_AMPS_READER.read(''.join(amp_lines))
        amps = np.array(amps, np.float64).view(np.complex128)
        return int(beam_id), k_in_x, k_in_y, amps

    # set up main interpretation function
    def interpret_file(content):
        file_hash = hash(content)
        energies = np.full((n_energies,), fill_value=np.nan, dtype=np.float64)
        v0i_substrate = np.full((n_energies,), fill_value=np.nan, dtype=np.float64)
        v0i_overlayer = np.full((n_energies,), fill_value=np.nan, dtype=np.float64)
        v0r_arr = np.full((n_energies,), fill_value=np.nan, dtype=np.float64)
        n_phaseshifts_per_energy = np.full((n_energies,), fill_value=0, dtype=np.int_)
        t_matrix = np.full((n_energies, int(n_t_matrix_floats/2)), fill_value=np.nan, dtype=np.complex128)
        ref_amps = np.full((n_energies, n_beams), fill_value=np.nan, dtype=np.complex128)
        incident_tensor_amps = np.full((n_energies, (lmax-1)**2), fill_value=0.0, dtype=np.complex128)
        outgoing_tensor_amp = np.full((n_energies, n_beams, (lmax-1)**2), fill_value=0.0, dtype=np.complex128)
        k_in_x_arr = np.full((n_energies, n_beams), fill_value=1.0E+10, dtype=np.float64)
        k_in_y_arr = np.full((n_energies, n_beams), fill_value=1.0E+10, dtype=np.float64)

        energy_blocks = content.split('\n   -1\n')[:-1]
        if len(energy_blocks) != n_energies:
            raise ValueError('Number of energy blocks does not match expected '
                            'number of energies')

        for en_id, block in enumerate(energy_blocks):

            energy_line, lmax_line, the_rest = block.split('\n', maxsplit=2)
            energy, v0i_sub, v0i_ovl, v0r = FF_READER_4E16_12.read(energy_line)
            energies[en_id] = energy
            v0i_substrate[en_id] = v0i_sub
            v0i_overlayer[en_id] = v0i_ovl
            v0r_arr[en_id] = v0r
            n_phaseshifts_per_energy[en_id] = int(lmax_line)

            *t_matrix_lines, the_rest = the_rest.split('\n', maxsplit=n_t_matrix_lines)
            t_matrix_block = ''.join(t_matrix_lines)
            t_matrix_block = FF_T_MATRIX_READER.read(t_matrix_block)
            t_matrix_block = np.array(t_matrix_block, np.float64).view(np.complex128)
            t_matrix[en_id] = t_matrix_block

            *ref_amps_lines, the_rest = the_rest.split('\n', maxsplit=n_ref_amps_lines)
            ref_amps_block = ''.join(ref_amps_lines)
            ref_amps_block = FF_REF_AMPS_READER.read(ref_amps_block)
            ref_amps_block = np.array(ref_amps_block, np.float64).view(np.complex128)
            ref_amps[en_id] = ref_amps_block


            outgoing_tensor_amp_blocks = [
                interpret_exit_amps_subblock(the_rest.split('\n')[n_outgoing_tensor_amp_block_lines*i:n_outgoing_tensor_amp_block_lines*(i+1)])
                for i in range(n_beams+1)
            ]

            for beam_id, k_in_x, k_in_y, amps in outgoing_tensor_amp_blocks:
                if beam_id is None:
                    continue
                if beam_id == 0:
                    incident_tensor_amps[en_id] = amps
                else:
                    outgoing_tensor_amp[en_id,beam_id-1] = amps
                    k_in_x_arr[en_id,beam_id-1] = k_in_x
                    k_in_y_arr[en_id,beam_id-1] = k_in_y

        return TensorFileData(
            raw_file_hash=file_hash,
            e_kin=energies,
            v0i_substrate=v0i_substrate,
            v0i_overlayer=v0i_overlayer,
            v0r=v0r_arr,
            n_phaseshifts_per_energy=n_phaseshifts_per_energy,
            ref_amps=ref_amps,
            t_matrix=t_matrix,
            tensor_amps_in=incident_tensor_amps,
            tensor_amps_out=outgoing_tensor_amp,
            kx_in=k_in_x_arr,
            ky_in=k_in_y_arr,
        )

    # read the contents of the zip file
    tensor_zip_path = zipfile.Path(tensor_path)
    tensor_files = [f.name for f in tensor_zip_path.glob('T_*')]

    # read raw contents from the zip file
    tensor_raw_contents = {
        f: (tensor_zip_path / f).read_text()
        for f in tensor_files
    }

    # process the contents
    tensors = {
        f: interpret_file(content)
        for f, content in tensor_raw_contents.items()
    }

    return tensors
