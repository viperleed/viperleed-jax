"""Module tensors.

Uses parallel processing using Joblib with spawn (loky) backend.
"""

__authors__ = (
    'Alexander M. Imre (@amimre)',
    'Paul Haidegger (@Paulhai7)',
    'Tobias Hable (@ElHablos)',
)
__created__ = '2024-08-29'

import re
import zipfile
from dataclasses import dataclass
from io import StringIO

import fortranformat as ff
import numpy as np
from joblib import Parallel, delayed  # could be replaced with multiprocessing
from joblib.externals.loky import get_reusable_executor

# Module-level constants and precompiled regex patterns
ENERGY_BLOCK_SEPARATOR = '\n   -1\n'
FF_READER_4E12_6 = ff.FortranRecordReader('4E12.6')
FF_READER_4E16_12 = ff.FortranRecordReader('4E16.12')
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

RE_12_CHAR_NO_WHITESPACE = re.compile(r'([^\s]{12})')
RE_4X12_CHAR_NO_WHITESPACE = re.compile(
    r'([^\s]{12})([^\s]{12})([^\s]{12})([^\s]{12})'
)
RE_16_CHAR_NO_WHITESPACE = re.compile(r'([^\s]{16})')


def number_of_lines(n_floats):
    return n_floats // NUMBERS_PER_LINE + bool(n_floats % NUMBERS_PER_LINE)


@dataclass
class TensorFileData:
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
        return sum(
            hash(HashableArray(arr))
            for arr in (
                self.e_kin,
                self.ref_amps,
                self.t_matrix,
            )
        )

    @property
    def n_energies(self):
        return self.e_kin.size

    @property
    def n_beams(self):  # NTO
        return self.ref_amps.shape[1]

    def is_consistent(self, other):
        """Check if two tensor files are consistent.

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
            and np.all(
                self.n_phaseshifts_per_energy == other.n_phaseshifts_per_energy
            )
            and np.all(self.n_beams == other.n_beams)
            and np.allclose(self.v0i_substrate, other.v0i_substrate)
            and np.allclose(self.v0i_overlayer, other.v0i_overlayer)
            and np.allclose(self.v0r, other.v0r)
            and np.allclose(self.ref_amps, other.ref_amps)
            and np.allclose(self.kx_in, other.kx_in)
            and np.allclose(self.ky_in, other.ky_in)
        )


def interpret_exit_amps_subblock(split_subblock):
    try:
        beam_id, momentum_line, *amp_lines = split_subblock
    except ValueError:
        return None, None, None, None
    momentum = np.fromregex(
        StringIO(momentum_line),
        RE_4X12_CHAR_NO_WHITESPACE,
        dtype=[('c1', 'f8'), ('c2', 'f8'), ('k_in_x', 'f8'), ('k_in_y', 'f8')],
    )
    k_in_x = momentum['k_in_x'][0]
    k_in_y = momentum['k_in_y'][0]
    joined = ''.join(amp_lines)
    stringio = StringIO(joined)
    amps = np.fromregex(
        stringio, RE_12_CHAR_NO_WHITESPACE, dtype=[('c1', 'f8')]
    )
    amps = amps['c1']
    amps = np.array(amps, np.float64).view(np.complex128)
    return int(beam_id), k_in_x, k_in_y, amps


def interpret_tensor_file(content, max_l_max, n_beams, n_energies):
    """Interprets the content of a single tensor file."""
    file_hash = hash(content)
    energies = np.full((n_energies,), np.nan, dtype=np.float64)
    v0i_substrate = np.full((n_energies,), np.nan, dtype=np.float64)
    v0i_overlayer = np.full((n_energies,), np.nan, dtype=np.float64)
    v0r_arr = np.full((n_energies,), np.nan, dtype=np.float64)
    n_phaseshifts_per_energy = np.full((n_energies,), 0, dtype=np.int_)
    t_matrix = np.zeros((n_energies, (max_l_max + 1)), dtype=np.complex128)
    ref_amps = np.full((n_energies, n_beams), np.nan, dtype=np.complex128)
    incident_tensor_amps = np.full(
        (n_energies, (max_l_max + 1) ** 2), 0.0, dtype=np.complex128
    )
    outgoing_tensor_amp = np.full(
        (n_energies, n_beams, (max_l_max + 1) ** 2), 0.0, dtype=np.complex128
    )
    k_in_x_arr = np.full((n_energies, n_beams), 1.0e10, dtype=np.float64)
    k_in_y_arr = np.full((n_energies, n_beams), 1.0e10, dtype=np.float64)

    # Precompute parameters and Fortran readers for each l_max value.
    n_ref_amps_floats = 2 * n_beams
    n_ref_amps_lines = number_of_lines(n_ref_amps_floats)
    n_t_matrix_lines = {}
    n_outgoing_tensor_amp_block_lines = {}
    FF_T_MATRIX_READERS = {}
    FF_REF_AMPS_READERS = {}
    FF_OUTGOING_TENSOR_AMPS_READERS = {}

    for l_max in range(1, max_l_max + 1):
        n_t_matrix_floats = 2 * (l_max + 1)
        n_outgoing_tensor_amps_floats = 2 * (l_max + 1) ** 2
        n_t_matrix_lines[l_max] = number_of_lines(n_t_matrix_floats)
        n_outgoing_tensor_amp_block_lines[l_max] = 2 + number_of_lines(
            n_outgoing_tensor_amps_floats
        )
        # set up Fortran record readers
        FF_T_MATRIX_READERS[l_max] = ff.FortranRecordReader(
            f'{n_t_matrix_floats}E16.12'
        )
        FF_REF_AMPS_READERS[l_max] = ff.FortranRecordReader(
            f'{n_ref_amps_floats}E16.12'
        )
        FF_OUTGOING_TENSOR_AMPS_READERS[l_max] = ff.FortranRecordReader(
            f'{n_outgoing_tensor_amps_floats}E12.6'
        )

    energy_blocks = content.split(ENERGY_BLOCK_SEPARATOR)[:-1]
    if len(energy_blocks) != n_energies:
        raise ValueError(
            'Number of energy blocks does not match expected number of energies'
        )

    for en_id, block in enumerate(energy_blocks):
        energy_line, lmax_line, the_rest = block.split('\n', maxsplit=2)
        energy, v0i_sub, v0i_ovl, v0r = FF_READER_4E16_12.read(energy_line)
        energies[en_id] = energy
        v0i_substrate[en_id] = v0i_sub
        v0i_overlayer[en_id] = v0i_ovl
        v0r_arr[en_id] = v0r
        n_phaseshifts_per_energy[en_id] = int(lmax_line)
        l_max = n_phaseshifts_per_energy[en_id] - 1  # current l_max

        # Process t_matrix block
        *t_matrix_lines, the_rest = the_rest.split(
            '\n', maxsplit=n_t_matrix_lines[l_max]
        )
        t_matrix_block = ''.join(t_matrix_lines)
        t_matrix_block = FF_T_MATRIX_READERS[l_max].read(t_matrix_block)
        t_matrix_block = np.array(t_matrix_block, np.float64).view(
            np.complex128
        )
        t_matrix[en_id, : l_max + 1] = t_matrix_block

        # Process ref_amps block
        *ref_amps_lines, the_rest = the_rest.split(
            '\n', maxsplit=n_ref_amps_lines
        )
        ref_amps_block = ''.join(ref_amps_lines)
        ref_amps_block = np.fromregex(
            StringIO(ref_amps_block),
            RE_16_CHAR_NO_WHITESPACE,
            dtype=[('c1', 'f8')],
        )
        ref_amps_block = ref_amps_block['c1'].view(np.complex128)
        ref_amps[en_id] = ref_amps_block

        # Process outgoing tensor amplitudes
        outgoing_tensor_amp_blocks = []
        split_rest = the_rest.split('\n')
        for i in range(n_beams + 1):
            start = n_outgoing_tensor_amp_block_lines[l_max] * i
            end = n_outgoing_tensor_amp_block_lines[l_max] * (i + 1)
            block_slice = split_rest[start:end]
            outgoing_tensor_amp_blocks.append(
                interpret_exit_amps_subblock(block_slice)
            )
        for beam_id, k_in_x, k_in_y, amps in outgoing_tensor_amp_blocks:
            if beam_id is None:
                continue
            if beam_id == 0:
                incident_tensor_amps[en_id, : (l_max + 1) ** 2] = amps
            else:
                outgoing_tensor_amp[en_id, beam_id - 1, : (l_max + 1) ** 2] = (
                    amps
                )
                k_in_x_arr[en_id, beam_id - 1] = k_in_x
                k_in_y_arr[en_id, beam_id - 1] = k_in_y

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


def process_tensor_file(file_name, tensor_path, lmax, n_beams, n_energies):
    """Read and processes a single tensor file from a zip archive.

    Parameters
    ----------
    file_name : str
        The name of the tensor file within the zip archive.
    tensor_path : path-like
        The path to the zip archive containing the tensor files.
    lmax : int
        The maximum angular momentum quantum number.
    n_beams : int
        The number of beams to process.
    n_energies : int
        The number of energy values to process.

    Returns
    -------
    tuple
        A tuple containing:
            - file_name (str): The name of the processed tensor file.
            - tensor_data (Any): The processed tensor data as returned by
              `interpret_tensor_file`.
    """
    with zipfile.ZipFile(tensor_path, 'r') as zip_ref:
        with zip_ref.open(file_name) as file:
            content = file.read().decode('utf-8')
    return file_name, interpret_tensor_file(content, lmax, n_beams, n_energies)


def read_tensor_zip(tensor_path, lmax, n_beams, n_energies):
    """Read and interpret tensor files from a zip archive.

    Reads and interprets the contents of a tensor zip file in parallel using
    Joblib. Returns the contents as a dictionary mapping file names to
    TensorFileData instances.
    lmax, n_beams, and n_energies are required information to interpret the
    tensor data.

    Parameters
    ----------
    tensor_path : path-like
        Path to the zip file containing tensor files.
    lmax : int
        Maximum value of l (angular momentum quantum number) used to generate
        the tensors.
    n_beams : int
        Number of diffracted beams considered in the calculation.
    n_energies : int
        Number of energy values for which the tensors are calculated.

    Returns
    -------
    dict
        A dictionary where keys are file names and values are TensorFileData
        instances containing the interpreted data from the tensor files.
    """
    with zipfile.ZipFile(tensor_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
    tensor_files = [f for f in all_files if f.startswith('T_')]

    # Use threading explicitly
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(process_tensor_file)(
            file, tensor_path, lmax, n_beams, n_energies
        )
        for file in tensor_files
    )
    return dict(results)