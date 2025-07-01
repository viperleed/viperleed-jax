"""Module tensors."""

__authors__ = (
    'Tobias Hable (@ElHablos)',
    'Alexander M. Imre (@amimre)',
    'Paul Haidegger (@Paulhai7)',
)
__created__ = '2024-10-01'

import warnings

import fortranformat as ff
import numpy as np
from tqdm import tqdm


def read_delta_file(filename, n_energies, read_header_only=False):
    """Read and return the contents of a
    TensErLEED delta-amplitude file.

    This function reads in one file of data and stores the data
    in arrays, which can be used in later functions
    (ex.: calc_delta_intensities).

    Parameters
    ----------
    filename : str
        The filename describes which file you want to read in
        (path from the function location to the file)
    n_energies : int
        Number of different energies. This should be a
        known factor for your files.
        Also possible to just loop the file until the end.
    read_header_only : bool
        If True does reads Header only and stops afterwards

    Returns
    -------
    (phi, theta): tuple of float
        Angles of how the beam hits the sample
    (trar1, trar2) : tuple of ndarray
        reciprocal lattice vectors
    n_beams : int
        Number of beams for which delta-amplitudes were present
        in the file
    nc_steps: numpy.ndarray
        Number of permutations between direction deltas and
        vibration deltas
    e_kin_array : numpy.ndarray
        Array that contains the kinetic energies of the
        elastically scattered electrons inside the crystal.
        shape=(n_energies)
        (Not the incidence energy!)
    v_imag_array : numpy.ndarray
        Imaginary part of the inner potential of the surface
        shape=(n_energies)
    VV_array : numpy.ndarray
        Real part of the inner potential of the surface
        shape=(n_energies)
    beam_indices : numpy.ndarray
        Array of beam indices, with beam_indices[i] == [h_i, k_i];
        shape=(n_beams,2)
    Cundisp : numpy.ndarray
        Position of the undisplaced atoms (always 0)
        shape=(3)
    geo_delta : numpy.ndarray
        Geometric displacement of given delta
        shape=(n_geo_vib_grid, 3);
        n_geo_vib_grid read from header_block_2
    amplitudes_ref : numpy.ndarray
        Array that contains all values of the reference amplitudes
        shape=(n_energies,n_beamsf)
    amplitudes_del : numpy.ndarray
        Array that contains all values of the delta amplitudes
        shape=(n_energies,n_vib,n_geo,n_beams)
        n_vib and n_geo read in header_block_6;
        they are the number of geometric and vibrational displacements
    """
    header_block_1 = []
    header_block_2 = []
    e_kin_array = np.full(n_energies, fill_value=np.nan)
    v_inner_array = np.full(n_energies, fill_value=np.nan, dtype=np.complex128)

    # we need three fortran format readers
    ff_reader_6E13_7 = ff.FortranRecordReader('6E13.7')
    ff_reader_10F10_5 = ff.FortranRecordReader('10F10.5')
    ff_reader_10F7_4 = ff.FortranRecordReader('10F7.4')

    # Reading in the data of a file
    try:
        with open(filename) as file:
            content = file.readlines()
    except Exception as err:
        warnings.warn(f'Unable to read Delta file: {filename}')
        raise err

    if not content:
        raise ValueError(f'File {filename} is empty.')

    if len(content) < 2:
        raise ValueError(
            f'Invalid delta file {filename}. '
            'Not enough lines. '
            f'Found {len(content)}, '
            'expected at least 2.'
        )
    # make into an iterator
    file_lines = iter(content)

    # 1.Block of Header - only 1 line - theta, phi, trar1,
    # trar2 variables
    line = next(file_lines)
    header_block_1 = ff_reader_6E13_7.read(line)

    # surface unit cell vectors - what used to be
    # trar1 is now trar[:,0].
    # Similarly trar2 -> trar[:,1]
    # [check if you also need a .T to keep the convention
    # that trar[:,0] =  trar1
    # Also, it makes much more sense to store unit-cell vectors
    # the other way around, such that trar1 = trar[0] (== trar[0, :]).
    # Usually makes many of the calculations easier.]
    theta, phi, *trar = header_block_1
    trar = np.array(trar).reshape(2, 2)

    # 2.Block of Header - also only 1 line - n_beams, n_atoms,
    # n_geo_vib_grid variables
    line = next(file_lines)
    header_block_2 = [int(p) for p in line.split()]

    if len(header_block_2) == 2:
        n_beams, n_geo_vib_grid = header_block_2
    elif len(header_block_2) == 3:
        n_beams, n_atoms, n_geo_vib_grid = header_block_2
        if n_atoms != 1:
            raise NotImplementedError(
                f'Unsupported delta-amplitude file {filename}. '
                f'Found NATOMS={n_atoms}, but only NATOMS=1 '
                'is supported.'
            )

    else:
        raise ValueError(
            f'Invalid header in file {filename}: '
            'second line should contain 2 or 3 elements. '
            f'Found (len{header_block_2}'
        )

    # 3.Block of Header - (h,k) indices of the beams
    beam_indices = read_block(
        reader=ff_reader_10F10_5, lines=file_lines, shape=(n_beams, 2)
    )

    # TODO: if we decide to throw CUNDISP out of TensErLEED entirely,
    # this block needs to become optional
    # 4.Block of Header - position of undisplaced atom
    # (Coordinates UNDISPlaced)
    # Unused quantity - only check if it is zero (as it should be)
    pos_undisplaced = read_block(
        reader=ff_reader_10F7_4, lines=file_lines, shape=(3,)
    )
    if np.linalg.norm(pos_undisplaced) > 1e-6:
        raise NotImplementedError(
            'A non-zero value of CUNDISP (undisplaced atom '
            f'positions) was read from Delta file {filename}. '
            'This quantity is currently unused and should always '
            'be zero. Rerun Refcalc and Delta calulation with a '
            'newer TensErLEED version.'
        )

    # 5.Block of Header - geometric displacements
    # (Coordinates DISPlaced)
    # For now, this contains, along the first axis,
    # n_vib repetitions of the same
    # displacements. We will figure out n_vib further below,
    # then reshape this
    geo_delta = read_block(
        reader=ff_reader_10F7_4, lines=file_lines, shape=(n_geo_vib_grid, 3)
    )

    # 6.Block of Header - list of (vib, 0,0,0,0,..., vib, 0,0,0,0,...)
    vib_delta = read_block(
        reader=ff_reader_10F7_4, lines=file_lines, shape=(n_geo_vib_grid,)
    )
    n_vib = sum(abs(v) > 1e-4 for v in vib_delta)
    n_geo = n_geo_vib_grid // n_vib
    assert n_geo_vib_grid % n_vib == 0
    geo_delta = geo_delta.reshape(n_vib, n_geo, 3)[0, :, :].reshape(n_geo, 3)
    # throw out the zeros from array vib_delta
    vib_delta = vib_delta[::n_geo]

    if read_header_only:
        return (
            (phi, theta),
            trar,
            (n_beams, n_geo, n_vib),
            beam_indices,
            geo_delta,
            vib_delta,
            e_kin_array,
            v_inner_array,
            None,
            None,
        )

    # Initialize arrays for reference and delta amplitudes
    amplitudes_ref = np.full(
        shape=(n_energies, n_beams), fill_value=np.nan, dtype=np.complex128
    )
    amplitudes_del = np.full(
        shape=(n_energies, n_vib, n_geo, n_beams),
        fill_value=np.nan,
        dtype=np.complex128,
    )

    # maybe working arrays for transfer into amplitude arrays ?

    # End of the Header - Start of Reading in the Delta Data
    for e_index, line in enumerate(file_lines):  # Energy loop
        # Energy, VPI and VV header
        e_kin, v_imag, v_real = (
            v for v in ff_reader_6E13_7.read(line) if v is not None
        )
        # Do NOT translate energy to eV!
        if e_index < n_energies:
            e_kin_array[e_index] = e_kin
            v_inner_array[e_index] = v_real + 1j * v_imag

        # Reference amplitudes
        as_real = read_block(
            reader=ff_reader_6E13_7, lines=file_lines, shape=(n_beams, 2)
        )
        if e_index < n_energies:
            amplitudes_ref[e_index, :] = as_real.view(dtype=np.complex128)[
                ..., 0
            ]

        # Delta amplitudes
        as_real = read_block(
            reader=ff_reader_6E13_7,
            lines=file_lines,
            shape=(n_geo_vib_grid * n_beams, 2),
        )
        as_complex = as_real.view(dtype=np.complex128)
        if e_index < n_energies:
            amplitudes_del[e_index, ...] = as_complex.reshape(
                n_vib, n_geo, n_beams
            )

    if e_index > n_energies:
        raise ValueError(
            'Number of energies does not match number of blocks in file: '
            f'Found {e_index} blocks'
        )

    return (
        (phi, theta),
        trar,
        (n_beams, n_geo, n_vib),  # available from the shapes of matrices
        beam_indices,
        geo_delta,
        vib_delta,
        e_kin_array,
        v_inner_array,
        amplitudes_ref,
        amplitudes_del,
    )


def Transform(n_E, directory, filename_list):
    """This function transforms the read in data to a form, where it can be read in by the GetInt function

    Parameters
    ----------
    n_E:
    Number of different energies of one file

    directory:
    Relative path to the file that gets read in


    Returns
    -------
    phi, theta:
    Angles of how the beam hits the sample

    trar:
    reciprocal lattice vectors

    array_sizes:
    The variables int0,  nc_steps for each file stored in an array

    beam_indices:
    Array with the order of beams

    CDisp:
    Geometric displacements of the atom

    E_array:
    Array that contains all the energies of the file

    VPI_array:
    Imaginary part of the inner potential of the surface

    VV_array:
    Real part of the inner potential of the surface

    amplitudes_ref:
    Array that contains all values of the reference amplitudes

    amplitudes_del:
    Array that contains all values of the delta amplitudes

    filename_list:
    List of the filenames that contain the data
    """
    data_list_all = {}
    prev_file_constants = None

    n_files = len(filename_list)
    n_disp_atoms = n_files  # TODO generalize

    file_disp_arrays = {}
    file_ref_deltas = {}

    n_geo = np.full(shape=(n_disp_atoms,), dtype=np.int32, fill_value=np.nan)
    n_vib = np.full(shape=(n_disp_atoms,), dtype=np.int32, fill_value=np.nan)

    for ii, name in enumerate(tqdm(filename_list)):
        filename = directory + name
        data_list_all[name] = read_delta_file(filename, n_E)

        (
            (phi, theta),
            trar,
            (n_beams, n_geo[ii], n_vib[ii]),
            beam_indices,
            geo_delta_temp,
            vib_delta_temp,
            e_kin_array,
            v_inner_array,
            amplitudes_ref,
            amplitudes_del,
        ) = read_delta_file(filename, n_E)

        # Sanity check
        # All files must belong to the same Delta dataset and thus share constants
        if prev_file_constants is not None:
            assert (phi, theta) == prev_file_constants[0]
            assert np.all(trar == prev_file_constants[1])
            assert np.all(n_beams == prev_file_constants[2])
            assert np.all(beam_indices == prev_file_constants[3])
            assert np.all(e_kin_array == prev_file_constants[4])
            assert np.all(v_inner_array == prev_file_constants[5])
            assert np.all(amplitudes_ref == prev_file_constants[6])

        # do we need pos_undisplaced even? I think it is unused?

        prev_file_constants = (
            (phi, theta),
            trar,
            n_beams,
            beam_indices,
            e_kin_array,
            v_inner_array,
            amplitudes_ref,
        )

        file_disp_arrays[name] = (geo_delta_temp, vib_delta_temp)
        file_ref_deltas[name] = (amplitudes_ref, amplitudes_del)

    n_geo_max = np.max(n_geo)
    n_vib_max = np.max(n_vib)

    geo_delta = np.full(shape=(n_disp_atoms, n_geo_max, 3), fill_value=np.nan)
    vib_delta = np.full(shape=(n_disp_atoms, n_vib_max), fill_value=np.nan)

    # push values from individual files into geo_delta, vib_delta arrays

    # saving the changing data in arrays
    CDisp = np.full(shape=(n_disp_atoms, n_geo_max, 3), fill_value=np.nan)
    amplitudes_del = np.full(
        shape=(n_files, n_E, n_vib_max, n_geo_max, n_beams),
        fill_value=np.nan,
        dtype=complex,
    )
    for ii, name in enumerate(filename_list):
        # n_beams, n_atoms, nc_steps = array_sizes[i]
        # n_beams = int(n_beams)
        # n_atoms = int(n_atoms)
        # nc_steps = int(nc_steps)
        amplitudes_del[ii, :, 0 : n_vib[ii], 0 : n_geo[ii], :] = (
            file_ref_deltas[name][1]
        )

        geo_delta[ii, 0 : n_geo[ii], :] = file_disp_arrays[name][0]
        vib_delta[ii, 0 : n_vib[ii]] = file_disp_arrays[name][1]

    amplitudes_ref = np.full(
        shape=(n_E, n_beams), fill_value=np.nan, dtype=complex
    )
    # reference amplitudes are same for every file so we can take any
    amplitudes_ref[...] = file_ref_deltas[name][0]

    delta_data = {
        'phi': phi,
        'theta': theta,
        'surface_vectors': trar,
        'n_beams': n_beams,
        'n_vib_max': n_vib_max,
        'n_geo_max': n_geo_max,
        'n_geo': n_geo,
        'n_vib': n_vib,
        'beam_indices': beam_indices,
        'geo_delta': geo_delta,
        'vib_delta': vib_delta,
        'e_kin_array': e_kin_array,
        'v_inner_array': v_inner_array,
        'amplitudes_ref': amplitudes_ref,
        'amplitudes_del': amplitudes_del,
        'filename_list': filename_list,
    }

    return delta_data


def read_block(reader, lines, shape, dtype=np.float64):
    """This function reads in the individual blocks of a
    TensErLEED delta-amplitude file.

    Parameters
    ----------
    reader: FortranReader object
        The Fortran reader that is used on this block.
    lines: iterator
        Lines of the whole data file.
    shape: numpy.ndarray
        Shape of the array that gets filled with information.
    dtype: np.float64
        Type of numbers that can be stored in the returned array.

    Returns
    -------
    np.array(llist,dtype): numpy.ndarray
        Array with the contents of an individual block with the
        dimensions defined in "shape".
    """
    llist = []
    len_lim = np.prod(shape)
    for line in lines:
        llist.extend(v for v in reader.read(line) if v is not None)
        if len(llist) >= len_lim:
            break
    return np.array(llist, dtype=dtype).reshape(shape)
