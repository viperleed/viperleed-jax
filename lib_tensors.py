import numpy as np
import fortranformat as ff

def read_tensor(filename, n_beams=9, n_energies=100, l_max = 11):
    # Reading in the data of a file
    try:
        with open(filename, mode="r") as file:
            content = file.readlines()
    except Exception:
        raise RuntimeError(f"Unable to read Delta file: {filename}")
    # make into an iterator
    file_lines = iter(content)

    # fortran format readers
    ff_reader_5E16_12 = ff.FortranRecordReader("5E16.12")
    ff_reader_4E16_12 = ff.FortranRecordReader("4E16.12")
    ff_reader_5E12_6 = ff.FortranRecordReader("5E12.6")
    ff_reader_4E12_6 = ff.FortranRecordReader("4E12.6")
    ff_reader_I5 = ff.FortranRecordReader("I5")
    
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
    tensor_amps_exit = np.full((n_energies, n_beams,(l_max+1)**2),
                          dtype=np.complex128, fill_value=0.0)

    delta_kx = np.full((n_energies, n_beams) ,dtype=np.float64, fill_value = 0)
    delta_ky = np.full((n_energies, n_beams) ,dtype=np.float64, fill_value = 0)
    kx_in = np.full((n_energies, n_beams) ,dtype=np.float64, fill_value = 1.0E+10) #fill with very large numbers
    ky_in = np.full((n_energies, n_beams) ,dtype=np.float64, fill_value = 1.0E+10)

    for e_step in range(n_energies):

        # energy values - version 1.71 and up compatible only
        line = next(file_lines)
        e_kin[e_step], v0i_substrate[e_step], v0i_overlayer[e_step], v0r[e_step] = ff_reader_4E16_12.read(
            line)


        # number of phaseshifts used - important for size of arrays
        line = next(file_lines)
        n_phaseshifts_per_energy[e_step] = ff_reader_I5.read(line)[0]
        
        # atomic T matrices
        # 2*(lmax +1) numbers
        t_matrix_as_real = read_block(ff_reader_5E16_12, file_lines, shape=(
            n_phaseshifts_per_energy[e_step], 2,))
        t_matrix_as_complex = t_matrix_as_real.view(dtype=np.complex128)
        t_matrix[e_step, :n_phaseshifts_per_energy[e_step]] = t_matrix_as_complex.flatten()
        
        # complex amplitudes of outgoing beams
        # 2*n_beams numbers
        amps_as_real = read_block(ff_reader_5E16_12, file_lines, shape=(n_beams, 2))
        amps_as_complex = amps_as_real.view(dtype=np.complex128)
        ref_amps[e_step, :] = amps_as_complex.flatten()
        
        # block for incident beam
        line = next(file_lines)
        n_beam = ff_reader_I5.read(line)[0]
        assert(n_beam == 0)
        line = next(file_lines) # discard relative momenta vs. incident beam (==0)
        
        tens_amps_as_real = read_block(ff_reader_5E12_6, file_lines, shape=(
        n_phaseshifts_per_energy[e_step]**2, 2))
        tens_amps_as_complex = tens_amps_as_real.view(np.complex128)[..., 0]
        tensor_amps_in[e_step, :n_phaseshifts_per_energy[e_step]**2] = tens_amps_as_complex

    
        # blocks for exiting beams
        while True:
            
            # number of beam
            line = next(file_lines)
            n_beam = ff_reader_I5.read(line)[0]
            
            if n_beam< 0:
                break
            
            # delta k and k
            line = next(file_lines)
            delta_kx[e_step, n_beam - 1], delta_ky[e_step, n_beam - 1], kx_in[e_step, n_beam - 1], ky_in[e_step, n_beam - 1] = ff_reader_4E12_6.read(
                line)
            # complex Tensor amplitudes
            
            tens_amps_as_real = read_block(ff_reader_5E12_6, file_lines, shape=(
                n_phaseshifts_per_energy[e_step]**2, 2))
            tens_amps_as_complex = tens_amps_as_real.view(np.complex128)[..., 0]
            tensor_amps_exit[e_step, n_beam-1, :n_phaseshifts_per_energy[e_step]**2] = tens_amps_as_complex


    k_in = np.full((n_energies, n_beams, 2) ,dtype=np.float64, fill_value = np.nan)
    k_in[:, :, 0] = kx_in
    k_in[:, :, 1] = ky_in
    
    k_delta = np.full((n_energies, 2, n_beams) ,dtype=np.float64, fill_value = np.nan)
    k_delta[:, 0, :] = delta_kx
    k_delta[:, 1, :] = delta_ky

    tensor_amps_exit = np.transpose(tensor_amps_exit, axes=(0, 2, 1))

    tensor_data_dict = {
        "e_kin": e_kin, #E
        "v0i_substrate": v0i_substrate, #VPIS
        "v0i_overlayer": v0i_overlayer, #VPIO
        "v0r": v0r, #VV
        "n_phaseshifts_per_energy": n_phaseshifts_per_energy, #L1DAT
        "t_matrix": t_matrix, #CAF
        "ref_amps": ref_amps, #XIST
        "tensor_amps_in": tensor_amps_in, #ALM
        "tensor_amps_out": tensor_amps_exit, #EXLM
        "kx_in": kx_in, #AK2M
        "ky_in": ky_in, #AK3M
        "k_delta": k_delta, #PSQ
        "n_energies": n_energies,
        "n_beams": n_beams, #NT0
        }
    
    return tensor_data_dict


def read_block(reader, lines, shape, dtype=np.float64): 
    llist = []
    len_lim = np.prod(shape)
    for line in lines:
        llist.extend((v for v in reader.read(line) if v is not None))
        if len(llist) >= len_lim:
            break
    return np.array(llist, dtype=dtype).reshape(shape)
