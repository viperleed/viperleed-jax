import pytest
from pathlib import Path

from src.lib_phaseshifts import *
from src.lib_tensors import *
from src.lib_delta import *
from src.delta import *

#From "PARAM"
LMAX = 14  # maximum angular momentum to be used in calculation
n_beams = 9  # no. of TLEED output beams
n_atoms = 1  # currently 1 is the only possible choice
n_geo = 1  # number of geometric variations ('displacements') to be considered
DR = 0.1908624 * BOHR
DR = np.array([DR,])
IEL = 1 #element no.

# unit vectors in Angstrom
u_vec1 = np.array([1.2722, -2.2036])
u_vec2 = np.array([1.2722,  2.2036])

# area of (overlayer) lateral unit cell - in case TLEED wrt smaller unit cell is used, TVA from reference computation must be set.
unit_cell_area = np.linalg.norm(np.cross(u_vec1, u_vec2))

cu111_dir = 'tests/test_data/Cu_111_2/'
phaseshifts_file = Path(cu111_dir) / "PHASESHIFTS"
T1_file = Path(cu111_dir) / "Tensors/T_1"

_, phaseshifts, _, _ = readPHASESHIFTS(None, None, readfile=phaseshifts_file,
                                       check=False, ignoreEnRange=False)

n_energies = 0
with open(T1_file, 'r') as datei:
    for zeile in datei:
        if '-1' in zeile:
            n_energies += 1

tensor_data = read_tensor(T1_file, n_beams=9, n_energies= n_energies, l_max=LMAX+1)
interpolated_phaseshifts = interpolate_phaseshifts(phaseshifts, LMAX, tensor_data.e_kin)
atom_phaseshifts = interpolated_phaseshifts[:, [IEL-1,], :]

"""
delta_amp = lambda displacement: delta_amplitude(LMAX, np.array([DR,]),
                                                 HashableArray(tensor_data.e_kin),
                                                 (tensor_data,),
                                                 unit_cell_area,
                                                 HashableArray(atom_phaseshifts),
                                                 displacement)
"""

tensors = (tensor_data,)

 # unpack hashable arrays
_energies = tensor_data.e_kin
_phaseshifts = atom_phaseshifts
# unpack tensor data
t_matrix_ref = jnp.array([t.t_matrix for t in tensors])
t_matrix_ref = t_matrix_ref.swapaxes(0, 1)  # swap energy and atom indices
v_imag = tensors[0].v0i_substrate[0]

# atom dependent quantities
tensor_amps_out = jnp.array([t.tensor_amps_out for t in tensors])
tensor_amps_in = jnp.array([t.tensor_amps_in for t in tensors])
tensor_amps_out = tensor_amps_out.swapaxes(0, 1)  # swap energy and beam indices
tensor_amps_in = tensor_amps_in.swapaxes(0, 1)  # swap energy and beam indices

# tensor_amps_out is for outgoing beams, so we need to swap indices m -> -m
# to do this in the dense representation, we do the following:
tensor_amps_out = tensor_amps_out[:, :, (DENSE_L[LMAX]+1)**2 - DENSE_L[LMAX] - DENSE_M[LMAX] -1, :]

# apply (-1)^m to tensor_amps_out - this factor is needed
# in the calculation of the amplitude differences
tensor_amps_out = jnp.einsum('l,ealb->ealb', MINUS_ONE_POW_M[LMAX], tensor_amps_out)

# energy dependent quantities
out_k_par2 = tensors[0].kx_in # same for all atoms
out_k_par3 = tensors[0].ky_in # same for all atoms

# Calculate the t-matrix with the vibrational displacements
tscatf_vmap = jax.vmap(apply_vibrational_displacements, in_axes=(None, 0, 0, None), out_axes=1)  # vmap over energy
t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
t_matrix_new = t_matrix_new.swapaxes(0, 1)  # swap energy and atom indices
tmatrix_vmap_energy = jax.vmap(TMATRIX_DWG, in_axes=(0, 0, None, 0, None, None))
C = np.array([[0.05, 0.0, 0.0]])
C = C/BOHR
output = tmatrix_vmap_energy(t_matrix_ref, t_matrix_new, C, _energies, v_imag, LMAX)
print(output)

class TestVibration:
    def test_vibration_no_change(self):
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1)  # swap energy and atom indices
        my_output = t_matrix_new[:,0,:]

class TestTMATRIX_DWG:
    def test_tmatrix_no_geo_or_vib_displacement(self):
        C = np.array([[0.0, 0.0, 0.0]])
        output = tmatrix_vmap_energy(t_matrix_ref, t_matrix_new, C, _energies, v_imag, LMAX)
        expected_output = np.zeros_like(output)
        assert jnp.allclose(output, expected_output, atol=1e-02) # very big but only for very few indizes and same in TensErLEED

if __name__ == "__main__":
    pytest.main([__file__])


