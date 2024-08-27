import pytest
from pathlib import Path

from otftleed.lib_phaseshifts import *
from otftleed.lib_tensors import *
from otftleed.t_matrix import vmap_vib_dependent_tmatrix
from otftleed.lib_delta import apply_geometric_displacements, TMATRIX_DWG
from otftleed.delta import *
from otftleed.dense_quantum_numbers import MINUS_ONE_POW_M, DENSE_L, DENSE_M

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
interpolated_phaseshifts = Phaseshifts(phaseshifts, tensor_data.e_kin, LMAX, [0])
atom_phaseshifts = interpolated_phaseshifts.phaseshifts[:, [IEL-1,], :]

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

k_inside = jnp.sqrt(2*_energies-2j*v_imag+1j*EPS)

# Propagator evaluated relative to the muffin tin zero i.e.
# it uses energy = incident electron energy + inner potential
out_k_par = out_k_par2**2 + out_k_par3**2
out_k_perp_inside = jnp.sqrt(
    ((2*_energies-2j*v_imag)[:, jnp.newaxis] - out_k_par)
    + 1j*EPS
)

tensor_amps_out_with_prefactors = jnp.einsum('ealb,e,eb,->ealb',
    tensor_amps_out,
    1/k_inside,
    1/out_k_perp_inside,
    1/(2*(unit_cell_area/BOHR**2))
)
tensor_amps_out_with_prefactors = tensor_amps_out_with_prefactors.swapaxes(2,3)

# Calculate the t-matrix with the vibrational displacements
tscatf_vmap = jax.vmap(vmap_vib_dependent_tmatrix, in_axes=(None, 0, 0, None), out_axes=1) # vmap over energy
tmatrix_vmap_energy = jax.vmap(TMATRIX_DWG, in_axes=(0, 0, None, 0, None, None))
matel_dwg_vmap_energy = jax.vmap(apply_geometric_displacements, in_axes=(0, 0, 0, None, None, 0, 0, None))
#with open('test_geo_disp_xyz_vib.npy','wb') as f:
#    np.save(f, d_amplitude)

"""All the premade cases (the .npy data which is used for reference in the tests) are made by a working
python version of the code. All data have been compared manually with the FORTRAN code and the error is small
enough. (relative error smaller than 1e-04 for numbers of regular magnitude. Very small numbers differ more)

Note that 0.166667*4 (approximation of 2/3) from the FORTRAN code is replaced with 2/3 which also changed
the output a little. If it is necessary to compare with the FORTRAN code again, please change it back."""

class TestVibration:
    def test_vibration_no_change(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1)  # swap energy and atom indices
        output = t_matrix_new
        expected_output = t_matrix_ref
        assert jnp.allclose(output, expected_output,atol=1e-02) # in refcalc the LMAX is changed depending on the energie.
                                                                # In the delta calculation always the same LMAX is used so 
                                                                # some matrix elements are not zero which are in refcalc.

    def test_vibration_positive_change(self):
        DR = 0.3 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1)  # swap energy and atom indices
        output = t_matrix_new
        with open(cu111_dir + 'Premade_cases/test_vibration_positive_change.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output)

    def test_vibration_negative_change(self):
        DR = 0.1 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1)  # swap energy and atom indices
        output = t_matrix_new
        with open(cu111_dir + 'Premade_cases/test_vibration_negative_change.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output)
    
    def test_vibration_zero_vibration(self): #with no vibration the t_matrix is the temperature independent
                                             #t_matrix, which can be calculated from the phaseshifts
        DR = 0
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1)  # swap energy and atom indices
        output = t_matrix_new
        expected_output = (np.exp(2j*_phaseshifts)-1)/2j
        assert output == pytest.approx(expected_output)


class TestTMATRIX_DWG:
    def test_tmatrix_no_displacement(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[0.0, 0.0, 0.0]])/BOHR
        C = C * jnp.array([1, 1, -1])   # The vector C must be expressed W.R.T. a right handed set of axes.
                                        # CDISP() is input W.R.T. a left handed set of axes.
        output = tmatrix_vmap_energy(t_matrix_ref, t_matrix_new, C, _energies, v_imag, LMAX)
        expected_output = np.zeros_like(output)
        assert jnp.allclose(output, expected_output, atol=1e-02) # very big because of the differnce in LMAX to the recalc 
                                                                 # mentioned in test_vibration_no_change
    
    def test_tmatrix_no_displacement_own_ref_matrix(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[0.0, 0.0, 0.0]])/BOHR
        C = C * jnp.array([1, 1, -1])
        output = tmatrix_vmap_energy(t_matrix_new, t_matrix_new, C, _energies, v_imag, LMAX)
        expected_output = np.zeros_like(output)
        assert output == pytest.approx(expected_output)

    def test_tmatrix_z_displacement(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[0.05, 0.0, 0.0]])/BOHR
        C = C * jnp.array([1, 1, -1])
        output = tmatrix_vmap_energy(t_matrix_ref, t_matrix_new, C, _energies, v_imag, LMAX)
        with open(cu111_dir + 'Premade_cases/test_tmatrix_dwg_disp_z.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output, abs=1e-07)

    def test_tmatrix_x_displacement(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[0.0, 0.05, 0.0]])/BOHR
        C = C * jnp.array([1, 1, -1])
        output = tmatrix_vmap_energy(t_matrix_ref, t_matrix_new, C, _energies, v_imag, LMAX)
        with open(cu111_dir + 'Premade_cases/test_tmatrix_dwg_disp_x.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output, abs=1e-07)

    def test_tmatrix_y_displacement(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[0.0, 0.0, 0.05]])/BOHR
        C = C * jnp.array([1, 1, -1]) 
        output = tmatrix_vmap_energy(t_matrix_ref, t_matrix_new, C, _energies, v_imag, LMAX)
        with open(cu111_dir + 'Premade_cases/test_tmatrix_dwg_disp_y.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output, abs=1e-07)

    def test_tmatrix_xyz_displacement(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[-0.01, 0.02, -0.03]])/BOHR
        C = C * jnp.array([1, 1, -1])
        output = tmatrix_vmap_energy(t_matrix_ref, t_matrix_new, C, _energies, v_imag, LMAX)
        with open(cu111_dir + 'Premade_cases/test_tmatrix_dwg_disp_xyz.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output)

    def test_tmatrix_xyz_and_vibrational_displacement(self):
        DR = 0.3* BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[-0.01, 0.02, -0.03]])/BOHR
        C = C * jnp.array([1, 1, -1])
        output = tmatrix_vmap_energy(t_matrix_ref, t_matrix_new, C, _energies, v_imag, LMAX)
        with open(cu111_dir + 'Premade_cases/test_tmatrix_dwg_disp_xyz_vib.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output)

class TestGeo:
    def test_geo_no_displacement(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[0.0, 0.0, 0.0]])
        output = matel_dwg_vmap_energy(t_matrix_ref, t_matrix_new, _energies, v_imag,
                        LMAX, tensor_amps_out_with_prefactors, tensor_amps_in,
                        C)
        expected_output = np.zeros_like(output)
        assert output == pytest.approx(expected_output, abs=1e-04) # very big because of the differnce in LMAX to the recalc 
                                                                 # mentioned in test_vibration_no_change
    
    def test_geo_no_displacement_own_ref_matrix(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[0.0, 0.0, 0.0]])
        output = matel_dwg_vmap_energy(t_matrix_new, t_matrix_new, _energies, v_imag,
                        LMAX, tensor_amps_out_with_prefactors, tensor_amps_in,
                        C)
        expected_output = np.zeros_like(output)
        assert output == pytest.approx(expected_output)
    
    def test_geo_z_displacement(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[0.05, 0.0, 0.0]])
        output = matel_dwg_vmap_energy(t_matrix_ref, t_matrix_new, _energies, v_imag,
                        LMAX, tensor_amps_out_with_prefactors, tensor_amps_in,
                        C)
        with open(cu111_dir + 'Premade_cases/test_geo_disp_z.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output)
    
    def test_geo_x_displacement(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[0.0, 0.05, 0.0]])
        output = matel_dwg_vmap_energy(t_matrix_ref, t_matrix_new, _energies, v_imag,
                        LMAX, tensor_amps_out_with_prefactors, tensor_amps_in,
                        C)
        with open(cu111_dir + 'Premade_cases/test_geo_disp_x.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output)
    
    def test_geo_y_displacement(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[0.0, 0.0, 0.05]])
        output = matel_dwg_vmap_energy(t_matrix_ref, t_matrix_new, _energies, v_imag,
                        LMAX, tensor_amps_out_with_prefactors, tensor_amps_in,
                        C)
        with open(cu111_dir + 'Premade_cases/test_geo_disp_y.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output)
    
    def test_geo_xyz_displacement(self):
        DR = 0.1908624 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[-0.03, 0.02, -0.01]])
        output = matel_dwg_vmap_energy(t_matrix_ref, t_matrix_new, _energies, v_imag,
                        LMAX, tensor_amps_out_with_prefactors, tensor_amps_in,
                        C)
        with open(cu111_dir + 'Premade_cases/test_geo_disp_xyz.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output)
    
    def test_geo_xyz_and_vibrational_displacement(self):
        DR = 0.1 * BOHR
        DR = np.array([DR,])
        t_matrix_new = tscatf_vmap(LMAX, _phaseshifts, _energies, DR)
        t_matrix_new = t_matrix_new.swapaxes(0, 1) 
        C = np.array([[-0.03, 0.02, -0.01]])
        output = matel_dwg_vmap_energy(t_matrix_ref, t_matrix_new, _energies, v_imag,
                        LMAX, tensor_amps_out_with_prefactors, tensor_amps_in,
                        C)
        with open(cu111_dir + 'Premade_cases/test_geo_disp_xyz_vib.npy', 'rb') as f:
            expected_output = np.load(f)
        assert output == pytest.approx(expected_output)

if __name__ == "__main__":
    pytest.main([__file__])