import numpy as np
import scipy
import matplotlib.pyplot as plt

from lib_tensors import *

HARTREE = 27.211396

fig = plt.figure(figsize=(10.5, 7))
ax1 = fig.add_subplot(111)
lws = 2

n_geo = 11
CDISP = np.full((n_geo, 1, 3),dtype=np.float64,fill_value=np.nan)
for i in range(n_geo):
    CDISP[i][0][0] = -0.01*i + 0.05
    CDISP[i][0][1] = 0
    CDISP[i][0][2] = 0

LMAX = 14
n_energies = 0
with open('T_1', 'r') as datei:
    for zeile in datei:
        if '-1' in zeile:
            n_energies += 1
my_dict = read_tensor('T_1', n_beams=9, n_energies= n_energies, l_max=LMAX+1)
ref_amps = my_dict['ref_amps']

e_kin = my_dict['e_kin']
v_real = my_dict['v0r']
v_imag = my_dict['v0i_substrate']
theta, phi = 0, 0
n_beams = 9
trar1 = [1.306759, -0.7544285]
trar2 = [1.306759, 0.7544285]
trar = np.empty(shape=(2, 2), dtype="float")
trar[0, :] = trar1
trar[1, :] = trar2
beam_indices = np.array([[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 1], [1, 2], [3, 0], [0, 3]])
# incident wave vector
in_k = np.sqrt(np.maximum(0, 2 * (e_kin - v_real)))
in_k_par = in_k * np.sin(theta)  # parallel component
bk_2 = in_k_par * np.cos(phi)  # shape =( n_energy )
bk_3 = in_k_par * np.sin(phi)  # shape =( n_energy )
bk_z = np.empty_like(e_kin, dtype="complex64")
bk_z = 2 * e_kin - bk_2 ** 2 - bk_3 ** 2 - 2 * 1j * v_imag
bk_z = np.sqrt(bk_z)

# outgoing wave vector components
bk_components = np.stack((bk_2, bk_3))  # shape =(n_en, 2)
bk_components = np.outer(bk_components, np.ones(shape=(n_beams,))).reshape(
    (n_energies, 2, n_beams))  # shape =(n_en ,2 ,n_beams)
out_wave_vec = np.dot(beam_indices, trar)  # shape =(n_beams, 2)
out_wave_vec = np.outer(np.ones_like(e_kin), out_wave_vec).reshape((n_energies, 2, n_beams))  # shape =(n_en , n_beams)
out_components = bk_components + out_wave_vec
# out k vector
out_k = (2 * np.outer(e_kin, np.ones(shape=(n_beams,)))  # 2*E
         + bk_components[:, 0, :] ** 2  # + h **2
         + bk_components[:, 1, :] ** 2  # + k **2
         ).astype(dtype="complex64")
out_k_z = np.empty_like(out_k, dtype="complex64")  # shape =(n_en , n_beams )
out_k_z = np.sqrt(out_k - 2 * 1.0j * np.outer(v_imag, np.ones(shape=(n_beams,))))
out_k_perp = out_k - 2 * np.outer(v_real, np.ones(shape=(n_beams,)))
out_k_par = 2 * 1.0j * np.outer(v_imag, np.ones(shape=(n_beams,)))
out_bk_2 = out_k_par * np.cos(phi)
out_bk_3 = out_k_par * np.sin(phi)

# prefactors (refaction) from amplitudes to intensities
a = np.sqrt(out_k_perp)
c = in_k * np.cos(theta)

prefactor = np.full((n_geo, n_energies, n_beams), dtype=np.float64, fill_value=np.nan)
for i in range(n_geo):
    CXDisp = CDISP[i,0,0]
    prefactor[i,:,:] = abs(np.exp(-1j * CXDisp * (np.outer(bk_z, np.ones(shape=(n_beams,))) + out_k_z
                                                  ))) ** 2 * a / np.outer(c, np.ones(shape=(n_beams,))).real

with open('delta.npy', 'rb') as f:
    delta_amps = np.load(f)
E = my_dict['e_kin']*HARTREE

for i in range(11):
    amp = ref_amps[:, 0] + delta_amps[:, i, 0]
    intensity = prefactor[i,:,0]*abs(amp)**2
    ax1.plot(E, intensity, lw=lws)

plt.savefig("plot.svg")
