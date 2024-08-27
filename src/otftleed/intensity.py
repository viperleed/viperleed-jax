import numpy as np
import scipy
import matplotlib.pyplot as plt

from src.files.tensors import *
from lib_intensity import *

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

prefactor = intensity_prefactor(CDISP, e_kin, v_real, v_imag, beam_indices, theta, phi, trar)

with open('delta.npy', 'rb') as f:
    delta_amps = np.load(f)
E = my_dict['e_kin']*HARTREE

for i in range(11):
    amp = ref_amps[:, 0] + delta_amps[:, i, 0]
    intensity = prefactor[i,:,0]*abs(amp)**2
    ax1.plot(E, intensity, lw=lws)

plt.savefig("plot.svg")
