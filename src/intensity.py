import numpy as np
import scipy
import matplotlib.pyplot as plt

from lib_tensors import *

HARTREE = 27.211396

fig = plt.figure(figsize=(10.5, 7))
ax1 = fig.add_subplot(111)
lws = 2


LMAX = 14
n_energies = 0
with open('T_5', 'r') as datei:
    for zeile in datei:
        if '-1' in zeile:
            n_energies += 1
my_dict = read_tensor('T_5', n_beams=9, n_energies= n_energies, l_max=LMAX+1)
ref_amps = my_dict['ref_amps']

with open('delta.npy', 'rb') as f:
    delta_amps = np.load(f)
E = my_dict['e_kin']*HARTREE
for i in range(11):
    amp = ref_amps[:, 0] + delta_amps[:, i, 0]
    ax1.plot(E, abs(amp)**2, lw=lws)

plt.savefig("plot.svg")
