from jax import config
config.update("jax_debug_nans", False)
config.update("jax_enable_x64", True)
config.update("jax_disable_jit", False)
config.update("jax_log_compiles", False)

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from viperleed_jax.lib_tensors import read_tensor
from viperleed_jax.lib_phaseshifts import *
from viperleed_jax.data_structures import ReferenceData
from viperleed_jax.tensor_calculator import TensorLEEDCalculator

from tqdm import tqdm

from viperleed.calc.files import poscar
from viperleed.calc.files import parameters
from viperleed.calc.classes.rparams import Rparams
from viperleed.calc.files.beams import readIVBEAMS, readOUTBEAMS
from viperleed.calc.files.phaseshifts import readPHASESHIFTS
from viperleed.calc.files.vibrocc import readVIBROCC
from viperleed.calc.files.iorfactor import beamlist_to_array
from viperleed.calc.files import iorfactor as rf_io


# load input files
data_path = Path('tests') / 'test_data' / 'Fe2O3_unrelaxed'

# Read in data from POSCAR and PARAMETERS files
slab = poscar.read(data_path / 'POSCAR')
rparams = parameters.read(data_path / 'PARAMETERS')
parameters.interpret(rparams, slab, silent=False)
slab.full_update(rparams)

# reading IVBEAMS
# rparams.ivbeams = readIVBEAMS(data_path / 'IVBEAMS')
# beam_indices = np.array([beam.hk for beam in rparams.ivbeams])

# reading VIBROCC
readVIBROCC(rparams, slab, data_path / 'VIBROCC')

# incidence angles
rparams.THETA = 0.0
rparams.PHI = 90.0

LMAX = rparams.LMAX.max if rparams.LMAX.max else 10


# load experimental data

expbeams = readOUTBEAMS(data_path / 'EXPBEAMS.csv')
exp_energies, _, _, exp_intensities = beamlist_to_array(expbeams)

theobeams = readOUTBEAMS(data_path / 'OUT'/ 'THEOBEAMS.csv')
theo_energies, _, _, theo_intensities = beamlist_to_array(theobeams)

beam_indices = ((1.00000,  0.00000), (1.00000,  1.00000), (1.00000, -1.00000), (0.00000,  2.00000), (0.00000, -2.00000), (2.00000,  0.00000), (1.00000,  2.00000), (1.00000, -2.00000), (2.00000,  1.00000), (2.00000, -1.00000), (2.00000,  2.00000), (2.00000, -2.00000), (1.00000,  3.00000), (1.00000, -3.00000), (3.00000,  0.00000), (3.00000,  1.00000), (3.00000, -1.00000), (2.00000,  3.00000), (2.00000, -3.00000), (3.00000,  2.00000), (3.00000, -2.00000), (0.00000,  4.00000), (0.00000, -4.00000), (1.00000,  4.00000), (1.00000, -4.00000), (4.00000,  0.00000), (3.00000,  3.00000), (3.00000, -3.00000), (4.00000,  1.00000), (4.00000, -1.00000), (2.00000,  4.00000), (4.00000,  2.00000), (1.00000,  5.00000), (4.00000,  3.00000), (4.00000, -3.00000), (4.00000,  4.00000), (3.00000,  5.00000), (0.00000, -6.00000), )

corr = [np.argmax([b == t.hk for t in expbeams])for b in beam_indices]

try:
    param_energies = np.linspace(rparams.THEO_ENERGIES.start,
                           rparams.THEO_ENERGIES.stop,
                           rparams.THEO_ENERGIES.n_energies)
except RuntimeError:
    param_energies = theo_energies


####################
# read Tensor files

read_tensor_num = lambda num: read_tensor(data_path / 'Tensors' / f'T_{num}',
                                          n_beams=len(beam_indices),
                                        n_energies=param_energies.size,
                                        l_max=LMAX+1)
non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
print('Loading tensors...')
tensors = [read_tensor_num(at.num) for at in tqdm(non_bulk_atoms)]

print('Combining data...')
ref = ReferenceData(tensors, fix_lmax=10)

#delete tensors to free up memory
for t in tensors:
    del t
del tensors

print('Processing phaseshifts...')
# read phase shifts
phaseshifts_path = data_path /  'PHASESHIFTS'
_, raw_phaseshifts, _, _ = readPHASESHIFTS(
    slab, rparams, readfile=phaseshifts_path, check=True, ignoreEnRange=False)# TODO: site_indices needs a general solution once we implement chemical pertubations
site_indices = [0,0,1,1,1,1,1,1,1,1,1,1,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

# TODO: with this current implementation, we can not treat chemical
#       pertubations, nor vacancies. We need to implement this.
#       See e.g. iodeltas.generateDeltaInput()
#       (Treating vacancies requires setting zeros for that site)

phaseshifts = Phaseshifts(raw_phaseshifts, ref.energies, LMAX, site_indices)

####################
# Set up the calculator
print('Setting up calculator...')
calculator = TensorLEEDCalculator(ref, phaseshifts, slab, rparams, beam_indices)

centered_vib_amps = calculator.ref_vibrational_amps
centered_displacements = np.array([[0.0, 0.0, 0.0],]*30)

# Set experimental intensities
print('Setting experimental data...')
aligned_exp_intensities = exp_intensities[:, corr]
# set reference point
calculator.set_experiment_intensity(aligned_exp_intensities,
                                    exp_energies)

v0r_range = (-3.5, +3.5) # in eV
vib_amp_range = (-0.05, +0.05) # in A
geo_range = (-0.15, +0.15) # in A

centered_reduced_vib_amps = np.array([0.089, 0.06, 0.141, 0.115])
centered_reduced_displacements = np.array([[0.0],]*5).flatten()

calculator.parameter_transformer.set_vib_amp_bounds(centered_reduced_vib_amps + vib_amp_range[0], centered_reduced_vib_amps + vib_amp_range[1])
calculator.parameter_transformer.set_v0r_bounds(*v0r_range)
calculator.parameter_transformer.set_displacement_bounds(centered_reduced_displacements + geo_range[0], centered_reduced_displacements + geo_range[1])

# read tensor files for non-bulk atoms
non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]

# vibration constraints to change all sites together
every_second_site = site_indices[::2] # for Fe2O3, every 2nd atom is symmetry independent
vib_constraints = np.zeros(shape=(calculator.parameter_transformer.n_irreducible_vib_amps, max(every_second_site)+1))
for at_id, site in enumerate(every_second_site):
    vib_constraints[at_id, site] = 1.0

# geometric constraints to move only z for the topmost layer (*L(1) z) in viperleed
atoms_in_first_layer = [0, 1, 8, 9, 10]
geo_constraints = np.zeros(shape=(calculator.parameter_transformer.n_irreducible_displacements, len(atoms_in_first_layer)))
for at_id, site in enumerate(atoms_in_first_layer):
    geo_constraints[site*3, at_id] = 1.0

# apply constraints
calculator.parameter_transformer.apply_geo_constraints(geo_constraints)
calculator.parameter_transformer.apply_vib_constraints(vib_constraints)

print(calculator.parameter_transformer.info)
