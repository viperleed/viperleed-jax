"""Constants for delta amplitude calculation."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-03-18'

# Displacement direction indices
ATOM_Z_DIR_ID = 2
DISP_Z_DIR_ID = 0

# Hartree to eV conversion factor
HARTREE = 27.211386245

# Bohr to Angstrom conversion factor
BOHR = 0.529177211

# array data types for single and double precision
FLOAT_DTYPE = {'single': 'float32', 'double': 'float64'}
COMPLEX_DTYPE = {'single': 'complex64', 'double': 'complex128'}
