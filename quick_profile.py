
import jax
from jax import config
config.update("jax_debug_nans", False)
config.update("jax_enable_x64", True)
config.update("jax_disable_jit", False)
config.update("jax_log_compiles", False)
import jax.numpy as jnp

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np


jax.devices()

import tempfile
import os, shutil
import logging
import zipfile

from viperleed_jax.data_structures import ReferenceData
from viperleed_jax.tensor_calculator import TensorLEEDCalculator
from viperleed_jax.files.phaseshifts import Phaseshifts

from viperleed_jax.from_state import calculator_from_state
from viperleed_jax.parameter_space import ParameterSpace

from viperleed.calc.run import run_calc
from viperleed.calc.files.displacements import readDISPLACEMENTS
from viperleed.calc import LOGGER as logger
from viperleed.calc.files.phaseshifts import readPHASESHIFTS
from viperleed.calc.files.iorfactor import beamlist_to_array

from viperleed_jax.files import tensors

t1 = Path('T_1')
with open(t1, 'r') as f:
    t1_content = f.read()


reader = tensors.prepare_tensor_file_reader(max_l_max=11, n_beams=354, n_energies=102)
reader(t1_content)
