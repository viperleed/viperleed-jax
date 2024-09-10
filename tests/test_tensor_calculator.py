"""Tests for the tensor_calculator module."""

from pathlib import Path
import pytest
import numpy as np
import jax

import pytest

from viperleed.calc.lib.matrix import rotation_matrix_order

from viperleed_jax.propagator import calc_propagator, symmetry_operations
from viperleed_jax.constants import BOHR
from viperleed_jax.lib_math import EPS

