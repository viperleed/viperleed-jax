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

def test_cu111_tensor_calculator_creation(cu111_tensor_calculator):
    assert cu111_tensor_calculator is not None
    assert cu111_tensor_calculator.n_beams == 9
    assert len(cu111_tensor_calculator) == 51
