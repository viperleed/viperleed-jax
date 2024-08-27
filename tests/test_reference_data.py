"""Tests for the reference_data module."""
import pickle

import pytest
import numpy as np

from otftleed.data_structures import ReferenceData

def test_create_fixed_lmax_ref_data(fe2o3_pickled_tensor):
    fixed_lmax = 12
    tensor_tuple = tuple(fe2o3_pickled_tensor.values())
    ref_data = ReferenceData(tensor_tuple, fix_lmax=fixed_lmax)
    assert all(lmax == fixed_lmax for lmax in ref_data.lmax)
