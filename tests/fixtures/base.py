"""Module test.fixtures.base"""
import os
from pathlib import Path

if 'VIPERLEED_ON_THE_FLY_TESTS_LARGE_FILE_PATH' not in os.environ:
    raise ValueError('VIPERLEED_ON_THE_FLY_TESTS_LARGE_FILE_PATH not set')

LARGE_FILE_PATH = Path(os.environ['VIPERLEED_ON_THE_FLY_TESTS_LARGE_FILE_PATH'])
