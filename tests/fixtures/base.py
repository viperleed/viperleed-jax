"""Module test.fixtures.base"""

import os
from collections.abc import Iterable
from pathlib import Path

if 'VIPERLEED_ON_THE_FLY_TESTS_LARGE_FILE_PATH' not in os.environ:
    raise ValueError('VIPERLEED_ON_THE_FLY_TESTS_LARGE_FILE_PATH not set')

LARGE_FILE_PATH = Path(__file__).parent.parent / 'test_data' / 'large_files'

if not LARGE_FILE_PATH.exists():
    raise ValueError(f'Large file path {LARGE_FILE_PATH} does not exist')


class ComparisonTensErLEEDDeltaAmps(Iterable):
    def __init__(self, reference_data, compare_abs):
        self.parameters = reference_data['parameters']
        self.expected = [
            {
                'v0r': reference_data['expected_v0r'][i],
                'vib_amplitudes': reference_data['expected_vib_amps'][i],
                'displacements': reference_data['expected_displacements'][i],
                'occ': reference_data['expected_occ'][i],
                'delta_amplitudes': reference_data[
                    'tenserleed_delta_amplitudes'
                ][i],
                'compare_abs': compare_abs,
            }
            for i in range(len(self.parameters))
        ]

    def __iter__(self):
        # Yield each set of data as a tuple
        for i in range(len(self.parameters)):
            yield (
                self.parameters[i],
                self.expected[i],
            )

    @property
    def ids(self):
        return [str(p) for p in self.parameters]
