from dataclasses import dataclass
from pathlib import Path

import pytest
import numpy as np


@dataclass
class DeltaAmplitudeCalcInfo:
    input_path: Path
    tensor_path: Path
    energies: np.ndarray
    n_beams: int
    max_l_max: int
    displacements_path: Path

    @property
    def n_energies(self):
        return len(self.energies)

    @property
    def has_reference_delta_amps(self):
        return self.reference_delta_amps is not None

    def check_reference_delta_amps_consistency(self):
        assert self.has_reference_delta_amps
        assert self.reference_delta_amps.energies == pytest.approx(self.energies)
        assert self.reference_delta_amps.n_beams == self.n_beams



class DeltaAmplitudeReferenceData:
    pass

class TensErLEEDDeltaReferenceData:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.energies = raw_data['energies']
        self.n_beam = raw_data['n_beam']
