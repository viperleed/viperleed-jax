from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DeltaAmplitudeCalcInfo:
    input_path: Path
    tensor_path: Path
    energies: np.ndarray
    n_beams: int
    max_l_max: int
    reference_delta_amps: None

    @property
    def n_energies(self):
        return len(self.energies)

    @property
    def has_reference_delta_amps(self):
        return self.reference_delta_amps is not None



class DeltaAmplitudeReferenceData:
    pass

class TensErLEEDReferenceData:
    data: dict
    geo_disp: dict
