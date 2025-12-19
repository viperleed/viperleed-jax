"""Module analysis.ref_calc_history."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-12-19'
__license__ = 'GPLv3+'


import re
from pathlib import Path

from viperleed.calc.constants import DEFAULT_OUT


class RefCalcHistory:
    # TODO: implement functionality to read and store ref-calc duration and timestamp

    def __init__(self, history_dir):
        history_dir = Path(history_dir)
        try:
            self.ref_R = self._read_ref_rfactor(history_dir)
        except ValueError as err:
            msg = f'Unable to read R factor from directory {history_dir}.'
            raise ValueError(msg) from err

    @staticmethod
    def _read_ref_rfactor(history_dir):
        out_dir = history_dir / DEFAULT_OUT
        file = list(out_dir.glob('R_OUT_refcalc_*'))[0]
        match = re.search(r'R=([0-9.]+)', file.name)
        return float(match.group(1))

    def __repr__(self):
        """Return string representation of RefCalcHistory."""
        return f'RefCalcHistory R: {self.ref_R}'
