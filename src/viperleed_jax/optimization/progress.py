"""Module progress."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-12-18'
__license__ = 'GPLv3+'


import re
from itertools import zip_longest
from pathlib import Path

from viperleed.calc.lib.time_utils import DateTimeFormat

from viperleed_jax.optimization.history import (
    OptimizationHistory,
    RefCalcHistory,
)

time_format = DateTimeFormat.FILE_SUFFIX


class CalcTrajectory:
    _ALLOWED_SEGMENT_CLASSES = (OptimizationHistory, RefCalcHistory)

    def __init__(self, base_path, ref_calc_dirs, search_calc_dirs):
        self.history_path = Path(base_path) / 'history'
        self.segments = []
        self._parse_trajectory(ref_calc_dirs, search_calc_dirs)

    def _add_segment(self, segment):
        if not isinstance(segment, (OptimizationHistory, RefCalcHistory)):
            msg = f'Can only add one of {self._ALLOWED_SEGMENT_CLASSES}.'
            raise TypeError(msg)

        self.segments.append(segment)

    def _parse_trajectory(self, ref_calc_dirs, search_calc_dirs):
        # parse ref-calc and search alternatingly, starting with ref-calc
        for _ref_dir, _search_dir in zip_longest(
            ref_calc_dirs, search_calc_dirs
        ):
            if _ref_dir is not None:
                tensor_id, run_id = _id_string_to_ids(_ref_dir)
                ref_dir, _ = _glob_history_dir(
                    self.history_path, tensor_id, run_id
                )
                ref_calc = RefCalcHistory(ref_dir)
                self._add_segment(ref_calc)

            if _search_dir is not None:
                tensor_id, run_id = _id_string_to_ids(_search_dir)
                search_dir, _ = _glob_history_dir(
                    self.history_path, tensor_id, run_id
                )
                supp_dir = search_dir / 'SUPP'
                paths = list(supp_dir.glob('TLOpt_*_*_history.npz'))
                paths = sorted(paths, key=lambda p: _get_opt_id(p.name))
                if len(paths) == 0:
                    msg = f'No optimization history files in "{supp_dir}".'
                    raise FileNotFoundError(msg)
                for opt_file in paths:
                    opt_history = OptimizationHistory.load_from_file(opt_file)
                    self._add_segment(opt_history)

    @property
    def ref_R_values(self):
        ref_Rs = []
        for segment in self.segments:
            if isinstance(segment, RefCalcHistory):
                ref_Rs.append(segment.ref_R)
        return ref_Rs


def _id_string_to_ids(id_string):
    """Convert an id string of the form 't{tensor_id}.r{run_id}' to integers."""
    match = re.match(r't(\d+)\.r(\d+)', id_string)
    if not match:
        msg = f'Invalid id string format: "{id_string}". Expected "t{{tensor_id}}.r{{run_id}}".'
        raise ValueError(msg)
    tensor_id = int(match.group(1))
    run_id = int(match.group(2))
    return tensor_id, run_id


def _glob_history_dir(history_path, tensor_id, run_id):
    """Given a tensor and run id, find the corresponding history directory.

    History directories are named as 't{tensor_id}.r{run_id}_{time_stamp}'.
    where time_stamp is in the format 'YYMMDD-HHMMSS'.
    """
    pattern = f't{tensor_id:03d}.r{run_id:03d}*'
    matching_dirs = list(history_path.glob(pattern))
    if not matching_dirs:
        msg = f'No history directory matching pattern "{pattern}" in "{history_path}".'
        raise FileNotFoundError(msg)
    elif len(matching_dirs) > 1:
        msg = f'Multiple history directories matching pattern "{pattern}" in "{history_path}".'
        raise ValueError(msg)
    history_dir_path = matching_dirs[0]
    time_stamp = history_dir_path.name.split('_')[-1]

    return history_dir_path, time_stamp


def _get_opt_id(opt_file_name):
    """Parse the optimization id from an optimization history file name."""
    match = re.search(r'TLOpt_(\d+)_.+_history\.npz', opt_file_name)
    if match:
        return int(match.group(1))
    msg = f'Could not parse optimization id from "{opt_file_name}".'
    raise ValueError(msg)
