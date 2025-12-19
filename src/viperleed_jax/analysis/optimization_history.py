"""Module optimization.history."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-03-11'
__license__ = 'GPLv3+'


import time

import numpy as np


class OptimizationHistory:
    """Class to keep track of optimization history and results."""

    def __init__(self, algorithm=None):
        self._start_time = time.time()

        # Core data storage (lists for efficient appending)
        self._data = {'x': [], 'R': [], 'timestamps': []}

        # Metadata storage (for scalars like 'message', 'algorithm', 'convergence_gens')
        self.metadata = {
            'algorithm': algorithm if algorithm else 'Unknown',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

    # --- Recording Phase ---

    def add_step(self, x, R, **kwargs):
        """
        Record a single step or generation of the optimization.

        Note: Enforces generalized dimensions.
        - x becomes (pop_size, params)
        - R becomes (pop_size,)
        For gradient descent (single point), pop_size is 1.
        """
        # 1. Normalize x to be 2D (pop_size, params)
        x_arr = np.array(x, copy=True)
        if x_arr.ndim == 1:
            x_arr = x_arr[np.newaxis, :]  # Add pop dimension -> (1, params)
        self._data['x'].append(x_arr)

        # 2. Normalize R to be 1D (pop_size,)
        # Handle None safely first
        val_R = np.nan if R is None else R
        R_arr = np.array(val_R)

        if R_arr.ndim == 0:
            R_arr = R_arr[np.newaxis]  # Add pop dimension -> (1,)

        self._data['R'].append(R_arr)
        self._data['timestamps'].append(time.time())

        # Dynamically store extra series
        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []

            if value is None:
                # Store NaN array of same shape as x, or just np.nan
                # Using np.nan is usually safer for flexible storage
                self._data[key].append(np.nan)
            else:
                self._data[key].append(value)

    def mark_complete(self, message, **kwargs):
        """Finalize the history with a message and any final metadata."""
        self.metadata['message'] = message
        self.metadata.update(kwargs)

    # --- Analysis Phase (Properties) ---

    @property
    def duration(self):
        if not self._data['timestamps']:
            return 0.0
        return self._data['timestamps'][-1] - self._start_time

    @property
    def x_history(self):
        return np.array(self._data['x'])

    @property
    def R_history(self):
        return np.array(self._data['R'])

    @property
    def R_running_min(self):
        min_R_per_gen = np.nanmin(self.R_history, axis=1)
        return np.fmin.accumulate(min_R_per_gen)

    @property
    def relative_times(self):
        return np.array(
            [
                timestamp - self._start_time
                for timestamp in self._data['timestamps']
            ]
        )

    def __getattr__(self, name):
        """
        Magic accessor for dynamic history.
        e.g., result.grad_R_history will return np.array(self._data['grad_R'])
        """
        # Specialized check to avoid recursion issues during serialization/copying
        if name.startswith('_'):
            raise AttributeError(name)

        # 1. Try to find it in _data (e.g. step_size_history -> step_size)
        key = name.replace('_history', '')
        if key in self._data:
            return np.array(self._data[key])

        # 2. Try to find it in metadata
        if name in self.metadata:
            return self.metadata[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    @property
    def best_R(self):
        """Smart retrieval of best R based on stored metadata."""
        R = self.R_history

        # Determine subset to look at
        if 'convergence_generations' in self.metadata:
            n = self.metadata['convergence_generations']
            subset_R = R if not n else R[-n:]
        else:
            subset_R = R

        # Handle NaNs (common in Gradient steps where R=None)
        # We assume if all are NaN, return NaN
        if np.all(np.isnan(subset_R)):
            return np.nan

        return np.nanmin(subset_R)

    @property
    def best_x(self):
        """Smart retrieval of best x based on stored metadata."""
        R = self.R_history
        x = self.x_history

        if 'convergence_generations' in self.metadata:
            n = self.metadata['convergence_generations']
            if not n:
                trunc_R, trunc_x = R, x
            else:
                trunc_R, trunc_x = R[-n:], x[-n:]
        else:
            trunc_R, trunc_x = R, x

        # nanargmin is safer if there are NaNs (e.g. gradient steps)
        # It flattens the array index
        flat_idx = np.nanargmin(trunc_R)
        # Unravel into (gen_idx, pop_idx)
        idx = np.unravel_index(flat_idx, trunc_R.shape)

        return trunc_x[idx]

    def expand_parameters(self, calculator):
        """
        Post-processing: Expand stored parameters into physics quantities.
        Stores results directly into self._data.
        """
        # Prepare storage
        for key in ['v0r_offsets', 'geo_displacements', 'vibration_amplitudes', 'occupations']:
            # We will overwrite any existing list to ensure clean structure
            self._data[key] = []

        x_arr = self.x_history
        n_gens, pop_size, _ = x_arr.shape

        # We need to preserve the Generation -> Population hierarchy
        for g in range(n_gens):
            gen_v0r, gen_geo, gen_vib, gen_occ = [], [], [], []

            for p in range(pop_size):
                v0r, geo, vib, occ = calculator.expand_params(x_arr[g, p])
                gen_v0r.append(v0r)
                gen_geo.append(geo)
                gen_vib.append(vib)
                gen_occ.append(occ)

            # Append the whole population block as one item in the main list
            # equivalent to how 'x' is stored
            self._data['v0r_offsets'].append(np.array(gen_v0r))
            self._data['geo_displacements'].append(np.array(gen_geo))
            self._data['vibration_amplitudes'].append(np.array(gen_vib))
            self._data['occupations'].append(np.array(gen_occ))

    # --- I/O Phase ---

    def write_to_file(self, file_path):
        """Save everything to .npz."""
        self.metadata['start_time'] = self._start_time

        # 1. Save data history arrays
        save_dict = {}
        for k, v in self._data.items():
            key_name = k if k.endswith('_history') else f'{k}_history'
            save_dict[key_name] = np.array(v)

        # 2. Save metadata (prefixed with meta_)
        for k, v in self.metadata.items():
            save_dict[f'meta_{k}'] = v

        np.savez(file_path, **save_dict)

    @classmethod
    def load_from_file(cls, file_path):
        """Factory method to recreate the object from .npz, supporting legacy files."""
        loaded = np.load(file_path, allow_pickle=True)
        instance = cls()

        # set start time to None - will be replaced from file
        instance._start_time = None

        # Check if this is a "New Format" file (contains meta_ keys)
        is_legacy = not any(k.startswith('meta_') for k in loaded.files)

        if is_legacy:
            instance._load_legacy(loaded)
        else:
            instance._load_standard(loaded)

        # fix start time if not stored
        if instance._start_time is None:
            instance._start_time = instance._data['timestamps'][0]

        return instance

    def _load_standard(self, loaded):
        """Internal method for loading current format."""
        for key in loaded.files:
            if key.startswith('meta_'):
                meta_key = key.replace('meta_', '')
                val = loaded[key]
                self.metadata[meta_key] = val.item() if val.ndim == 0 else val
            else:
                # History Data
                data_key = key.replace('_history', '')
                self._data[data_key] = list(loaded[key])

        if 'start_time' in self.metadata:
            self._start_time = self.metadata['start_time']

    def _load_legacy(self, loaded):
        """Internal method for loading old format."""
        self.metadata['message'] = 'Loaded from legacy file'
        self.metadata['algorithm'] = 'Legacy (Unknown)'

        # Infer algorithm and map keys
        keys = loaded.files

        # 1. Map standard keys (present in both old Grad and CMAES)
        if 'x_history' in keys:
            raw_x = loaded['x_history']
            # If it was Gradient (2D), reshape. If CMAES (3D), keep.
            # But wait, old CMAES was saved as 3D (gen, pop, param)?
            # Let's check dimensions carefully.

            # Old Grad: (steps, params) -> Need (steps, 1, params)
            # Old CMA: (gens, pop, params) -> Keep

            if raw_x.ndim == 2:
                self._data['x'] = list(raw_x[:, np.newaxis, :])
            else:
                self._data['x'] = list(raw_x)

        # Load R
        if 'R_history' in keys:
            raw_R = loaded['R_history']
            # Old Grad: (steps,) -> Need (steps, 1)
            # Old CMA: (gens, pop) -> Keep

            if raw_R.ndim == 1:
                self._data['R'] = list(raw_R[:, np.newaxis])
            else:
                self._data['R'] = list(raw_R)

        if 'timestamp_history' in keys:
            self._data['timestamps'] = list(loaded['timestamp_history'])

        # Extended Physics Params (Assume these follow X structure roughly)
        for k in [
            'v0r_offsets',
            'geo_displacements',
            'vibration_amplitudes',
            'occupations',
        ]:
            if k in keys:
                self._data[k] = list(loaded[k])

        # Detect Algorithm
        if 'step_size_history' in keys:
            self.metadata['algorithm'] = 'Legacy CMA-ES'
            self._data['step_size'] = list(loaded['step_size_history'])
            self.metadata['convergence_generations'] = None
        elif 'grad_R_history' in keys:
            self.metadata['algorithm'] = 'Legacy Gradient'
            self._data['grad_R'] = list(loaded['grad_R_history'])

    def __repr__(self):
        """Return string representation."""
        algo = self.metadata.get('algorithm', 'Generic')
        return f'OptimizationHistory ({algo}) R:   {self.best_R:.4f}\n'
