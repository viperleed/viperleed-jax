"""Module optimization.history."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-03-11'

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
        """
        # SAFETY 1: Copy x to prevent reference issues if solver reuses memory
        self._data['x'].append(np.array(x, copy=True))

        # SAFETY 2: Ensure R is a float (NaN) if None is passed
        # This guarantees R_history remains a float array, not object array
        val_R = np.nan if R is None else R
        self._data['R'].append(val_R)
        self._data['timestamps'].append(time.time())

        # SAFETY 3: Handle None in kwargs (like grad_R=None)
        # If grad_R is None, we store a generic NaN placeholder matching x's shape
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
    def relative_times(self):
        return np.array(
            timestamp - self._start_time
            for timestamp in self._data['timestamps']
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

        # If CMA-ES (indicated by convergence_generations in metadata)
        if 'convergence_generations' in self.metadata:
            n = self.metadata['convergence_generations']
            # If n is None or 0, use full history
            if not n:
                return np.min(R)
            return np.min(R[-n:])

        # Default / Gradient / Legacy
        # Use last non-nan value
        non_nan = R[~np.isnan(R)]
        return non_nan[-1] if non_nan.size > 0 else np.nan

    @property
    def best_x(self):
        """Smart retrieval of best x based on stored metadata."""
        R = self.R_history
        x = self.x_history

        if 'convergence_generations' in self.metadata:
            n = self.metadata['convergence_generations']
            if not n:
                trunc_R = R
                trunc_x = x
            else:
                trunc_R = R[-n:]
                trunc_x = x[-n:]

            min_idx = np.unravel_index(np.argmin(trunc_R), trunc_R.shape)
            return trunc_x[min_idx]

        return x[-1]

    def expand_parameters(self, calculator):
        """
        Post-processing: Expand stored parameters into physics quantities.
        Stores results directly into self._data.
        """
        # Prepare storage
        for key in [
            'v0r_offsets',
            'geo_displacements',
            'vibration_amplitudes',
            'occupations',
        ]:
            self._data[key] = []

        x_arr = self.x_history
        # Check if x_history is 3D (generations, population, params) or 2D (steps, params)
        is_population = x_arr.ndim == 3

        # Helper to process one vector and append to lists
        def _process_one(vec):
            v0r, geo, vib, occ = calculator.expand_params(vec)
            self._data['v0r_offsets'].append(v0r)
            self._data['geo_displacements'].append(geo)
            self._data['vibration_amplitudes'].append(vib)
            self._data['occupations'].append(occ)

        if is_population:
            n_gens, pop_size, _ = x_arr.shape
            for g in range(n_gens):
                for p in range(pop_size):
                    _process_one(x_arr[g, p])
        else:
            for vec in x_arr:
                _process_one(vec)

    # --- I/O Phase ---

    def write_to_file(self, file_path):
        """Save everything to .npz."""
        save_dict = {}

        # 1. Save data history arrays
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
                # Metadata
                meta_key = key.replace('meta_', '')
                val = loaded[key]
                self.metadata[meta_key] = val.item() if val.ndim == 0 else val
            else:
                # History Data
                data_key = key.replace('_history', '')
                self._data[data_key] = list(loaded[key])

    def _load_legacy(self, loaded):
        """Internal method for loading old format (pre-unification)."""
        self.metadata['message'] = 'Loaded from legacy file'
        self.metadata['algorithm'] = 'Legacy (Unknown)'

        # Infer algorithm and map keys
        keys = loaded.files

        # 1. Map standard keys (present in both old Grad and CMAES)
        if 'x_history' in keys:
            self._data['x'] = list(loaded['x_history'])
        if 'R_history' in keys:
            self._data['R'] = list(loaded['R_history'])
        if 'timestamp_history' in keys:
            self._data['timestamps'] = list(loaded['timestamp_history'])

        # 2. Map Extended Physics Params (if present)
        legacy_physics_keys = [
            'v0r_offsets',
            'geo_displacements',
            'vibration_amplitudes',
            'occupations',
        ]
        for k in legacy_physics_keys:
            if k in keys:
                self._data[k] = list(loaded[k])

        # 3. Detect Algorithm Specifics
        if 'step_size_history' in keys:
            # It was CMA-ES
            self.metadata['algorithm'] = 'Legacy CMA-ES'
            self._data['step_size'] = list(loaded['step_size_history'])
            # Old CMAES didn't save convergence_generations,
            # so we set it to None (implies full history) to avoid index errors.
            self.metadata['convergence_generations'] = None

        elif 'grad_R_history' in keys:
            # It was Gradient Descent
            self.metadata['algorithm'] = 'Legacy Gradient'
            self._data['grad_R'] = list(loaded['grad_R_history'])

    def __repr__(self):
        algo = self.metadata.get('algorithm', 'Generic')
        msg = self.metadata.get('message', 'In Progress...')
        steps = len(self._data['R'])
        return (
            f'OptimizationHistory ({algo})\n'
            f'--------------------------\n'
            f'Status:   {msg}\n'
            f'Steps:    {steps}\n'
            f'Duration: {self.duration:.2f}s\n'
            f'Best R:   {self.best_R:.4f}'
        )
