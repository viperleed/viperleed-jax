"""Module plotting.cmaes_parameter_scatter"""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-12-19'
__license__ = 'GPLv3+'


import matplotlib.pyplot as plt
import numpy as np

from viperleed_jax.analysis.optimization_history import OptimizationHistory

PARAMETER_PLOT_DEFAULT_OPTIONS = {'cmap': 'viridis', 'density': 'violin'}


def draw_parameters(
    opt_history, axis=None, options=PARAMETER_PLOT_DEFAULT_OPTIONS
):
    """Plot parameter distribution."""
    if axis is not None:
        ax = axis
    else:
        _, ax = plt.subplots(figsize=(15, 8))

    if not isinstance(opt_history, OptimizationHistory):
        msg = f'Expected OptimizationHistory, got {type(opt_history)}'
        raise TypeError(msg)

    # --- 1. Data Preparation ---
    # Shape: (Generations, Pop_Size, Params)
    data = opt_history.x_history
    # Shape: (Generations, Pop_Size)
    Rs = opt_history.R_history

    n_params = data.shape[2]

    # Flatten
    data_flat = data.reshape(-1, n_params)
    rewards_flat = Rs.flatten()

    # --- 2. Normalization (0 to 1) ---
    # Avoid div by zero if a parameter is constant
    min_vals = data_flat.min(axis=0)
    max_vals = data_flat.max(axis=0)
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0

    data_norm = (data_flat - min_vals) / denom

    # --- 3. Sorting ---
    # User logic: Highest R at the "bottom" (drawn first).
    # argsort(-R) -> Descending order (Best to Worst).
    sort_idx = np.argsort(-rewards_flat)
    data_sorted = data_norm[sort_idx]
    rewards_sorted = rewards_flat[sort_idx]

    # --- 4. Plotting with Density Jitter ---
    # Max width for the jitter (0.4 means nearly touching the next param)
    MAX_WIDTH = 0.4

    for i in range(n_params):
        y_vals = data_sorted[:, i]

        if options['density'] == 'violin':
            # -- Density-Dependent Jitter Calculation --
            # We use a histogram to estimate local density fast (kde is slow for N > 10k)
            counts, edges = np.histogram(y_vals, bins=50, density=False)

            # Map each Y-value to its density bin
            # np.digitize returns 1-based indices, so we subtract 1
            bin_indices = np.digitize(y_vals, edges) - 1
            bin_indices = np.clip(bin_indices, 0, len(counts) - 1)

            # Get the density (count) for each point
            local_density = counts[bin_indices]

            # Normalize density to 0-1 range for this specific parameter
            # (Avoid div by zero if single point)
            max_d = local_density.max() if local_density.max() > 0 else 1
            density_norm = local_density / max_d

            # Generate jitter proportional to local density
            # Points in dense areas get wider spread to avoid overlap
            jitter_offsets = (
                np.random.uniform(-MAX_WIDTH, MAX_WIDTH, size=len(y_vals))
                * density_norm
            )

            x_vals = i + jitter_offsets
        elif options['density'] == 'random':
            x_vals = np.random.normal(i, 0.08, size=len(rewards_sorted))
        else:
            msg = f'Unknown density option: {options["density"]}'
            raise ValueError(msg)

        # -- Scatter --
        sc = ax.scatter(
            x_vals,
            y_vals,
            c=rewards_sorted,
            cmap=options['cmap'],
            s=3,
            alpha=0.3,
            rasterized=True,  # avoids long render times for PDFs
        )

    # --- 5. Formatting ---
    ax.set_xticks(range(n_params))
    ax.set_xticklabels([f'p{i}' for i in range(n_params)], fontsize=12)
    ax.set_ylabel(r'Normalized Parameter Values $\tilde{\xi}_i$', fontsize=14)

    # Grid separators
    for i in range(n_params):
        ax.axvline(i + 0.5, color='gray', linestyle=':', alpha=0.3)

    # Axis cleanup
    ax.set_xlim(-0.5, n_params - 0.5)
    ax.set_ylim(-0.05, 1.05)

    # Remove top/right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add Colorbar (attached to the axis provided)
    # Check if figure exists to add colorbar properly
    if axis is None:
        cbar = plt.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label('$R_P$', fontsize=14)
    else:
        # If part of a subplot, usually handled outside,
        # but we can try to attach it if requested.
        pass

    return ax
