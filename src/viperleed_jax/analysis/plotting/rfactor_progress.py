"""Module plotting.rfactor_progress."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-12-18'
__license__ = 'GPLv3+'


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from viperleed_jax.analysis.optimization_history import OptimizationHistory
from viperleed_jax.analysis.ref_calc_history import RefCalcHistory

DEFAULT_STYLES = {
    'ref_calc_marker': {
        'color': 'tab:blue',
        'marker': 'o',
        's': 40,
        'zorder': 10,
        'label': 'Ref Calc',
    },
    'ref_calc_line': {
        'color': 'tab:blue',
        'linestyle': 'dashed',
        'linewidth': 1.0,
        'zorder': 9,
    },
    'opt_running_min': {
        'color': 'tab:blue',
        'linestyle': '-',
        'linewidth': 1.5,
        'label': 'Running Min',
        'marker': '',
    },
    # High transparency for dense clouds
    'opt_evals_multiple': {
        'color': 'tab:orange',
        'alpha': 0.05,
        's': 10,
        'marker': '.',
        'label': 'Evaluations',
    },
    # More opaque for sparse points
    'opt_evals_single': {
        'color': 'tab:red',
        'alpha': 0.2,
        's': 15,
        'marker': '.',
        'label': 'Evaluations',
    },
}

DEFAULT_FONTS = {
    'labelsize': 14,
    'ticksize': 12,
}

DEFAULT_PLOT_OPTIONS = {
    'x_scale': 'linear',
    'y_scale': 'sqrt',
    'running_min_overall': True,
    'draw_vlines': True,
}


def draw_rfactor_progress(
    trajectory,
    axis=None,
    options=DEFAULT_PLOT_OPTIONS,
    styles=DEFAULT_STYLES,
    font_options=DEFAULT_FONTS,
):
    """Draw R-factor progress with customizable styles and fonts."""
    if axis is not None:
        ax = axis
    else:
        _, ax = plt.subplots(figsize=(10, 6))

    cum_time = 0.0
    overall_running_min = np.inf
    min_R, max_R = 2.0, 0.0

    # Merge user styles with defaults to ensure all keys exist
    # (Shallow copy update is usually sufficient here)
    st = DEFAULT_STYLES.copy()
    if styles is not None:
        st.update(styles)

    fonts = DEFAULT_FONTS.copy()
    if font_options is not None:
        fonts.update(font_options)

    for segment in trajectory.segments:
        # --- Reference Calculation ---
        if isinstance(segment, RefCalcHistory):
            if options['draw_vlines']:
                # Unpack kwargs for line
                ax.vlines(cum_time, ymin=0.0, ymax=5.0, **st['ref_calc_line'])

            # Unpack kwargs for marker
            # We filter 'label' here to avoid duplicate legend entries if looped
            marker_kwargs = st['ref_calc_marker'].copy()
            if cum_time > 0:
                marker_kwargs.pop('label', None)

            ax.scatter(cum_time, segment.ref_R, **marker_kwargs)

            overall_running_min = segment.ref_R
            min_R = min(min_R, segment.ref_R)
            max_R = max(max_R, segment.ref_R)

        # --- Optimization History ---
        if isinstance(segment, OptimizationHistory):
            times = segment.relative_times + cum_time

            # 1. Plot Running Min
            running_min = segment.R_running_min
            if options['running_min_overall']:
                combined = np.concatenate(
                    (np.array([overall_running_min]), running_min)
                )
                running_min = np.minimum.accumulate(combined)[1:]
                overall_running_min = running_min[-1]

            ax.plot(times, running_min, **st['opt_running_min'])

            min_R = min(min_R, np.min(running_min))
            max_R = max(max_R, np.max(running_min))

            # 2. Scatter Evaluations
            # Select style based on population size
            if segment.R_history.shape[1] > 1:
                scatter_style = st['opt_evals_multiple']
            else:
                scatter_style = st['opt_evals_single']

            ax.set_autoscale_on(False)
            times_repeat = np.repeat(times, segment.R_history.shape[1])

            ax.scatter(times_repeat, segment.R_history, **scatter_style)
            ax.set_autoscale_on(True)

            cum_time += segment.duration

    # --- Scaling ---
    # (Assuming _f_sqrt and _f_inv_func are defined globally as in your snippet)
    if options['y_scale'] == 'log':
        ax.set_yscale('log')
    elif options['y_scale'] == 'sqrt':
        try:
            ax.set_yscale(
                matplotlib.scale.FuncScale(ax, (_f_sqrt, _f_inv_func))
            )
        except NameError:
            print('Warning: _f_sqrt not defined, falling back to linear.')

    if options['x_scale'] == 'log':
        ax.set_xscale('log')
    elif options['x_scale'] == 'sqrt':
        try:
            ax.set_xscale(
                matplotlib.scale.FuncScale(ax, (_f_sqrt, _f_inv_func))
            )
        except NameError:
            print('Warning: _f_sqrt not defined, falling back to linear.')

    # --- Formatting & Fonts ---
    ax.set_xlabel('Time (s)', fontsize=fonts['labelsize'])
    ax.set_ylabel('$R_P$', fontsize=fonts['labelsize'])

    ax.tick_params(axis='both', which='major', labelsize=fonts['ticksize'])

    # Dynamic Limits
    y_margin = 0.1 * (max_R - min_R)
    ax.set_ylim(max(0.0, min_R - y_margin), max_R + y_margin)

    return ax


# square root scale and its inverse for axes
def _f_sqrt(e):
    return np.abs(np.sqrt(e + 0j)) * np.sign(e)


def _f_inv_func(e):
    return abs(e**2)
