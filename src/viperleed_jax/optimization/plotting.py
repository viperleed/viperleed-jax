"""Module plotting."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-12-18'
__license__ = 'GPLv3+'


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from viperleed_jax.optimization.history import (
    OptimizationHistory,
    RefCalcHistory,
)

DEFAULT_COLORS = {
    'ref_calc': 'tab:blue',
    'opt_running_min': 'tab:blue',
    'opt_evals_single': 'tab:red',
    'opt_evals_multiple': 'tab:orange',
}

DEFAULT_PLOT_OPTIONS = {
    'x_scale': 'linear',
    'y_scale': 'sqrt',
    'running_min_overall': True,
}


def draw_trajectory_rfactor(
    trajectory, axis=None, options=DEFAULT_PLOT_OPTIONS, colors=DEFAULT_COLORS
):
    """Draw R-factor progress over a calculation trajectory.

    Parameters
    ----------
    trajectory : CalcTrajectory
        The calculation trajectory containing segments of optimization and
        reference calculations.
    axis : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis are created.
    options : dict, optional
        Plotting options, including 'x_scale', 'y_scale', and
        'running_min_overall'.
    colors : dict, optional
        Colors for different plot elements.

    Returns
    -------
    matplotlib.axes.Axes
        The axis with the plotted R-factor progress.
    """
    if axis is not None:
        ax = axis
    else:
        _, ax = plt.subplots()

    cum_time = 0.0
    overall_running_min = np.inf

    min_R, max_R = 2.0, 0.0

    for segment in trajectory.segments:
        if isinstance(segment, RefCalcHistory):
            ax.vlines(
                cum_time,
                ymin=0.0,
                ymax=5.0,
                colors=colors['ref_calc'],
                linestyles='dashed',
                label='Ref Calc' if cum_time == 0.0 else '',
                zorder=9,
            )
            ax.scatter(
                cum_time,
                segment.ref_R,
                marker='o',
                s=40,
                color=colors['ref_calc'],
                zorder=10,
            )

            # set running_min to ref_R
            overall_running_min = segment.ref_R

            # update min and max R for axis limits
            min_R = min(min_R, segment.ref_R)
            max_R = max(max_R, segment.ref_R)

        if isinstance(segment, OptimizationHistory):
            times = segment.relative_times + cum_time
            # plot running min
            running_min = segment.R_running_min
            if options['running_min_overall']:
                combined = np.concatenate(
                    (np.array([overall_running_min]), running_min)
                )
                running_min = np.minimum.accumulate(combined)[1:]
                overall_running_min = running_min[-1]
            ax.plot(times, running_min, '-', color=colors['opt_running_min'])
            # update min and max R for axis limits
            min_R = min(min_R, np.min(running_min))
            max_R = max(max_R, np.max(running_min))

            # scatter all evaluations
            ax.set_autoscale_on(False)
            times_repeat = np.repeat(times, segment.R_history.shape[1])
            # if there are multiple evals per time, use alpha 0.05, else 0.2
            color = (
                colors['opt_evals_multiple']
                if segment.R_history.shape[1] > 1
                else colors['opt_evals_single']
            )
            alpha = 0.05 if segment.R_history.shape[1] > 1 else 0.2
            ax.scatter(
                times_repeat, segment.R_history, alpha=alpha, color=color
            )
            ax.set_autoscale_on(True)

            cum_time += segment.duration

    # set R factor axis scale
    if options['y_scale'] == 'log':
        ax.set_yscale('log')
    elif options['y_scale'] == 'sqrt':
        ax.set_yscale(matplotlib.scale.FuncScale(ax, (_f_sqrt, _f_inv_func)))
    elif options['y_scale'] == 'linear':
        ax.set_yscale('linear')
    else:
        msg = f'Unknown scale "{options["y_scale"]}".'
        raise ValueError(msg)

    # set time axis scale
    if options['x_scale'] == 'log':
        ax.set_xscale('log')
    elif options['x_scale'] == 'sqrt':
        ax.set_xscale(matplotlib.scale.FuncScale(ax, (_f_sqrt, _f_inv_func)))
    elif options['x_scale'] == 'linear':
        ax.set_xscale('linear')
    else:
        msg = f'Unknown scale "{options["x_scale"]}".'
        raise ValueError(msg)

    # axis labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$R_P$')

    # set axis limits
    y_margin = 0.1 * (max_R - min_R)
    ax.set_ylim(max(0.0, min_R - y_margin), max_R + y_margin)

    return ax


# square root scale and its inverse for axes
def _f_sqrt(e):
    return np.abs(np.sqrt(e + 0j)) * np.sign(e)


def _f_inv_func(e):
    return abs(e**2)
