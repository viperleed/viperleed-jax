import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from viperleed_jax.optimization.history import (
    OptimizationHistory,
    RefCalcHistory,
)


def make_plot(trajectory, y_scale='sqrt', x_scale='linear'):
    fig, ax = plt.subplots()

    cum_time = 0.0

    for segment in trajectory.segments:
        if isinstance(segment, RefCalcHistory):
            ax.scatter(cum_time, segment.ref_R, marker='o', s=30, color='blue')

        if isinstance(segment, OptimizationHistory):
            times = segment.relative_times + cum_time
            # plot running min
            running_min = segment.R_running_min
            plt.plot(times, running_min, '-', color='blue')

            # scatter all evaluations
            times_repeat = np.repeat(times, segment.R_history.shape[1])
            ax.scatter(
                times_repeat, segment.R_history, alpha=0.1, color='orange'
            )

            cum_time += segment.duration

    # set R factor axis scale
    if y_scale == 'log':
        ax.set_yscale('log')
    elif y_scale == 'sqrt':
        ax.set_yscale(matplotlib.scale.FuncScale(ax, (_f_sqrt, _f_inv_func)))
    elif y_scale == 'linear':
        ax.set_yscale('linear')
    else:
        msg = f'Unknown scale "{y_scale}".'
        raise ValueError(msg)

    # set time axis scale
    if x_scale == 'log':
        ax.set_xscale('log')
    elif x_scale == 'sqrt':
        ax.set_xscale(matplotlib.scale.FuncScale(ax, (_f_sqrt, _f_inv_func)))
    elif x_scale == 'linear':
        ax.set_xscale('linear')
    else:
        msg = f'Unknown scale "{y_scale}".'
        raise ValueError(msg)

    # axis labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('R-factor')

    return fig, ax


# sqaure root scale and its inverse for axes
def _f_sqrt(e):
    return np.abs(np.sqrt(e + 0j)) * np.sign(e)


def _f_inv_func(e):
    return abs(e**2)
