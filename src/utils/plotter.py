"""Convergence, statistical and every type of visual-based plots.
"""

import opytimizer.visualization.convergence as c
import statys.plotters.significance as s


def plot_wilcoxon_report(report, color_map='YlOrRd', labels=None):
    """Plots h-indexes and p-values from Wilcoxon-based tests.

    Args:
        report (dict): A dictionary-based report holding h-index and p-value tuples.
        color_map (str): A color map from matplotlib.
        labels (list): List of stringed labels.

    """

    # Plots the h-indexes
    s.plot_h_index(report, color_map=color_map, labels=labels,
                   title='Wilcoxon Signed-Rank Test ($h$-indexes)')

    # Plots the p-values
    s.plot_p_value(report, color_map=color_map, labels=labels,
                   title='Wilcoxon Signed-Rank Test ($p$-values)')


def plot_args_convergence(*args, labels=None, title=None, subtitle=None, xlabel='iteration', ylabel='value'):
    """Plots the convergence of a set of arguments.

    Args:
        labels (list): List of defined labels.
        title (str): Title of the plot.
        subtitle (str): Subtitle of the plot.
        xlabel (str): Axis `x` label.
        ylabel (str): Axis `y` label.

    """

    # Plots the desired arguments
    c.plot(*args, labels=labels, title=title, subtitle=subtitle, xlabel=xlabel, ylabel=ylabel)
