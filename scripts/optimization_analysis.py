# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse

import numpy as np
from opytimizer.utils.history import History

import utils.plotter as p


def load_history_wrapper(input_file):
    """Wraps optimization history loading into a single method.

    Args:
        input_file (str): File to be loaded.

    Returns:
        An optimization history object.

    """

    # Instantiates object and loads the file
    h = History()
    h.load(input_file)

    return h


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Analyzes a set of .pkl optimization history files.')

    parser.add_argument('history_files', help='Input .pkl files with optimization history', type=str, nargs='+')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    history_files = args.history_files

    # Loads a set of optimization history files
    histories = [load_history_wrapper(history_file) for history_file in history_files]

    # Gathers best agent's position and fitness
    # Note that we will perform this for every input history file
    best_agent_pos = [h.get(key='best_agent', index=(0,)) for h in histories]
    best_agent_fit = [h.get(key='best_agent', index=(1,)) for h in histories]

    # Peforms a mean calculation over the positions and fitnesses
    mean_best_agent_pos = np.mean(best_agent_pos, axis=0)
    mean_best_agent_fit = np.mean(best_agent_fit, axis=0)

    # Plots the convergence of variables or fitness
    p.plot_args_convergence(*mean_best_agent_pos)
    p.plot_args_convergence(mean_best_agent_fit)
