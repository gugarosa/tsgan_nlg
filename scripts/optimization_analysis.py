# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse
import glob

import numpy as np
from natsort import natsorted
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

    parser = argparse.ArgumentParser(usage='Analyzes sets of .pkl optimization history files.')

    parser.add_argument('history_files_stem', help='Input .pkl optimization history files without extension and seed',
                        type=str, nargs='+')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    history_files_stem = args.history_files_stem

    # Checks the folder and creates list of lists of input files
    list_history_files = [natsorted(glob.glob(f'{history_file_stem}*'))
                          for history_file_stem in history_files_stem]

    # Loads sets of optimization history files
    histories = [[load_history_wrapper(history_file) for history_file in history_files]
                for history_files in list_history_files]

    # Gathers best agent's fitnesses
    # Note that we will perform this for every set of input history files
    best_agent_fit = [[h.get(key='best_agent', index=(1,)) for h in history]
                      for history in histories]

    # Peforms a mean calculation over their fitnesses
    mean_best_agent_fit = [np.mean(fit, axis=0) for fit in best_agent_fit]

    # Plots the convergence of fitnesses
    p.plot_args_convergence(*mean_best_agent_fit)
