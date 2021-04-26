# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse
import glob

import numpy as np
from natsort import natsorted

import utils.pickler as pck
import utils.plotter as p


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Analyzes a set of .pkl training history files.')

    parser.add_argument('key', help='Key that should be used to plot the analysis', type=str)

    parser.add_argument('history_files_stem', help='Input .pkl training history files without extension and seed',
                        type=str, nargs='+')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    key = args.key
    history_files_stem = args.history_files_stem

    # Checks the folder and creates list of lists of input files
    list_history_files = [natsorted(glob.glob(f'{history_file_stem}*'))
                          for history_file_stem in history_files_stem]

    # Loads sets of training history files
    history_keys = [[pck.load_from_file(history_file)[key] for history_file in history_files]
                    for history_files in list_history_files]

    # Calculates mean and standard deviation of keys
    mean_history_keys = [np.mean(history_key, axis=0) for history_key in history_keys]
    std_history_keys = [np.std(history_key, axis=0) for history_key in history_keys]

    # Plots their convergence
    p.plot_args_convergence(*mean_history_keys)
