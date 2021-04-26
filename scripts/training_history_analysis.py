# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse

import utils.pickler as pck
import utils.plotter as p


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Analyzes a set of .pkl training history files.')

    parser.add_argument('key', help='Key that should be used to plot the analysis', type=str)

    parser.add_argument('history_files', help='Input .pkl files with training history', type=str, nargs='+')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    key = args.key
    history_files = args.history_files

    # Loads a set of training history files
    history_keys = [pck.load_from_file(history_file)[key] for history_file in history_files]

    # Plots their convergence
    p.plot_args_convergence(*history_keys)
