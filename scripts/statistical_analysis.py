import os
import sys

# Caveat to allow running from inside or outside scripts package
sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse
import csv


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Statistically analyze a set of .csv metrics files.')

    parser.add_argument('csv_files', help='Input file .csv with generated text', type=str, nargs='+')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    csv_files = args.csv_files

    # Iterates through every possible file
    for csv_file in csv_files:
        # Opens the .csv file
        with open(csv_file, 'r') as f:
            # Defines the reader object
            reader = csv.DictReader(f)