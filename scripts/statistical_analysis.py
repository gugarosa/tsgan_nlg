# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse
import glob

import numpy as np
import pandas as pd
from natsort import natsorted

import utils.plotter as p
import utils.statistical as s


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Statistically analyzes sets of .csv metrics files.')

    parser.add_argument('csv_files_stem', help='Input .csv metric files without extension and seed', type=str, nargs='+')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    csv_files_stem = args.csv_files_stem

    # Checks the folder and creates list of lists of input files
    list_csv_files = [natsorted(glob.glob(f'{csv_file_stem}*'))
                      for csv_file_stem in csv_files_stem]

    # Reads sets of .csv files and concatenates into a list of dataframes
    # Also, group the dataframes by their `metric` column
    dfs = [pd.concat([pd.read_csv(csv_file) for csv_file in csv_files])
           for csv_files in list_csv_files]
    dfs = [df.groupby('metric').agg(list).reset_index() for df in dfs]

    # Converts entire dataframes to lists
    dfs = [df.values.tolist() for df in dfs]

    # Gathers mean of samplings over individual metrics (note we are indexing to 1: to remove the metric label)
    # BLEU-1, BLEU-2, BLEU-3, METEOR, ROUGE-1, ROUGE-2, ROUGE-L
    bleu_1 = [np.mean(df[0][1:], axis=0) for df in dfs]
    bleu_2 = [np.mean(df[1][1:], axis=0) for df in dfs]
    bleu_3 = [np.mean(df[2][1:], axis=0) for df in dfs]
    meteor = [np.mean(df[3][1:], axis=0) for df in dfs]
    rouge_1 = [np.mean(df[4][1:], axis=0) for df in dfs]
    rouge_2 = [np.mean(df[5][1:], axis=0) for df in dfs]
    rouge_L = [np.mean(df[6][1:], axis=0) for df in dfs]

    # Calculates the statistical reports
    mean, std = s.measurement_report(*bleu_1, *bleu_2, *bleu_3, *meteor, *rouge_1, *rouge_2, *rouge_L)
    signed_rank, rank_sum = s.wilcoxon_report(*bleu_1)

    # Plots a Wilcoxon-based report
    p.plot_wilcoxon_report(signed_rank)
