# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse

import pandas as pd

import utils.plotter as p
import utils.statistical as s


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Statistically analyzes a set of .csv metrics files.')

    parser.add_argument('csv_files', help='Input .csv files with generated text', type=str, nargs='+')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    csv_files = args.csv_files

    # Reads a set of .csv files and concatenates into a single dataframe
    # Also, group the dataframe by its `metric` column
    df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files])
    df = df.groupby('metric').agg(list).reset_index()

    # Converts entire dataframe to a list
    df = df.values.tolist()

    # Gathers individual metrics (note we are indexing to 1: to remove the metric label)
    # BLEU-1, BLEU-2, BLEU-3, METEOR, ROUGE-1, ROUGE-2, ROUGE-L
    bleu_1, bleu_2, bleu_3 = df[0][1:], df[1][1:], df[2][1:]
    meteor = df[3][1:]
    rouge_1, rouge_2, rouge_L = df[4][1:], df[5][1:], df[6][1:]

    # Calculates the statistical reports
    mean, std = s.measurement_report(*bleu_1, *bleu_2, *bleu_3, *meteor, *rouge_1, *rouge_2, *rouge_L)
    signed_rank, rank_sum = s.wilcoxon_report(*bleu_1, *bleu_2, *bleu_3, *meteor, *rouge_1, *rouge_2, *rouge_L)

    # Plots a Wilcoxon-based report
    p.plot_wilcoxon_report(signed_rank)
