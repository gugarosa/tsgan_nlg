# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse

import pandas as pd

import utils.metrics as m


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Calculates a set of metrics from .csv files with generated texts.')

    parser.add_argument('csv_file', help='Input .csv file with generated text', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    csv_file = args.csv_file
    output_csv_file = os.path.splitext(csv_file)[0] + '_metrics.csv'

    # Reads a .csv file into a dataframe
    df = pd.read_csv(csv_file)

    # Gathers each reference and prediction
    meteor_rouge_refs = df['reference'].tolist()
    bleu_refs = [[ref] for ref in meteor_rouge_refs]
    greedy_preds = df['greedy_search'].tolist()
    temp_preds = df['temperature_sampling'].tolist()
    top_preds = df['top_sampling'].tolist()

    # Calculates a set of BLEU-based metrics
    bleu_greedy_1 = m.bleu_score(greedy_preds, bleu_refs, n_grams=1)['bleu']
    bleu_temp_1 = m.bleu_score(temp_preds, bleu_refs, n_grams=1)['bleu']
    bleu_top_1 = m.bleu_score(top_preds, bleu_refs, n_grams=1)['bleu']
    bleu_greedy_2 = m.bleu_score(greedy_preds, bleu_refs, n_grams=2)['bleu']
    bleu_temp_2 = m.bleu_score(temp_preds, bleu_refs, n_grams=2)['bleu']
    bleu_top_2 = m.bleu_score(top_preds, bleu_refs, n_grams=2)['bleu']
    bleu_greedy_3 = m.bleu_score(greedy_preds, bleu_refs, n_grams=3)['bleu']
    bleu_temp_3 = m.bleu_score(temp_preds, bleu_refs, n_grams=3)['bleu']
    bleu_top_3 = m.bleu_score(top_preds, bleu_refs, n_grams=3)['bleu']

    # Calculates a set of METEOR-based metrics
    meteor_greedy = m.meteor_score(greedy_preds, meteor_rouge_refs)['meteor']
    meteor_temp = m.meteor_score(temp_preds, meteor_rouge_refs)['meteor']
    meteor_top = m.meteor_score(top_preds, meteor_rouge_refs)['meteor']

    # Calculates a set of ROUGE-based metrics
    rouge_greedy = m.rouge_score(greedy_preds, meteor_rouge_refs)
    rouge_temp = m.rouge_score(temp_preds, meteor_rouge_refs)
    rouge_top = m.rouge_score(top_preds, meteor_rouge_refs)

    # Converts lists to a dataframe
    df = pd.DataFrame({'metric': ['BLEU-1', 'BLEU-2', 'BLEU-3', 'METEOR', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
                       'greedy_search': [bleu_greedy_1, bleu_greedy_2, bleu_greedy_3, meteor_greedy,
                                        rouge_greedy["rouge1"].mid.fmeasure, rouge_greedy["rouge2"].mid.fmeasure,
                                        rouge_greedy["rougeL"].mid.fmeasure],
                       'temperature_sampling': [bleu_temp_1, bleu_temp_2, bleu_temp_3, meteor_temp,
                                        rouge_temp["rouge1"].mid.fmeasure, rouge_temp["rouge2"].mid.fmeasure,
                                        rouge_temp["rougeL"].mid.fmeasure],
                       'top_sampling': [bleu_top_1, bleu_top_2, bleu_top_3, meteor_top,
                                        rouge_top["rouge1"].mid.fmeasure, rouge_top["rouge2"].mid.fmeasure,
                                        rouge_top["rougeL"].mid.fmeasure]})

    # Saves the dataframe to an output .csv file
    df.to_csv(output_csv_file, index=False)
