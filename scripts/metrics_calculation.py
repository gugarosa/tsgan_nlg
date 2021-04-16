import os
import sys

# Caveat to allow running from inside or outside scripts package
sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse
import csv

import utils.metrics as m


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Calculates a set of metrics from .csv files with generated texts.')

    parser.add_argument('csv_file', help='Input file .csv with generated text', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    csv_file = args.csv_file
    output_csv_file = os.path.splitext(csv_file)[0] + '_metrics.csv'

    # Instantiates a set of lists for references and predictions
    bleu_refs, meteor_rouge_refs, = [], []
    greedy_preds, temp_preds, top_preds = [], [], []

    # Opens the .csv file
    with open(csv_file, 'r') as f:
        # Defines the reader object
        reader = csv.DictReader(f)

        # Iterates over every row
        for row in reader:
            # Appends each row information to its proper list
            bleu_refs.append([row['reference']])
            meteor_rouge_refs.append(row['reference'])
            greedy_preds.append(row['greedy_search'])
            temp_preds.append(row['temperature_sampling'])
            top_preds.append(row['top_sampling'])

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

    # Opens the output .csv file
    with open(output_csv_file, 'w') as f:
        # Creates the .csv writer
        writer = csv.writer(f)

        # Dumps the data
        writer.writerow(['metric', 'greedy_search',
                        'temperature sampling', 'top sampling'])
        writer.writerow(['BLEU-1', bleu_greedy_1, bleu_temp_1, bleu_top_1])
        writer.writerow(['BLEU-2', bleu_greedy_2, bleu_temp_2, bleu_top_2])
        writer.writerow(['BLEU-3', bleu_greedy_3, bleu_temp_3, bleu_top_3])
        writer.writerow(['METEOR', meteor_greedy, meteor_temp, meteor_top])
        writer.writerow(['ROUGE-1', rouge_greedy["rouge1"].mid.fmeasure,
                        rouge_temp["rouge1"].mid.fmeasure, rouge_top["rouge1"].mid.fmeasure])
        writer.writerow(['ROUGE-2', rouge_greedy["rouge2"].mid.fmeasure,
                        rouge_temp["rouge2"].mid.fmeasure, rouge_top["rouge2"].mid.fmeasure])
        writer.writerow(['ROUGE-L', rouge_greedy["rougeL"].mid.fmeasure,
                        rouge_temp["rougeL"].mid.fmeasure, rouge_top["rougeL"].mid.fmeasure])
