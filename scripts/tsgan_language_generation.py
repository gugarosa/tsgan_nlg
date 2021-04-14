# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import csv

import utils.pickler as p
from core import TSGANContrastive, TSGANTriplet

if __name__ == '__main__':
    # Common-based arguments
    n_tokens = 3
    temp = 0.4
    top_k = 25
    top_p = 0.97

    # Model-based arguments
    model_name = 'tsgan'

    # Loads pre-pickled objects
    corpus = p.load_from_file(f'outputs/{model_name}_corpus.pkl')
    encoder = p.load_from_file(f'outputs/{model_name}_encoder.pkl')
    enc_test = p.load_from_file(f'outputs/{model_name}_enc_test.pkl')

    model = TSGANContrastive(encoder=encoder, vocab_size=corpus.vocab_size,
                             embedding_size=64, hidden_size=128, temperature=0.5,
                             n_pairs=25)

    # Loads weights and builds model
    model.load_weights(f'outputs/{model_name}').expect_partial()
    model.G.build((1, None))

    # Instantiates lists for the outputs
    tokens, greedy_tokens, temp_tokens, top_tokens = [], [], [], []

    # Iterates over every sentence in testing set
    for token in enc_test:
        # Decodes the token back to words
        decoded_token = encoder.decode(token)

        # Gathers `n` tokens as the starting token
        start_token = decoded_token[:n_tokens]

        # Generates with three distinct strategies
        greedy_token = model.G.generate_greedy_search(start=decoded_token, max_length=len(decoded_token))
        temp_token = model.G.generate_temperature_sampling(start=decoded_token, max_length=len(decoded_token),
                                                           temperature=temp)
        top_token = model.G.generate_top_sampling(start=decoded_token, max_length=len(decoded_token),
                                                  k=top_k, p=top_p)

        # Appends the outputs to lists
        tokens.append(' '.join(decoded_token))
        greedy_tokens.append(' '.join(start_token + greedy_token))
        temp_tokens.append(' '.join(start_token + temp_token))
        top_tokens.append(' '.join(start_token + top_token))

    # Opens an output .csv file
    with open(f'outputs/{model_name}.csv', 'w') as f:
        # Creates the .csv writer
        writer = csv.writer(f)

        # Dumps the data
        writer.writerow(['reference', 'greedy_search', 'temperature_sampling', 'top_sampling'])
        writer.writerows(zip(tokens, greedy_tokens, temp_tokens, top_tokens))
