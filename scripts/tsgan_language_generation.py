# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse

import pandas as pd

import generation.adversarial_models as a
import utils.pickler as p


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Generates language with TSGAN-based models.')

    parser.add_argument('model', help='Type of model', choices=['tsgan_contrastive', 'tsgan_entropy', 'tsgan_triplet'])

    parser.add_argument('n_tokens', help='How many tokens to be used as start string', type=int, default=3)

    parser.add_argument('-temp', help='Temperature sampling', type=float, default=0.5)

    parser.add_argument('-top_k', help='Amount of `k` for top-k sampling', type=int, default=50)

    parser.add_argument('-top_p', help='Probability for nucleus sampling', type=float, default=0.95)

    parser.add_argument('-max_pad_length', help='Maximum pad length of tokens', type=int, default=10)

    parser.add_argument('-embedding_size', help='Number of embedding units', type=int, default=256)

    parser.add_argument('-hidden_size', help='Number of hidden units', type=int, default=512)

    parser.add_argument('-triplet_loss_type', help='Triplet loss type', type=str, default='hard')

    parser.add_argument('-distance_metric', help='Distance metric', type=str, default='L2')

    parser.add_argument('-tau', help='Temperature', type=float, default=0.5)

    parser.add_argument('-n_pairs', help='Number of data pairs', type=int, default=25)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    n_tokens = args.n_tokens
    temp = args.temp
    top_k = args.top_k
    top_p = args.top_p

    # Model-based arguments
    model_name = args.model
    model_obj = a.get_adversarial_model(model_name).obj
    max_pad_length = args.max_pad_length
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    triplet_loss_type = args.triplet_loss_type
    distance_metric = args.distance_metric
    tau = args.tau
    n_pairs = args.n_pairs
    output_path = f'outputs/{model_name}_nlg.csv'

    # Loads pre-pickled objects
    corpus = p.load_from_file(f'outputs/{model_name}_corpus.pkl')
    encoder = p.load_from_file(f'outputs/{model_name}_encoder.pkl')
    enc_test = p.load_from_file(f'outputs/{model_name}_enc_test.pkl')

    # Checks if supplied model is a TSGAN with Contrastive Loss
    if model_name == 'tsgan_contrastive':
        # Instantiates the model
        model = model_obj(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=embedding_size,
                          hidden_size=hidden_size, distance_metric=distance_metric, temperature=tau,
                          n_pairs=n_pairs)

    # Checks if supplied model is a TSGAN with Cross-Entropy Loss
    elif model_name == 'tsgan_entropy':
        # Instantiates the model
        model = model_obj(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=embedding_size,
                          hidden_size=hidden_size, distance_metric=distance_metric, temperature=tau,
                          n_pairs=n_pairs)

    # Checks if supplied model is a TSGAN with Triplet Loss
    elif model_name == 'tsgan_triplet':
        # Instantiates the model
        model = model_obj(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=embedding_size,
                          hidden_size=hidden_size, loss=triplet_loss_type, distance_metric=distance_metric,
                          temperature=tau)

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
        greedy_token = model.G.generate_greedy_search(start=start_token, max_length=len(decoded_token)-n_tokens)
        temp_token = model.G.generate_temperature_sampling(start=start_token, max_length=len(decoded_token)-n_tokens,
                                                         temperature=temp)
        top_token = model.G.generate_top_sampling(start=start_token, max_length=len(decoded_token)-n_tokens,
                                                k=top_k, p=top_p)

        # Appends the outputs to lists
        tokens.append(' '.join(decoded_token))
        greedy_tokens.append(' '.join(start_token + greedy_token))
        temp_tokens.append(' '.join(start_token + temp_token))
        top_tokens.append(' '.join(start_token + top_token))

    # Converts lists to a dataframe
    df = pd.DataFrame({'reference': tokens,
                       'greedy_search': greedy_tokens,
                       'temperature_sampling': temp_tokens,
                       'top_sampling': top_tokens})

    # Saves the dataframe to an output .csv file
    df.to_csv(output_path, index=False)
