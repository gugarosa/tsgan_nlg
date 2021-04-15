# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse

import numpy as np
from nalp.corpus import SentenceCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder

import generation.adversarial_models as a
import optimization.heuristics as h
import optimization.target as t
import optimization.wrapper as w
import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Optimizes TSGAN-based models over NLG tasks.')

    parser.add_argument('mh', help='Meta-heuristic identifier', choices=['ba', 'cs', 'fa', 'gp', 'pso'])

    parser.add_argument('model', help='Type of model', choices=['tsgan_contrastive', 'tsgan_triplet'])

    parser.add_argument('dataset', help='Dataset', choices=['amazon_customer_reviews', 'coco_image_captions',
                                                            'google_one_billion_words', 'wmt_emnlp17_news'])

    parser.add_argument('-train_split', help='Percentage of the training set', type=float, default=0.8)

    parser.add_argument('-val_split', help='Percentage of the validation set', type=float, default=0.1)

    parser.add_argument('-test_split', help='Percentage of the testing set', type=float, default=0.1)

    parser.add_argument('-n_tokens', help='How many tokens to be used as start string', type=int, default=3)

    parser.add_argument('-temp', help='Temperature sampling', type=float, default=0.5)

    parser.add_argument('-top_k', help='Amount of `k` for top-k sampling', type=int, default=0)

    parser.add_argument('-top_p', help='Probability for nucleus sampling', type=float, default=1.0)

    parser.add_argument('-min_frequency', help='Minimum frequency of tokens', type=int, default=1)

    parser.add_argument('-max_pad_length', help='Maximum pad length of tokens', type=int, default=10)

    parser.add_argument('-embedding_size', help='Number of embedding units', type=int, default=256)

    parser.add_argument('-hidden_size', help='Number of hidden units', type=int, default=512)

    parser.add_argument('-tau', help='Temperature', type=float, default=0.5)

    parser.add_argument('-n_pairs', help='Number of data pairs', type=int, default=25)

    parser.add_argument('-batch_size', help='Size of batches', type=int, default=4)

    parser.add_argument('-pre_d_lr', help='Pre-training discriminator learning rate', type=float, default=0.001)

    parser.add_argument('-pre_g_lr', help='Pre-training generator learning rate', type=float, default=0.001)

    parser.add_argument('-d_lr', help='Discriminator learning rate', type=float, default=0.001)

    parser.add_argument('-g_lr', help='Generator learning rate', type=float, default=0.001)

    parser.add_argument('-pre_d_epochs', help='Amount of pre-training discriminator epochs', type=int, default=1)

    parser.add_argument('-pre_g_epochs', help='Amount of pre-training generator epochs', type=int, default=1)

    parser.add_argument('-epochs', help='Amount of training epochs', type=int, default=1)

    parser.add_argument('-d_epochs', help='Amount of discriminator training epochs', type=int, default=1)

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=2)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=1)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    dataset = args.dataset
    train_split = args.train_split
    val_split = args.val_split
    test_split = args.test_split
    n_tokens = args.n_tokens
    temp = args.temp
    top_k = args.top_k
    top_p = args.top_p
    min_frequency = args.min_frequency
    max_pad_length = args.max_pad_length
    seed = args.seed

    # Model-based arguments
    model_name = args.model
    model_obj = a.get_adversarial_model(model_name).obj
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    tau = args.tau
    n_pairs = args.n_pairs
    batch_size = args.batch_size
    pre_d_lr = args.pre_d_lr
    pre_g_lr = args.pre_g_lr
    d_lr = args.d_lr
    g_lr = args.g_lr
    pre_d_epochs = args.pre_d_epochs
    pre_g_epochs = args.pre_g_epochs
    epochs = args.epochs
    d_epochs = args.d_epochs

    # Gathering optimization variables
    meta_name = args.mh
    meta_obj = h.get_heuristic(meta_name).obj
    hyperparams = h.get_heuristic(meta_name).hyperparams
    n_agents = args.n_agents
    n_iterations = args.n_iter
    output_path = f'outputs/{meta_name}_{model_name}'

    # Defines numpy seed
    np.random.seed(seed)

    # Loads and tokenizes the data
    data = l.load_data(dataset)
    tokens = l.tokenize_data(data)

    # Creates the sentence-based corpus
    corpus = SentenceCorpus(tokens=tokens, min_frequency=min_frequency, max_pad_length=max_pad_length)

    # Initializes, learns the encoder and encodes the data
    encoder = IntegerEncoder()
    encoder.learn(corpus.vocab_index, corpus.index_vocab)
    encoded_tokens = encoder.encode(corpus.tokens)

    # Splits the tokens
    enc_train, enc_val, _ = l.split_data(encoded_tokens, train_split, val_split, test_split, seed)

    # Creates Language Modeling datasets
    train = LanguageModelingDataset(enc_train, batch_size=batch_size)

    # Defines the optimization variables bounds
    n_variables = 1
    lb = [0]
    ub = [1]

    # Initializes the optimization target
    opt_fn = t.fine_tune_tsgan(model_name, model_obj, train, enc_val, encoder, corpus.vocab_size,
                               embedding_size, hidden_size, tau, n_pairs, pre_d_lr,
                               pre_g_lr, d_lr, g_lr, pre_d_epochs, pre_g_epochs, epochs,
                               d_epochs, n_tokens, temp, top_k, top_p)

    # Runs the optimization task
    history = w.start_opt(meta_name, meta_obj, opt_fn, n_agents, n_variables, n_iterations, lb, ub, hyperparams)

    # Saves the history object to an output file
    history.save(output_path + '.pkl')
