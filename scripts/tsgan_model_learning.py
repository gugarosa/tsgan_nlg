# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import argparse

import tensorflow as tf
from nalp.corpus import SentenceCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder

import generation.adversarial_models as a
import utils.loader as l
import utils.pickler as p


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Trains TSGAN-based models over NLG tasks.')

    parser.add_argument('model', help='Type of model', choices=['tsgan_contrastive', 'tsgan_entropy', 'tsgan_triplet'])

    parser.add_argument('dataset', help='Dataset', choices=['amazon_customer_reviews', 'coco_image_captions',
                                                            'google_one_billion_words', 'wmt_emnlp17_news'])

    parser.add_argument('-train_split', help='Percentage of the training set', type=float, default=0.8)

    parser.add_argument('-val_split', help='Percentage of the validation set', type=float, default=0.1)

    parser.add_argument('-test_split', help='Percentage of the testing set', type=float, default=0.1)

    parser.add_argument('-min_frequency', help='Minimum frequency of tokens', type=int, default=1)

    parser.add_argument('-max_pad_length', help='Maximum pad length of tokens', type=int, default=10)

    parser.add_argument('-embedding_size', help='Number of embedding units', type=int, default=256)

    parser.add_argument('-hidden_size', help='Number of hidden units', type=int, default=512)

    parser.add_argument('-triplet_loss_type', help='Triplet loss type', type=str, default='hard')

    parser.add_argument('-distance_metric', help='Distance metric', type=str, default='L2')

    parser.add_argument('-tau', help='Temperature', type=float, default=0.5)

    parser.add_argument('-n_pairs', help='Number of data pairs', type=int, default=25)

    parser.add_argument('-batch_size', help='Size of batches', type=int, default=4)

    parser.add_argument('-pre_d_lr', help='Pre-training discriminator learning rate', type=float, default=0.001)

    parser.add_argument('-pre_g_lr', help='Pre-training generator learning rate', type=float, default=0.001)

    parser.add_argument('-d_lr', help='Discriminator learning rate', type=float, default=0.001)

    parser.add_argument('-g_lr', help='Generator learning rate', type=float, default=0.001)

    parser.add_argument('-pre_d_epochs', help='Amount of pre-training discriminator epochs', type=int, default=10)

    parser.add_argument('-pre_g_epochs', help='Amount of pre-training generator epochs', type=int, default=10)

    parser.add_argument('-epochs', help='Amount of training epochs', type=int, default=10)

    parser.add_argument('-d_epochs', help='Amount of discriminator training epochs', type=int, default=10)

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
    min_frequency = args.min_frequency
    max_pad_length = args.max_pad_length
    seed = args.seed

    # Model-based arguments
    model_name = args.model
    model_obj = a.get_adversarial_model(model_name).obj
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    triplet_loss_type = args.triplet_loss_type
    distance_metric = args.distance_metric
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
    output_path = f'outputs/{model_name}'

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
    enc_train, _, enc_test = l.split_data(encoded_tokens, train_split, val_split, test_split, seed)

    # Creates Language Modeling datasets
    train = LanguageModelingDataset(enc_train, batch_size=batch_size)

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
                          hidden_size=hidden_size, temperature=tau, n_pairs=n_pairs)

    # Checks if supplied model is a TSGAN with Triplet Loss
    elif model_name == 'tsgan_triplet':
        # Instantiates the model
        model = model_obj(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=embedding_size,
                          hidden_size=hidden_size, loss=triplet_loss_type, distance_metric=distance_metric,
                          temperature=tau)

    # Compiles the model
    model.compile(pre_d_optimizer=tf.optimizers.Adam(learning_rate=pre_d_lr),
                  pre_g_optimizer=tf.optimizers.Adam(learning_rate=pre_g_lr),
                  d_optimizer=tf.optimizers.Adam(learning_rate=d_lr),
                  g_optimizer=tf.optimizers.Adam(learning_rate=g_lr))

    # Pre-fits the model
    model.pre_fit(train.batches, g_epochs=pre_g_epochs, d_epochs=pre_d_epochs)

    # Fits the model
    model.fit(train.batches, epochs=epochs, d_epochs=d_epochs)

    # Saves model and objects to files
    model.save_weights(output_path, save_format='tf')
    p.save_to_file(output_path, train_history=model.history,
                   corpus=corpus, encoder=encoder, enc_test=enc_test)
