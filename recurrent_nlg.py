import argparse

import tensorflow as tf

import utils.loader as l
from nalp.corpus import SentenceCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder
from nalp.models.generators import LSTMGenerator


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Train and evaluates recurrent-based NLG.')

    parser.add_argument('dataset', help='Dataset', choices=['amazon_customer_reviews', 'coco_image_captions'])

    parser.add_argument('-train_split', help='Percentage of the training set', type=float, default=0.8)

    parser.add_argument('-val_split', help='Percentage of the validation set', type=float, default=0.1)

    parser.add_argument('-test_split', help='Percentage of the testing set', type=float, default=0.1)

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
    seed = args.seed

    # Loads and tokenizes the data
    data = l.load_data(dataset)
    tokens = l.tokenize_data(data)

    # Creates the sentence-based corpus
    corpus = SentenceCorpus(tokens=tokens)

    # Initializes, learns the encoder and encodes the data
    encoder = IntegerEncoder()
    encoder.learn(corpus.vocab_index, corpus.index_vocab)
    encoded_tokens = encoder.encode(tokens)

    # Splits the tokens
    enc_train, enc_val, enc_test = l.split_data(encoded_tokens, train_split, val_split, test_split, seed)

    # Creates Language Modeling datasets
    train = LanguageModelingDataset(enc_train, batch_size=8)
    val = LanguageModelingDataset(enc_val, batch_size=8)
    test = LanguageModelingDataset(enc_test, batch_size=8)

    # Creates the recurrent-based generator, builds and compiles it
    lstm = LSTMGenerator(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)
    lstm.build((8, None))
    lstm.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                 loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

    # Fits the generator
    lstm.fit(train.batches, epochs=10, validation_data=val.batches)

    # Evaluates the generator
    lstm.evaluate(test.batches)
