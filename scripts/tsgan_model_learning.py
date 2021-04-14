# Caveat to allow running from inside or outside scripts package
import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../src'))

import tensorflow as tf
from nalp.corpus import SentenceCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder

import utils.loader as l
import utils.pickler as p
from core import TSGANContrastive, TSGANTriplet

if __name__ == '__main__':
    # Loads and tokenizes the data
    data = l.load_data('coco_image_captions')
    tokens = l.tokenize_data(data)

    # Creates the sentence-based corpus
    corpus = SentenceCorpus(tokens=tokens, min_frequency=1, max_pad_length=10)

    # Initializes, learns the encoder and encodes the data
    encoder = IntegerEncoder()
    encoder.learn(corpus.vocab_index, corpus.index_vocab)
    encoded_tokens = encoder.encode(corpus.tokens)

    # Splits the tokens
    enc_train, enc_val, enc_test = l.split_data(encoded_tokens, 0.8, 0.1, 0.1, 0)

    # Creates Language Modeling datasets
    train = LanguageModelingDataset(enc_train, batch_size=4)
    val = LanguageModelingDataset(enc_val, batch_size=4)

    #
    model = TSGANContrastive(encoder=encoder, vocab_size=corpus.vocab_size,
                             embedding_size=64, hidden_size=128, temperature=0.5,
                             n_pairs=25)

    # model = TSGANTriplet(encoder=encoder, vocab_size=corpus.vocab_size,
    #                      embedding_size=64, hidden_size=128)

    # Compiles the model
    model.compile(pre_d_optimizer=tf.optimizers.Adam(learning_rate=0.01),
                pre_g_optimizer=tf.optimizers.Adam(learning_rate=0.001),
                d_optimizer=tf.optimizers.Adam(learning_rate=0.001),
                g_optimizer=tf.optimizers.Adam(learning_rate=0.001))

    #
    model.pre_fit(train.batches, g_epochs=100, d_epochs=10)

    #
    model.fit(train.batches, epochs=25, d_epochs=1)

    # Saves model and objects to files
    output_path = 'outputs/tsgan'
    model.save_weights(output_path, save_format='tf')
    p.save_to_file(output_path, train_history=model.history,
                   corpus=corpus, encoder=encoder, enc_test=enc_test)
