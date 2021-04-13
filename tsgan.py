import tensorflow as tf
from nalp.corpus import SentenceCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder

import utils.loader as l
from core.tsgan import TSGAN


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
model = TSGAN(encoder=encoder, vocab_size=corpus.vocab_size,
              embedding_size=64, hidden_size=128)

# Compiles the model
model.compile(pre_d_optimizer=tf.optimizers.Adam(learning_rate=0.001),
              pre_g_optimizer=tf.optimizers.Adam(learning_rate=0.001),
              d_optimizer=tf.optimizers.Adam(learning_rate=0.001),
              g_optimizer=tf.optimizers.Adam(learning_rate=0.001))

#
model.pre_fit(train.batches, g_epochs=0, d_epochs=100)
