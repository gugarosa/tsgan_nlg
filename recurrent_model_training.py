import argparse

import tensorflow as tf

import generation.recurrent_models as r
import utils.loader as l
import utils.pickler as p
from nalp.corpus import SentenceCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Trains recurrent-based models over NLG tasks.')

    parser.add_argument('model', help='Type of model', choices=['gru', 'lstm', 'rmc', 'rnn'])

    parser.add_argument('dataset', help='Dataset', choices=['amazon_customer_reviews', 'coco_image_captions',
                                                            'google_one_billion_words', 'wmt_emnlp17_news'])

    parser.add_argument('-train_split', help='Percentage of the training set', type=float, default=0.8)

    parser.add_argument('-val_split', help='Percentage of the validation set', type=float, default=0.1)

    parser.add_argument('-test_split', help='Percentage of the testing set', type=float, default=0.1)

    parser.add_argument('-min_frequency', help='Minimum frequency of tokens', type=int, default=1)

    parser.add_argument('-max_pad_length', help='Maximum pad length of tokens', type=int, default=10)

    parser.add_argument('-embedding_size', help='Number of embedding units', type=int, default=256)

    parser.add_argument('-hidden_size', help='Number of hidden units', type=int, default=512)

    parser.add_argument('-n_slots', help='Number of RMC slots', type=int, default=5)

    parser.add_argument('-n_heads', help='Number of RMC heads', type=int, default=5)

    parser.add_argument('-head_size', help='Size of RMC head', type=int, default=25)

    parser.add_argument('-n_blocks', help='Number of RMC blocks', type=int, default=1)

    parser.add_argument('-n_layers', help='Number of RMC layers', type=int, default=3)

    parser.add_argument('-batch_size', help='Size of batches', type=int, default=4)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)

    parser.add_argument('-epochs', help='Amount of training epochs', type=int, default=10)

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
    model_obj = r.get_recurrent_model(model_name).obj
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    output_path = f'outputs/{model_name}'

    # RMC-based arguments
    n_slots = args.n_slots
    n_heads = args.n_heads
    head_size = args.head_size
    n_blocks = args.n_blocks
    n_layers = args.n_layers

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
    enc_train, enc_val, enc_test = l.split_data(encoded_tokens, train_split, val_split, test_split, seed)

    # Creates Language Modeling datasets
    train = LanguageModelingDataset(enc_train, batch_size=batch_size)
    val = LanguageModelingDataset(enc_val, batch_size=batch_size)

    # Checks if supplied model is an RMC
    if model_name == 'rmc':
        # Instantiates the model
        model = model_obj(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=embedding_size,
                          n_slots=n_slots, n_heads=n_heads, head_size=head_size,
                          n_blocks=n_blocks, n_layers=n_layers)

    # If is something else
    else:
        # Instantiates the model
        model = model_obj(encoder=encoder, vocab_size=corpus.vocab_size,
                          embedding_size=embedding_size, hidden_size=hidden_size)

    # Builds and compiles the model
    model.build((batch_size, None))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

    # Fits the model and saves its weights
    model.fit(train.batches, epochs=epochs, validation_data=val.batches)
    
    # Saves model and objects to files
    model.save_weights(output_path, save_format='tf')
    p.save_to_file(output_path, corpus=corpus, encoder=encoder, enc_test=enc_test)
