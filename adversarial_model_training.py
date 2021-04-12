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

    parser = argparse.ArgumentParser(usage='Trains adversarial-based models over NLG tasks.')

    parser.add_argument('model', help='Type of model', choices=['gsgan', 'maligan', 'relgan', 'seqgan'])

    parser.add_argument('dataset', help='Dataset', choices=['amazon_customer_reviews', 'coco_image_captions',
                                                            'google_one_billion_words', 'wmt_emnlp17_news'])

    parser.add_argument('-train_split', help='Percentage of the training set', type=float, default=0.8)

    parser.add_argument('-val_split', help='Percentage of the validation set', type=float, default=0.1)

    parser.add_argument('-test_split', help='Percentage of the testing set', type=float, default=0.1)

    parser.add_argument('-min_frequency', help='Minimum frequency of tokens', type=int, default=1)

    parser.add_argument('-max_pad_length', help='Maximum pad length of tokens', type=int, default=10)

    parser.add_argument('-embedding_size', help='Number of embedding units', type=int, default=256)

    parser.add_argument('-hidden_size', help='Number of hidden units', type=int, default=512)

    parser.add_argument('-tau', help='Temperature', type=float, default=5)

    parser.add_argument('-n_filters', help='Number of filters', type=int, nargs='+', default=[64, 128, 256])

    parser.add_argument('-filters_size', help='Filters size', type=int, nargs='+', default=[3, 5, 5])

    parser.add_argument('-dropout', help='Dropout rate', type=float, default=0.25)

    parser.add_argument('-n_slots', help='Number of RMC slots', type=int, default=5)

    parser.add_argument('-n_heads', help='Number of RMC heads', type=int, default=5)

    parser.add_argument('-head_size', help='Size of RMC head', type=int, default=25)

    parser.add_argument('-n_blocks', help='Number of RMC blocks', type=int, default=1)

    parser.add_argument('-n_layers', help='Number of RMC layers', type=int, default=3)

    parser.add_argument('-batch_size', help='Size of batches', type=int, default=4)

    parser.add_argument('-pre_lr', help='Pre-training learning rate', type=float, default=0.01)

    parser.add_argument('-d_lr', help='Discriminator learning rate', type=float, default=0.001)

    parser.add_argument('-g_lr', help='Generator learning rate', type=float, default=0.001)

    parser.add_argument('-pre_epochs', help='Amount of pre-training epochs', type=int, default=10)

    parser.add_argument('-pre_d_epochs', help='Amount of pre-training discriminator epochs', type=int, default=10)

    parser.add_argument('-pre_g_epochs', help='Amount of pre-training generator epochs', type=int, default=10)

    parser.add_argument('-epochs', help='Amount of training epochs', type=int, default=10)

    parser.add_argument('-d_epochs', help='Amount of discriminator training epochs', type=int, default=10)

    parser.add_argument('-n_rollouts', help='Number of rollouts', type=int, default=16)

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
    tau = args.tau
    n_filters = tuple(args.n_filters)
    filters_size = tuple(args.filters_size)
    dropout = args.dropout
    batch_size = args.batch_size
    pre_lr = args.pre_lr
    d_lr = args.d_lr
    g_lr = args.g_lr
    pre_epochs = args.pre_epochs
    pre_d_epochs = args.pre_d_epochs
    pre_g_epochs = args.pre_g_epochs
    epochs = args.epochs
    d_epochs = args.d_epochs
    n_rollouts = args.n_rollouts
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

    # Checks if supplied model is a GSGAN
    if model_name == 'gsgan':
        # Instantiates the model
        model = model_obj(encoder=encoder, vocab_size=corpus.vocab_size,
                          embedding_size=embedding_size, hidden_size=hidden_size,
                          tau=tau)

    # Checks if supplied model is a MaliGAN
    elif model_name == 'maligan':
        # Instantiates the model
        model = model_obj(encoder=encoder, vocab_size=corpus.vocab_size, max_length=max_pad_length,
                          embedding_size=embedding_size, hidden_size=hidden_size, n_filters=n_filters,
                          filters_size=filters_size, dropout_rate=dropout, temperature=tau)

    # Checks if supplied model is a RelGAN
    elif model_name == 'relgan':
        # Instantiates the model
        model = model_obj(encoder=encoder, vocab_size=corpus.vocab_size, max_length=max_pad_length,
                          embedding_size=embedding_size, n_slots=n_slots, n_heads=n_heads,
                          head_size=head_size, n_blocks=n_blocks, n_layers=n_layers, n_filters=n_filters,
                          filters_size=filters_size, dropout_rate=dropout, tau=tau)

    # Checks if supplied model is a SeqGAN
    elif model_name == 'seqgan':
        # Instantiates the model
        model = model_obj(encoder=encoder, vocab_size=corpus.vocab_size, max_length=max_pad_length,
                          embedding_size=embedding_size, hidden_size=hidden_size, n_filters=n_filters,
                          filters_size=filters_size, dropout_rate=dropout, temperature=tau)

    # Compiles the model
    model.compile(pre_optimizer=tf.optimizers.Adam(learning_rate=pre_lr),
                  d_optimizer=tf.optimizers.Adam(learning_rate=d_lr),
                  g_optimizer=tf.optimizers.Adam(learning_rate=g_lr))

    # Checks if supplied model is a GSGAN or RelGAN
    if model_name in ['gsgan', 'relgan']:
        # Pre-fits the model
        model.pre_fit(train.batches, epochs=pre_epochs)

        # Fits the model
        model.fit(train.batches, epochs=epochs)

    # Checks if supplied model is a MaliGAN
    elif model_name == 'maligan':
        # Pre-fits the model
        model.pre_fit(train.batches, g_epochs=pre_g_epochs, d_epochs=pre_d_epochs)

        # Fits the model
        model.fit(train.batches, epochs=epochs, d_epochs=d_epochs)
    
    # Checks if supplied model is a SeqGAN
    elif model_name == 'seqgan':
        # Pre-fits the model
        model.pre_fit(train.batches, g_epochs=pre_g_epochs, d_epochs=pre_d_epochs)

        # Fits the model
        model.fit(train.batches, epochs=epochs, d_epochs=d_epochs, n_rollouts=n_rollouts)
    
    # Saves model and objects to files
    model.save_weights(output_path, save_format='tf')
    p.save_to_file(output_path, corpus=corpus, encoder=encoder, enc_test=enc_test)
