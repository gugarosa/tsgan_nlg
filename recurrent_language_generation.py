import argparse

import generation.recurrent_models as r
import utils.pickler as p


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Generates language with recurrent-based models.')

    parser.add_argument('model', help='Type of model', choices=['gru', 'lstm', 'rmc', 'rnn'])

    parser.add_argument('n_tokens', help='How many tokens to be used as start string', type=int, default=3)

    parser.add_argument('-embedding_size', help='Number of embedding units', type=int, default=256)

    parser.add_argument('-hidden_size', help='Number of hidden units', type=int, default=512)

    parser.add_argument('-n_slots', help='Number of RMC slots', type=int, default=5)

    parser.add_argument('-n_heads', help='Number of RMC heads', type=int, default=5)

    parser.add_argument('-head_size', help='Size of RMC head', type=int, default=25)

    parser.add_argument('-n_blocks', help='Number of RMC blocks', type=int, default=1)

    parser.add_argument('-n_layers', help='Number of RMC layers', type=int, default=3)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Common-based arguments
    n_tokens = args.n_tokens

    # Model-based arguments
    model_name = args.model
    model_obj = r.get_recurrent_model(model_name).obj
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size

    # RMC-based arguments
    n_slots = args.n_slots
    n_heads = args.n_heads
    head_size = args.head_size
    n_blocks = args.n_blocks
    n_layers = args.n_layers

    # Loads pre-pickled objects
    corpus = p.load_from_file(f'outputs/{model_name}_corpus.pkl')
    encoder = p.load_from_file(f'outputs/{model_name}_encoder.pkl')
    enc_test = p.load_from_file(f'outputs/{model_name}_enc_test.pkl')

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

    # Loads weights and builds model
    model.load_weights(f'outputs/{model_name}').expect_partial()
    model.build((1, None))

    # Iterates over every sentence in testing set
    for token in enc_test:
        # Decodes the token back to words
        decoded_token = encoder.decode(token)

        # Gathers `n` tokens as the starting token
        start_token = decoded_token[:n_tokens]

        #
        print(model.generate_top_sampling(start=decoded_token, max_length=len(decoded_token)))
