"""Optimization targets.
"""

import tensorflow as tf
import utils.metrics as m


def fine_tune_tsgan(model_name, model_obj, train, val_tokens, encoder, vocab_size, embedding_size, hidden_size,
                    tau, n_pairs, pre_d_lr, pre_g_lr, d_lr, g_lr, pre_d_epochs, pre_g_epochs,
                    epochs, d_epochs, n_tokens, temp, top_k, top_p):
    """Wraps the fine-tuning procedure of TSGAN-based models.

    Args:
        model_name (str): Model's identifier.
        model_obj (TSGAN): Model's object.
        train (LanguageModellingDataset): Training dataset.
        val_tokens (list): List of validation tokens.
        encoder (IntegerEncoder): An index to vocabulary encoder for the generator.
        vocab_size (int): The size of the vocabulary for both discriminator and generator.
        embedding_size (int): The size of the embedding layer for both discriminator and generator.
        hidden_size (int): The amount of hidden neurons for both discriminator and generator.
        tau (float): Temperature value to sample the token inside TSGAN.
        n_pairs (int): Number of pairs to feed to the discriminator.
        pre_d_lr (float): Pre-train discriminator learning rate.
        pre_g_lr (float): Pre-train generator learning rate.
        d_lr (float): Discriminator learning rate.
        g_lr (float): Generator learning rate.
        pre_d_epochs (int): Amount of pre-training discriminator epochs.
        pre_g_epochs (int): Amount of pre-training generator epochs.
        epochs (int): Amount of training epochs.
        d_epochs (int): Amount of discriminator training epochs.
        n_tokens (int): How many tokens to be used as start string.
        temp (float): Temperature sampling.
        top_k (int): Amount of `k` for top-k sampling.
        top_p (float): Probability for nucleus sampling.

    """

    def f(w):
        """Fits on training data and validades on validation data.

        Args:
            w (float): Array of variables.

        Returns:
            1 - average of BLEU scores.

        """

        # Checks if supplied model is a TSGAN with Contrastive Loss
        if model_name == 'tsgan_contrastive':
            # Instantiates the model
            model = model_obj(encoder=encoder, vocab_size=vocab_size, embedding_size=embedding_size,
                              hidden_size=hidden_size, temperature=tau, n_pairs=n_pairs)

        # Checks if supplied model is a TSGAN with Triplet Loss
        elif model_name == 'tsgan_triplet':
            # Instantiates the model
            model = model_obj(encoder=encoder, vocab_size=vocab_size, embedding_size=embedding_size,
                              hidden_size=hidden_size, temperature=tau)

        # Compiles the model
        model.compile(pre_d_optimizer=tf.optimizers.Adam(learning_rate=pre_d_lr),
                      pre_g_optimizer=tf.optimizers.Adam(learning_rate=pre_g_lr),
                      d_optimizer=tf.optimizers.Adam(learning_rate=d_lr),
                      g_optimizer=tf.optimizers.Adam(learning_rate=g_lr))

        # Pre-fits the model
        model.pre_fit(train.batches, g_epochs=pre_g_epochs, d_epochs=pre_d_epochs)

        # Fits the model
        model.fit(train.batches, epochs=epochs, d_epochs=d_epochs)

        # Saves temporary weights
        model.save_weights('outputs/temp_opt', save_format='tf')

        # Re-creates the objects
        if model_name == 'tsgan_contrastive':
            # Instantiates the model
            model = model_obj(encoder=encoder, vocab_size=vocab_size, embedding_size=embedding_size,
                              hidden_size=hidden_size, temperature=tau, n_pairs=n_pairs)

        # Checks if supplied model is a TSGAN with Triplet Loss
        elif model_name == 'tsgan_triplet':
            # Instantiates the model
            model = model_obj(encoder=encoder, vocab_size=vocab_size, embedding_size=embedding_size,
                              hidden_size=hidden_size, temperature=tau)

        # Loads the weights and build the generator
        model.load_weights('outputs/temp_opt').expect_partial()
        model.G.build((1, None))

        # Instantiates lists for the outputs
        tokens, greedy_tokens, temp_tokens, top_tokens = [], [], [], []

        # Iterates over every sentence in testing set
        for token in val_tokens:
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
            tokens.append([' '.join(decoded_token)])
            greedy_tokens.append(' '.join(start_token + greedy_token))
            temp_tokens.append(' '.join(start_token + temp_token))
            top_tokens.append(' '.join(start_token + top_token))

        # Calculates BLEU-based metrics
        greedy_bleu = m.bleu_score(greedy_tokens, tokens, n_grams=3)['bleu']
        temp_bleu = m.bleu_score(temp_tokens, tokens, n_grams=3)['bleu']
        top_bleu = m.bleu_score(top_tokens, tokens, n_grams=3)['bleu']

        return 1 - (greedy_bleu + temp_bleu + top_bleu) / 3

    return f
