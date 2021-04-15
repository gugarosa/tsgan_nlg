"""Optimization targets.
"""

import tensorflow as tf


def fine_tune_tsgan(model_name, model_obj, train, val, encoder, vocab_size, embedding_size, hidden_size,
                    tau, n_pairs, pre_d_lr, pre_g_lr, d_lr, g_lr, pre_g_epochs, pre_d_epochs,
                    epochs, d_epochs, n_tokens, temp, top_k, top_p):
    """Wraps the fine-tuning procedure of TSGAN-based models.

    """

    def f(w):
        """Fits on training data and validades on validation data.

        Args:
            w (float): Array of variables.

        Returns:
            BLEU score.

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
        for token in val:
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
            tokens.append(' '.join(decoded_token))
            greedy_tokens.append(' '.join(start_token + greedy_token))
            temp_tokens.append(' '.join(start_token + temp_token))
            top_tokens.append(' '.join(start_token + top_token))

        return 1

    return f
