"""Text-Similarity Generative Adversarial Network with Triplet Loss-based Discriminator.
"""

import nalp.utils.logging as l
import tensorflow as tf
from nalp.core import Adversarial
from nalp.models.generators import RNNGenerator, LSTMGenerator, GRUGenerator, RMCGenerator
from tensorflow.keras.utils import Progbar

from core.discriminators import TripletDiscriminator

logger = l.get_logger(__name__)


class TSGANTriplet(Adversarial):
    """A TSGANTriplet class is the one in charge of Text-Similarity Generative Adversarial Networks
    implementation with a triplet loss-based discriminator.

    References:
        Not published yet.

    """

    def __init__(self, encoder=None, vocab_size=1, embedding_size=32, hidden_size=64,
                 n_slots=3, n_heads=5, head_size=10, n_blocks=1, n_layers=3,
                 loss='hard', distance_metric='L2', temperature=0.5):
        """Initialization method.

        Args:
            encoder (IntegerEncoder): An index to vocabulary encoder for the generator.
            vocab_size (int): The size of the vocabulary for both discriminator and generator.
            embedding_size (int): The size of the embedding layer for both discriminator and generator.
            hidden_size (int): The amount of hidden neurons for both discriminator and generator.
            n_slots (int): Number of memory slots.
            n_heads (int): Number of attention heads.
            head_size (int): Size of each attention head.
            n_blocks (int): Number of feed-forward networks.
            n_layers (int): Amout of layers per feed-forward network.
            loss (str): Whether network should use hard or semi-hard negative mining.
            distance_metric (str): Distance metric.
            temperature (float): Temperature value to sample the token.

        """

        logger.info('Overriding class: Adversarial -> TSGANTriplet.')

        # Creating the discriminator network
        D = TripletDiscriminator(vocab_size, embedding_size, hidden_size,
                                 loss=loss, distance_metric=distance_metric)

        # Creating the generator network
        G = LSTMGenerator(encoder, vocab_size, embedding_size, hidden_size)
        # G = RMCGenerator(encoder, vocab_size, embedding_size, n_slots, n_heads, head_size, n_blocks, n_layers)

        # Overrides its parent class with any custom arguments if needed
        super(TSGANTriplet, self).__init__(D, G, name='TSGANTriplet')

        # Defining a property for holding the vocabulary size
        self.vocab_size = vocab_size

        # Temperature
        self.T = temperature

        logger.info('Class overrided.')

    @property
    def vocab_size(self):
        """int: The size of the vocabulary.

        """

        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size):
        self._vocab_size = vocab_size

    @property
    def T(self):
        """float: Temperature value to sample the token.
        """

        return self._T

    @T.setter
    def T(self, T):
        self._T = T

    def compile(self, pre_d_optimizer, pre_g_optimizer, d_optimizer, g_optimizer):
        """Main building method.

        Args:
            pre_d_optimizer (tf.keras.optimizers): An optimizer instance for pre-training the discriminator.
            pre_g_optimizer (tf.keras.optimizers): An optimizer instance for pre-training the generator.
            d_optimizer (tf.keras.optimizers): An optimizer instance for the discriminator.
            g_optimizer (tf.keras.optimizers): An optimizer instance for the generator.

        """

        # Compiles the discriminator with the `pre_d_optimizer`
        self.D.compile(optimizer=pre_d_optimizer)

        # Creates optimizers for pre-training, discriminator and generator
        self.P_optimizer = pre_g_optimizer
        self.D_optimizer = d_optimizer
        self.G_optimizer = g_optimizer

        # Defining the loss function
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits

        # Defining both loss metrics
        self.D_loss = tf.metrics.Mean(name='D_loss')
        self.G_loss = tf.metrics.Mean(name='G_loss')

        # Storing losses as history keys
        self.history['pre_D_loss'] = []
        self.history['pre_G_loss'] = []
        self.history['D_loss'] = []
        self.history['G_loss'] = []

    def generate_batch(self, batch_size=1, length=1):
        """Generates a batch of tokens by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            batch_size (int): Size of the batch to be generated.
            length (int): Length of generated tokens.
            temperature (float): A temperature value to sample the token.

        Returns:
            A (batch_size, length) tensor of generated tokens.

        """

        # Generating an uniform tensor between 0 and vocab_size
        start_batch = tf.random.uniform(
            [batch_size, 1], 0, self.vocab_size, dtype='int32')

        # Copying the sampled batch with the start batch tokens
        sampled_batch = start_batch

        # Resetting the network states
        self.G.reset_states()

        # For every possible generation
        for _ in range(length):
            # Predicts the current token
            preds = self.G(start_batch)

            # Removes the second dimension of the tensor
            preds = tf.squeeze(preds, 1)

            # Regularize the prediction with the temperature
            preds /= self.T

            # Samples a predicted batch
            start_batch = tf.random.categorical(preds, 1, dtype='int32')

            # Concatenates the sampled batch with the predicted batch
            sampled_batch = tf.concat([sampled_batch, start_batch], 1)

        # Ignoring the last column to get the input sampled batch
        x_sampled_batch = sampled_batch[:, :length]

        # Ignoring the first column to get the input sampled batch
        y_sampled_batch = sampled_batch[:, 1:]

        return x_sampled_batch, y_sampled_batch

    def _get_reward(self, x1, x2):
        """Calculates rewards over an input using a Maximum-Likelihood approach.

        Args:
            x1 (tf.Tensor): Tensor containing first samples from input pairs.
            x2 (tf.Tensor): Tensor containing second samples from input pairs.

        """

        # Gathers the batch size and maximum sequence length
        batch_size, max_length = x1.shape[0], x1.shape[1]

        # Calculates the reward mechanism
        rewards = self.D.predict(x1, x2)

        # Normalizes the reward
        rewards = tf.math.divide(rewards, 1 - rewards)
        rewards = tf.math.divide(rewards, tf.math.reduce_sum(rewards))

        # Broadcasts the tensor along the max_length dimensions
        rewards = tf.broadcast_to(tf.expand_dims(rewards, 1), [batch_size, max_length])

        return rewards

    @tf.function
    def G_pre_step(self, x, y):
        """Performs a single batch optimization pre-fitting step over the generator.

        Args:
            x (tf.tensor): A tensor containing the inputs.
            y (tf.tensor): A tensor containing the inputs' labels.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the predictions based on inputs
            preds = self.G(x)

            # Calculate the loss
            loss = tf.reduce_mean(self.loss(y, preds))

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.G.trainable_variables)

        # Apply gradients using an optimizer
        self.P_optimizer.apply_gradients(
            zip(gradients, self.G.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(loss)

    @tf.function
    def G_step(self, x, y, rewards):
        """Performs a single batch optimization step over the generator.

        Args:
            x (tf.tensor): A tensor containing the inputs.
            y (tf.tensor): A tensor containing the inputs' labels.
            rewards (tf.tensor): A tensor containing the rewards for the input.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the predictions based on inputs
            preds = self.G(x)

            # Calculate the loss
            loss = tf.reduce_mean(self.loss(y, preds) * rewards)

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.G.trainable_variables)

        # Apply gradients using an optimizer
        self.G_optimizer.apply_gradients(
            zip(gradients, self.G.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(loss)

    @tf.function
    def D_step(self, x1, x2, y):
        """Performs a single batch optimization step over the discriminator.

        Args:
            x (tf.tensor): A tensor containing the inputs.
            y (tf.tensor): A tensor containing the inputs' labels.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the predictions based on inputs
            preds = tf.expand_dims(self.D.predict(x1, x2), -1)

            # Calculate the loss
            loss = tf.reduce_mean(self.loss(y, preds))

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.D.trainable_variables)

        # Apply gradients using an optimizer
        self.D_optimizer.apply_gradients(
            zip(gradients, self.D.trainable_variables))

        # Updates the discriminator's loss state
        self.D_loss.update_state(loss)

    def pre_fit(self, batches, g_epochs=50, d_epochs=10):
        """Pre-trains the model.

        Args:
            batches (Dataset): Pre-training batches containing samples.
            g_epochs (int): The maximum number of pre-training generator epochs.
            d_epochs (int): The maximum number of pre-training discriminator epochs.

        """

        logger.info('Pre-fitting generator ...')

        # Gathering the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        # Iterate through all generator epochs
        for e in range(g_epochs):
            logger.info('Epoch %d/%d', e+1, g_epochs)

            # Resetting state to further append losses
            self.G_loss.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(G)'])

            # Iterate through all possible pre-training batches
            for x_batch, y_batch in batches:
                # Performs the optimization pre-step over the generator
                self.G_pre_step(x_batch, y_batch)

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(G)', self.G_loss.result())])

            # Dump loss to history
            self.history['pre_G_loss'].append(self.G_loss.result().numpy())

            logger.to_file('Loss(G): %s', self.G_loss.result().numpy())

        logger.info('Pre-fitting discriminator ...')

        # Iterate through all discriminator epochs
        for e in range(d_epochs):
            logger.info('Epoch %d/%d', e+1, d_epochs)

            # Resetting state to further append losses
            self.D.loss_metric.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(D)'])

            # Iterate through all possible pre-training batches
            for x_batch, _ in batches:
                # Gathering the batch size and the maximum sequence length
                batch_size, max_length = x_batch.shape[0], x_batch.shape[1]

                # Generates a batch of fake inputs and concatenates with current batch
                x_fake_batch, _ = self.generate_batch(batch_size, max_length)
                x_concat_batch = tf.concat([x_batch, x_fake_batch], 0)
                y_concat_batch = tf.concat(
                    [tf.zeros(batch_size, dtype='int32'), tf.ones(batch_size, dtype='int32')], 0)

                # Performs the optimization pre-step over the discriminator
                self.D.step(x_concat_batch, y_concat_batch)

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(D)', self.D.loss_metric.result())])

            # Dump loss to history
            self.history['pre_D_loss'].append(self.D.loss_metric.result().numpy())

            logger.to_file('Loss(D): %s', self.D.loss_metric.result().numpy())

    def fit(self, batches, epochs=10, d_epochs=5):
        """Trains the model.

        Args:
            batches (Dataset): Training batches containing samples.
            epochs (int): The maximum number of total training epochs.
            d_epochs (int): The maximum number of discriminator epochs per total epoch.

        """

        logger.info('Fitting model ...')

        # Gathering the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        # Iterate through all epochs
        for e in range(epochs):
            logger.info('Epoch %d/%d', e+1, epochs)

            # Resetting state to further append losses
            self.G_loss.reset_states()
            self.D_loss.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(G)', 'loss(D)'])

            # Iterate through all possible training batches
            for x_batch, _ in batches:
                # Gathering the batch size and the maximum sequence length
                batch_size, max_length = x_batch.shape[0], x_batch.shape[1]

                # Iterate through all possible discriminator's epochs
                for _ in range(d_epochs):
                    # Generates a batch of fake inputs and concatenates
                    x_fake_batch, _ = self.generate_batch(batch_size, max_length)
                    x_concat_batch = tf.concat([x_batch, x_fake_batch], 0)
                    y_concat_batch = tf.concat(
                        [tf.zeros(batch_size, dtype='int32'), tf.ones(batch_size, dtype='int32')], 0)

                    # Performs the optimization step over the discriminator
                    self.D.step(x_concat_batch, y_concat_batch)

                # Generates a batch of fake inputs
                x_fake_batch, y_fake_batch = self.generate_batch(batch_size, max_length)

                # Gathers the rewards based on the sampled batch
                rewards = self._get_reward(x_batch, x_fake_batch)

                # Performs the optimization step over the generator
                self.G_step(x_fake_batch, y_fake_batch, rewards)

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(G)', self.G_loss.result()),
                                 ('loss(D)', self.D.loss_metric.result())])

            # Dumps the losses to history
            self.history['G_loss'].append(self.G_loss.result().numpy())
            self.history['D_loss'].append(self.D.loss_metric.result().numpy())

            logger.to_file('Loss(G): %s | Loss(D): %s', self.G_loss.result().numpy(), self.D.loss_metric.result().numpy())
